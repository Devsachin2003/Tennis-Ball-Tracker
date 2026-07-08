"""Temporal player tracking and pose extraction for tennis video.

The tracker in this module wraps Ultralytics YOLO pose tracking and keeps a
lightweight frame-by-frame history keyed by track ID. It is intended for the
validation spike: fast enough to iterate, structured enough to replace later.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Iterable, Sequence

import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    from .court_calibrator import TennisCourtCalibrator
except ImportError:  # Allows direct execution during notebook-style exploration.
    from court_calibrator import TennisCourtCalibrator  # type: ignore


PERSON_CLASS_ID = 0
COCO_POSE_KEYPOINT_COUNT = 17


@dataclass(frozen=True)
class PlayerPoseDetection:
    """Single-frame pose and box output for one tracked person."""

    frame_index: int
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    bbox_confidence: float
    keypoints_xyc: np.ndarray
    court_position_xy: tuple[float, float] | None = None


@dataclass
class PlayerTrajectory:
    """Historical pose track for one player ID over a video chunk."""

    track_id: int
    detections: list[PlayerPoseDetection] = field(default_factory=list)

    @property
    def latest_detection(self) -> PlayerPoseDetection | None:
        """Return the most recent detection for this track."""

        if not self.detections:
            return None
        return self.detections[-1]


class TennisTracker:
    """Extract tracked tennis-player boxes and 17-joint pose keypoints.

    Args:
        model_path: Ultralytics YOLO pose model path/name. ``yolov8n-pose.pt``
            is a good lightweight default for spike validation.
        court_calibrator: Optional calibrated court geometry. When provided,
            detections whose bounding boxes fall entirely outside the court
            polygon are filtered out as likely spectators or officials.
        tracker_config: Ultralytics tracker config. Use ``bytetrack.yaml`` for
            fast validation or ``botsort.yaml`` when appearance cues are useful.
        confidence_threshold: Minimum person detection confidence to retain.
        court_margin_px: Pixel margin added around the court polygon filter to
            avoid dropping players near court edges due to calibration noise.
        device: Optional explicit compute device. When omitted, CUDA is selected
            when available; otherwise CPU is used.
    """

    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        court_calibrator: TennisCourtCalibrator | None = None,
        tracker_config: str = "bytetrack.yaml",
        confidence_threshold: float = 0.25,
        court_margin_px: float = 12.0,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(device) if device is not None else self._select_device()
        self.model = YOLO(model_path)
        self.model.to(str(self.device))
        self.court_calibrator = court_calibrator
        self.tracker_config = tracker_config
        self.confidence_threshold = confidence_threshold
        self.court_margin_px = court_margin_px

        self.frame_index = 0
        self.player_history: DefaultDict[int, list[PlayerPoseDetection]] = defaultdict(list)
        self.keypoint_history_by_track_id: DefaultDict[int, list[np.ndarray]] = defaultdict(list)
        self._transient_track_id = -1

    def process_frame(self, frame: np.ndarray) -> list[PlayerPoseDetection]:
        """Run YOLO pose tracking on one BGR OpenCV frame.

        Args:
            frame: Standard OpenCV BGR image with shape ``H x W x 3``.

        Returns:
            A list of retained player detections for the current frame. Empty
            frames, off-camera players, and filtered spectator detections return
            an empty list instead of raising.
        """

        self._validate_frame(frame)
        self.frame_index += 1

        results = self.model.track(
            source=frame,
            persist=True,
            classes=[PERSON_CLASS_ID],
            tracker=self.tracker_config,
            device=str(self.device),
            verbose=False,
        )

        if not results:
            return []

        current_detections = self._extract_person_detections(results[0])
        for detection in current_detections:
            if detection.track_id >= 0:
                self.player_history[detection.track_id].append(detection)
                self.keypoint_history_by_track_id[detection.track_id].append(
                    detection.keypoints_xyc
                )

        return current_detections

    def get_player_trajectories(self) -> dict[str, object]:
        """Return historical coordinates for near- and far-court players.

        Returns:
            Dictionary with:
            ``by_track_id``:
                All retained track histories keyed by integer track ID.
            ``near_court``:
                The trajectory whose latest court Y position is closest to the
                near baseline. Empty when no retained tracks exist.
            ``far_court``:
                The trajectory whose latest court Y position is closest to the
                far baseline. Empty when no retained tracks exist.
            ``keypoints_by_track_id``:
                Frame-by-frame ``17 x 3`` pose keypoint arrays keyed by track ID.
        """

        trajectories = {
            track_id: PlayerTrajectory(track_id=track_id, detections=list(detections))
            for track_id, detections in self.player_history.items()
            if detections
        }

        near_id, far_id = self._select_near_far_track_ids(trajectories.values())
        return {
            "by_track_id": trajectories,
            "near_court": trajectories.get(near_id) if near_id is not None else None,
            "far_court": trajectories.get(far_id) if far_id is not None else None,
            "keypoints_by_track_id": {
                track_id: list(history)
                for track_id, history in self.keypoint_history_by_track_id.items()
            },
        }

    def reset(self) -> None:
        """Clear temporal state for a new video chunk."""

        self.frame_index = 0
        self.player_history.clear()
        self.keypoint_history_by_track_id.clear()
        self._transient_track_id = -1

    def _extract_person_detections(self, result: object) -> list[PlayerPoseDetection]:
        boxes = getattr(result, "boxes", None)
        keypoints = getattr(result, "keypoints", None)
        if boxes is None or keypoints is None or len(boxes) == 0:
            return []

        xyxy = self._tensor_to_numpy(boxes.xyxy)
        classes = self._tensor_to_numpy(boxes.cls).astype(int)
        confidences = self._tensor_to_numpy(boxes.conf)
        track_ids = self._extract_track_ids(boxes, detection_count=len(xyxy))
        keypoint_data = self._extract_keypoints(keypoints, detection_count=len(xyxy))

        detections: list[PlayerPoseDetection] = []
        for index, bbox in enumerate(xyxy):
            if classes[index] != PERSON_CLASS_ID:
                continue

            confidence = float(confidences[index])
            if confidence < self.confidence_threshold:
                continue

            bbox_tuple = tuple(float(value) for value in bbox)
            if self._is_outside_court_polygon(bbox_tuple):
                continue

            court_position = self._estimate_court_position(bbox_tuple)
            detection = PlayerPoseDetection(
                frame_index=self.frame_index,
                track_id=int(track_ids[index]),
                bbox_xyxy=bbox_tuple,
                bbox_confidence=confidence,
                keypoints_xyc=keypoint_data[index],
                court_position_xy=court_position,
            )
            detections.append(detection)

        return detections

    def _extract_track_ids(self, boxes: object, detection_count: int) -> np.ndarray:
        raw_ids = getattr(boxes, "id", None)
        if raw_ids is None:
            return np.array(
                [self._next_transient_track_id() for _ in range(detection_count)],
                dtype=np.int32,
            )

        ids = self._tensor_to_numpy(raw_ids).astype(np.int32)
        if ids.shape[0] != detection_count:
            return np.array(
                [self._next_transient_track_id() for _ in range(detection_count)],
                dtype=np.int32,
            )

        return ids

    def _extract_keypoints(self, keypoints: object, detection_count: int) -> np.ndarray:
        data = getattr(keypoints, "data", None)
        if data is None:
            return np.zeros((detection_count, COCO_POSE_KEYPOINT_COUNT, 3), dtype=np.float32)

        keypoint_array = self._tensor_to_numpy(data).astype(np.float32)
        if keypoint_array.ndim != 3 or keypoint_array.shape[0] != detection_count:
            return np.zeros((detection_count, COCO_POSE_KEYPOINT_COUNT, 3), dtype=np.float32)

        if keypoint_array.shape[2] == 2:
            confidence = np.ones((*keypoint_array.shape[:2], 1), dtype=np.float32)
            keypoint_array = np.concatenate([keypoint_array, confidence], axis=2)

        if keypoint_array.shape[1] != COCO_POSE_KEYPOINT_COUNT or keypoint_array.shape[2] != 3:
            padded = np.zeros((detection_count, COCO_POSE_KEYPOINT_COUNT, 3), dtype=np.float32)
            joint_count = min(COCO_POSE_KEYPOINT_COUNT, keypoint_array.shape[1])
            channel_count = min(3, keypoint_array.shape[2])
            padded[:, :joint_count, :channel_count] = keypoint_array[
                :, :joint_count, :channel_count
            ]
            return padded

        return keypoint_array

    def _is_outside_court_polygon(
        self,
        bbox_xyxy: tuple[float, float, float, float],
    ) -> bool:
        if self.court_calibrator is None:
            return False

        court_polygon = self._expanded_court_polygon()
        bbox_polygon = self._bbox_to_polygon(bbox_xyxy)
        intersection_area, _ = cv2.intersectConvexConvex(
            bbox_polygon.astype(np.float32),
            court_polygon.astype(np.float32),
        )

        if intersection_area > 0:
            return False

        bbox_center = np.array(
            [[(bbox_xyxy[0] + bbox_xyxy[2]) / 2.0, (bbox_xyxy[1] + bbox_xyxy[3]) / 2.0]],
            dtype=np.float32,
        )
        return cv2.pointPolygonTest(court_polygon, tuple(bbox_center[0]), False) < 0

    def _estimate_court_position(
        self,
        bbox_xyxy: tuple[float, float, float, float],
    ) -> tuple[float, float] | None:
        if self.court_calibrator is None:
            return None

        footpoint_x = (bbox_xyxy[0] + bbox_xyxy[2]) / 2.0
        footpoint_y = bbox_xyxy[3]
        court_point = self.court_calibrator.transform_pixel_to_court(footpoint_x, footpoint_y)
        return court_point.x, court_point.y

    def _select_near_far_track_ids(
        self,
        trajectories: Iterable[PlayerTrajectory],
    ) -> tuple[int | None, int | None]:
        latest_positions: list[tuple[int, float]] = []
        for trajectory in trajectories:
            latest = trajectory.latest_detection
            if latest is None:
                continue

            if latest.court_position_xy is not None:
                latest_positions.append((trajectory.track_id, latest.court_position_xy[1]))
            else:
                bbox = latest.bbox_xyxy
                latest_positions.append((trajectory.track_id, bbox[3]))

        if not latest_positions:
            return None, None

        if self.court_calibrator is not None:
            far_id = min(latest_positions, key=lambda item: item[1])[0]
            near_id = max(latest_positions, key=lambda item: item[1])[0]
        else:
            far_id = min(latest_positions, key=lambda item: item[1])[0]
            near_id = max(latest_positions, key=lambda item: item[1])[0]

        if near_id == far_id:
            return near_id, None

        return near_id, far_id

    def _expanded_court_polygon(self) -> np.ndarray:
        if self.court_calibrator is None:
            raise RuntimeError("court_calibrator is required for court polygon filtering.")

        polygon = self.court_calibrator.source_pixel_points.astype(np.float32)
        if self.court_margin_px <= 0:
            return polygon

        centroid = polygon.mean(axis=0, keepdims=True)
        vectors = polygon - centroid
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        safe_norms = np.maximum(norms, 1.0)
        return polygon + (vectors / safe_norms) * self.court_margin_px

    @staticmethod
    def _bbox_to_polygon(bbox_xyxy: Sequence[float]) -> np.ndarray:
        x1, y1, x2, y2 = bbox_xyxy
        return np.array(
            [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _tensor_to_numpy(value: object) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if frame is None:
            raise ValueError("frame cannot be None.")

        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy ndarray.")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have shape H x W x 3 in BGR format.")

        if frame.size == 0:
            raise ValueError("frame cannot be empty.")

    def _next_transient_track_id(self) -> int:
        track_id = self._transient_track_id
        self._transient_track_id -= 1
        return track_id

    @staticmethod
    def _select_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")
