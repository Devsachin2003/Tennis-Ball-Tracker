"""Time-series data fusion for tennis shot prediction.

This module merges calibrated court geometry, player pose tracks, and ball
detections into fixed-shape sequences suitable for LSTM or Transformer inputs.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from .ball_tracker import BallDetection
    from .court_calibrator import TennisCourtCalibrator
    from .trackers import PlayerPoseDetection
except ImportError:  # Allows notebook-style imports from inside src/.
    from ball_tracker import BallDetection  # type: ignore
    from court_calibrator import TennisCourtCalibrator  # type: ignore
    from trackers import PlayerPoseDetection  # type: ignore


LOGGER = logging.getLogger(__name__)

COCO_LEFT_HIP = 11
COCO_RIGHT_HIP = 12
COCO_LEFT_ANKLE = 15
COCO_RIGHT_ANKLE = 16
ROOT_KEYPOINT_CANDIDATES = (
    (COCO_LEFT_ANKLE, COCO_RIGHT_ANKLE),
    (COCO_LEFT_HIP, COCO_RIGHT_HIP),
)

DEFAULT_FEATURE_COLUMNS = (
    "frame_idx",
    "ball_x_m",
    "ball_y_m",
    "ball_confidence",
    "ball_visible",
    "near_player_x_m",
    "near_player_y_m",
    "near_player_visible",
    "near_player_track_id",
    "far_player_x_m",
    "far_player_y_m",
    "far_player_visible",
    "far_player_track_id",
)

PlayerDetectionInput = (
    Mapping[int, PlayerPoseDetection | Mapping[str, Any]]
    | Sequence[PlayerPoseDetection | Mapping[str, Any]]
)


@dataclass(frozen=True)
class PlayerCourtState:
    """Per-frame real-world player position derived from pose or bbox."""

    track_id: int
    x_m: float
    y_m: float
    source: str


@dataclass(frozen=True)
class FusedFrameData:
    """One aligned time step after ball, player, and court fusion."""

    frame_idx: int
    ball_x_m: float | None
    ball_y_m: float | None
    ball_confidence: float
    near_player: PlayerCourtState | None
    far_player: PlayerCourtState | None
    missing_ball: bool
    boundary_marker: bool = False

    def to_feature_dict(self) -> dict[str, float]:
        """Return scalar features with ``NaN`` for unavailable coordinates."""

        return {
            "frame_idx": float(self.frame_idx),
            "ball_x_m": self._nan_if_none(self.ball_x_m),
            "ball_y_m": self._nan_if_none(self.ball_y_m),
            "ball_confidence": float(self.ball_confidence),
            "ball_visible": 0.0 if self.missing_ball else 1.0,
            "near_player_x_m": self._nan_if_none(
                None if self.near_player is None else self.near_player.x_m
            ),
            "near_player_y_m": self._nan_if_none(
                None if self.near_player is None else self.near_player.y_m
            ),
            "near_player_visible": 0.0 if self.near_player is None else 1.0,
            "near_player_track_id": self._nan_if_none(
                None if self.near_player is None else float(self.near_player.track_id)
            ),
            "far_player_x_m": self._nan_if_none(
                None if self.far_player is None else self.far_player.x_m
            ),
            "far_player_y_m": self._nan_if_none(
                None if self.far_player is None else self.far_player.y_m
            ),
            "far_player_visible": 0.0 if self.far_player is None else 1.0,
            "far_player_track_id": self._nan_if_none(
                None if self.far_player is None else float(self.far_player.track_id)
            ),
        }

    @staticmethod
    def _nan_if_none(value: float | None) -> float:
        if value is None:
            return float("nan")
        return float(value)


class DataFusionEngine:
    """Fuse per-frame ball and player detections into court-coordinate records.

    Args:
        court_calibrator: Calibrator used to convert pixel coordinates into
            physical court meters.
        keypoint_confidence_threshold: Minimum keypoint confidence used for
            ankle/hip root-point estimation.
    """

    def __init__(
        self,
        court_calibrator: TennisCourtCalibrator,
        keypoint_confidence_threshold: float = 0.25,
    ) -> None:
        self.court_calibrator = court_calibrator
        self.keypoint_confidence_threshold = keypoint_confidence_threshold
        self.track_roles: dict[str, int] = {}
        self.track_initial_y_m: dict[int, float] = {}
        self.fused_frames: list[FusedFrameData] = []
        self._last_frame_idx: int | None = None

    def fuse_frame_data(
        self,
        frame_idx: int,
        ball_detection: BallDetection | None,
        player_detections: PlayerDetectionInput,
    ) -> FusedFrameData:
        """Fuse one frame of ball and player outputs into court coordinates.

        Args:
            frame_idx: Monotonic video frame index.
            ball_detection: Ball pixel detection from ``BallTracker`` or ``None``.
            player_detections: Either a list of ``PlayerPoseDetection``-like
                objects or a mapping from track ID to detection dictionaries.

        Returns:
            ``FusedFrameData`` containing real-world ball and player positions.
        """

        self._check_frame_alignment(frame_idx)

        ball_x_m: float | None = None
        ball_y_m: float | None = None
        ball_confidence = 0.0
        if ball_detection is not None:
            court_point = self.court_calibrator.transform_pixel_to_court(
                ball_detection.x,
                ball_detection.y,
            )
            ball_x_m = court_point.x
            ball_y_m = court_point.y
            ball_confidence = float(ball_detection.confidence)

        players = self._extract_player_states(player_detections)
        self._assign_player_roles(players)

        near_player = self._state_for_role("near", players)
        far_player = self._state_for_role("far", players)
        fused = FusedFrameData(
            frame_idx=frame_idx,
            ball_x_m=ball_x_m,
            ball_y_m=ball_y_m,
            ball_confidence=ball_confidence,
            near_player=near_player,
            far_player=far_player,
            missing_ball=ball_detection is None,
        )
        self.fused_frames.append(fused)
        self._last_frame_idx = frame_idx
        return fused

    def get_fused_dataframe(self) -> pd.DataFrame:
        """Return all fused frames collected so far as a Pandas DataFrame."""

        return pd.DataFrame(
            [frame.to_feature_dict() for frame in self.fused_frames],
            columns=DEFAULT_FEATURE_COLUMNS,
        )

    def reset(self) -> None:
        """Clear frame history and near/far track assignments."""

        self.track_roles.clear()
        self.track_initial_y_m.clear()
        self.fused_frames.clear()
        self._last_frame_idx = None

    def _extract_player_states(
        self,
        player_detections: PlayerDetectionInput,
    ) -> dict[int, PlayerCourtState]:
        normalized = self._normalize_player_detections(player_detections)
        states: dict[int, PlayerCourtState] = {}

        for track_id, detection in normalized.items():
            root_pixel = self._extract_root_pixel(detection)
            if root_pixel is None:
                LOGGER.warning("Dropped track %s: missing usable keypoints and bbox.", track_id)
                continue

            court_point = self.court_calibrator.transform_pixel_to_court(
                root_pixel[0],
                root_pixel[1],
            )
            source = root_pixel[2]
            states[track_id] = PlayerCourtState(
                track_id=track_id,
                x_m=court_point.x,
                y_m=court_point.y,
                source=source,
            )

            self.track_initial_y_m.setdefault(track_id, court_point.y)

        return states

    def _assign_player_roles(self, players: Mapping[int, PlayerCourtState]) -> None:
        if len(self.track_roles) >= 2:
            self._warn_for_unassigned_tracks(players)
            return

        candidate_y = dict(self.track_initial_y_m)
        for track_id, state in players.items():
            candidate_y.setdefault(track_id, state.y_m)

        if len(candidate_y) < 2:
            return

        sorted_tracks = sorted(candidate_y.items(), key=lambda item: item[1])
        far_track_id = sorted_tracks[0][0]
        near_track_id = sorted_tracks[-1][0]
        self.track_roles = {"far": far_track_id, "near": near_track_id}
        self._warn_for_unassigned_tracks(players)

    def _warn_for_unassigned_tracks(self, players: Mapping[int, PlayerCourtState]) -> None:
        assigned_ids = set(self.track_roles.values())
        for track_id in players:
            if track_id not in assigned_ids:
                LOGGER.warning("Dropped unassigned/spectator track %s during fusion.", track_id)

    def _state_for_role(
        self,
        role: str,
        players: Mapping[int, PlayerCourtState],
    ) -> PlayerCourtState | None:
        track_id = self.track_roles.get(role)
        if track_id is None:
            return None

        state = players.get(track_id)
        if state is None:
            LOGGER.warning("Missing %s-court track %s in current frame.", role, track_id)
        return state

    def _normalize_player_detections(
        self,
        player_detections: PlayerDetectionInput,
    ) -> dict[int, Any]:
        if isinstance(player_detections, Mapping):
            normalized: dict[int, Any] = {}
            for raw_track_id, detection in player_detections.items():
                track_id = self._coerce_track_id(raw_track_id, detection)
                if track_id is not None:
                    normalized[track_id] = detection
            return normalized

        normalized = {}
        for detection in player_detections:
            track_id = self._coerce_track_id(None, detection)
            if track_id is not None:
                normalized[track_id] = detection
        return normalized

    @staticmethod
    def _coerce_track_id(raw_track_id: object, detection: Any) -> int | None:
        if raw_track_id is not None:
            return int(raw_track_id)

        if isinstance(detection, Mapping):
            value = detection.get("track_id")
        else:
            value = getattr(detection, "track_id", None)

        if value is None:
            LOGGER.warning("Dropped player detection without track_id.")
            return None

        return int(value)

    def _extract_root_pixel(self, detection: Any) -> tuple[float, float, str] | None:
        keypoints = self._get_detection_value(detection, "keypoints_xyc")
        if keypoints is None:
            keypoints = self._get_detection_value(detection, "keypoints")

        if keypoints is not None:
            root = self._root_from_keypoints(np.asarray(keypoints, dtype=np.float32))
            if root is not None:
                return root[0], root[1], "pose_root"

        bbox = self._get_detection_value(detection, "bbox_xyxy")
        if bbox is None:
            bbox = self._get_detection_value(detection, "bbox")

        if bbox is None:
            return None

        x1, _, x2, y2 = np.asarray(bbox, dtype=np.float32).reshape(-1)[:4]
        return float((x1 + x2) / 2.0), float(y2), "bbox_footpoint"

    def _root_from_keypoints(self, keypoints: np.ndarray) -> tuple[float, float] | None:
        if keypoints.ndim != 2 or keypoints.shape[1] < 2:
            return None

        for candidate_indices in ROOT_KEYPOINT_CANDIDATES:
            visible_points = []
            for index in candidate_indices:
                if index >= keypoints.shape[0]:
                    continue

                x_value = float(keypoints[index, 0])
                y_value = float(keypoints[index, 1])
                confidence = float(keypoints[index, 2]) if keypoints.shape[1] >= 3 else 1.0
                if (
                    np.isfinite(x_value)
                    and np.isfinite(y_value)
                    and confidence >= self.keypoint_confidence_threshold
                ):
                    visible_points.append((x_value, y_value))

            if visible_points:
                return tuple(np.mean(np.asarray(visible_points), axis=0))  # type: ignore[return-value]

        return None

    @staticmethod
    def _get_detection_value(detection: Any, key: str) -> Any:
        if isinstance(detection, Mapping):
            return detection.get(key)
        return getattr(detection, key, None)

    def _check_frame_alignment(self, frame_idx: int) -> None:
        if self._last_frame_idx is None:
            return

        if frame_idx <= self._last_frame_idx:
            raise ValueError(
                f"frame_idx must be strictly increasing: got {frame_idx} after "
                f"{self._last_frame_idx}."
            )

        expected = self._last_frame_idx + 1
        if frame_idx != expected:
            LOGGER.warning("Missing frame(s) between %s and %s.", self._last_frame_idx, frame_idx)


class SequenceBuilder:
    """Build fixed-shape temporal windows from fused frame records.

    Args:
        window_size: Number of frames in each output sequence.
        max_missing_ball_frames: Missing-ball gap size allowed for interpolation.
            Longer gaps are marked as sequence boundaries and not interpolated.
        feature_columns: Ordered feature list used in the output matrix.
    """

    def __init__(
        self,
        window_size: int = 30,
        max_missing_ball_frames: int = 3,
        feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if max_missing_ball_frames < 0:
            raise ValueError("max_missing_ball_frames cannot be negative.")

        self.window_size = window_size
        self.max_missing_ball_frames = max_missing_ball_frames
        self.feature_columns = tuple(feature_columns)
        self.buffer: Deque[FusedFrameData] = deque(maxlen=window_size)

    def add_frame(self, fused_frame: FusedFrameData) -> np.ndarray | None:
        """Append one fused frame and return a sequence when the buffer is full."""

        if self.buffer and fused_frame.frame_idx != self.buffer[-1].frame_idx + 1:
            LOGGER.warning(
                "Sequence buffer reset due to missing frame(s): previous=%s current=%s.",
                self.buffer[-1].frame_idx,
                fused_frame.frame_idx,
            )
            self.buffer.clear()

        self.buffer.append(fused_frame)
        if len(self.buffer) < self.window_size:
            return None

        return self.build_shot_sequence(list(self.buffer))

    def build_shot_sequence(
        self,
        frame_buffer: Sequence[FusedFrameData],
        as_dataframe: bool = False,
    ) -> np.ndarray | pd.DataFrame:
        """Convert a continuous frame window into model-ready features.

        Missing short ball gaps are linearly interpolated. Longer ball gaps are
        preserved as ``NaN`` coordinates and boundary markers so training code
        can split or mask those windows.
        """

        if len(frame_buffer) != self.window_size:
            raise ValueError(
                f"frame_buffer must contain exactly {self.window_size} frames."
            )

        self._strict_check_contiguous(frame_buffer)
        dataframe = pd.DataFrame(
            [frame.to_feature_dict() for frame in frame_buffer],
            columns=self.feature_columns,
        )
        dataframe["sequence_boundary"] = self._mark_long_missing_ball_gaps(dataframe)
        dataframe = self._interpolate_short_ball_gaps(dataframe)

        if as_dataframe:
            return dataframe

        return dataframe.to_numpy(dtype=np.float32)

    def export_sequences(
        self,
        sequences: Sequence[np.ndarray | pd.DataFrame],
        output_name: str = "shot_sequences.npz",
        output_dir: str | Path = "processed_data",
    ) -> Path:
        """Save sequence tensors to ``processed_data`` as ``.npz`` or ``.parquet``."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / output_name

        if file_path.suffix == ".npz":
            arrays = [
                sequence.to_numpy(dtype=np.float32)
                if isinstance(sequence, pd.DataFrame)
                else np.asarray(sequence, dtype=np.float32)
                for sequence in sequences
            ]
            if not arrays:
                raise ValueError("At least one sequence is required for export.")
            np.savez_compressed(
                file_path,
                sequences=np.stack(arrays, axis=0),
                feature_columns=np.asarray(
                    [*self.feature_columns, "sequence_boundary"],
                    dtype=object,
                ),
            )
            return file_path

        if file_path.suffix == ".parquet":
            frames = []
            for sequence_idx, sequence in enumerate(sequences):
                frame = (
                    sequence.copy()
                    if isinstance(sequence, pd.DataFrame)
                    else pd.DataFrame(
                        sequence,
                        columns=[*self.feature_columns, "sequence_boundary"],
                    )
                )
                frame.insert(0, "sequence_idx", sequence_idx)
                frames.append(frame)

            if not frames:
                raise ValueError("At least one sequence is required for export.")
            pd.concat(frames, ignore_index=True).to_parquet(file_path, index=False)
            return file_path

        raise ValueError("output_name must end with .npz or .parquet.")

    def _strict_check_contiguous(self, frame_buffer: Sequence[FusedFrameData]) -> None:
        frame_indices = [frame.frame_idx for frame in frame_buffer]
        expected = list(range(frame_indices[0], frame_indices[0] + len(frame_indices)))
        if frame_indices != expected:
            raise ValueError(
                f"frame_buffer contains missing frames: expected {expected}, got {frame_indices}."
            )

    def _mark_long_missing_ball_gaps(self, dataframe: pd.DataFrame) -> np.ndarray:
        missing = dataframe["ball_visible"].to_numpy(dtype=np.float32) == 0.0
        boundary = np.zeros(len(dataframe), dtype=np.float32)

        start: int | None = None
        for index, is_missing in enumerate(missing):
            if is_missing and start is None:
                start = index
            if (not is_missing or index == len(missing) - 1) and start is not None:
                end = index if not is_missing else index + 1
                gap_length = end - start
                touches_window_edge = start == 0 or end == len(missing)
                if gap_length > self.max_missing_ball_frames or touches_window_edge:
                    boundary[start:end] = 1.0
                start = None

        return boundary

    def _interpolate_short_ball_gaps(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.copy()
        ball_columns = ["ball_x_m", "ball_y_m"]

        short_gap_mask = (
            (dataframe["ball_visible"] == 0.0)
            & (dataframe["sequence_boundary"] == 0.0)
        )
        interpolated = dataframe[ball_columns].interpolate(
            method="linear",
            limit=self.max_missing_ball_frames,
            limit_area="inside",
        )
        dataframe.loc[short_gap_mask, ball_columns] = interpolated.loc[
            short_gap_mask,
            ball_columns,
        ]
        non_ball_columns = [column for column in dataframe.columns if column not in ball_columns]
        dataframe[non_ball_columns] = dataframe[non_ball_columns].fillna(0.0)
        return dataframe
