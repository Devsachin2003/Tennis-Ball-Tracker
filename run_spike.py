"""End-to-end validation spike runner for tennis shot prediction.

This script orchestrates court calibration, player pose tracking, TrackNet-style
ball tracking, time-series fusion, sequence export, and a tiny classifier
training run to catch integration and tensor-shape issues.
"""

from __future__ import annotations

import argparse
import logging
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from src.ball_tracker import BallDetection, BallTracker
from src.court_calibrator import TennisCourtCalibrator
from src.data_fusion import DataFusionEngine, FusedFrameData, SequenceBuilder
from src.trackers import PlayerPoseDetection, TennisTracker
from src.train_prototype import train as train_prototype


LOGGER = logging.getLogger("run_spike")

COCO_SKELETON_EDGES = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)
KEYPOINT_CONFIDENCE_THRESHOLD = 0.25


def configure_logging() -> None:
    """Configure INFO-level console logging for pipeline progress."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def select_compute_device() -> torch.device:
    """Select the compute device for all model inference in this spike."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_compute_device(device: torch.device) -> None:
    """Log active accelerator details once at startup."""

    if device.type == "cuda":
        LOGGER.info(
            "Active compute device: %s (%s)",
            device,
            torch.cuda.get_device_name(device),
        )
        return

    LOGGER.info("Active compute device: %s", device)


def clear_cuda_cache(device: torch.device) -> None:
    """Release cached CUDA allocator memory when running long videos."""

    if device.type == "cuda":
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full validation spike pipeline.")
    parser.add_argument("--video-path", default="test_data/sample_rally.mp4")
    parser.add_argument("--ball-weights", default="models/tracknetv2.pt")
    parser.add_argument("--yolo-model", default="yolov8n-pose.pt")
    parser.add_argument("--output-path", default="processed_data/spike_test.npz")
    parser.add_argument("--annotated-video-path", default="processed_data/spike_output.mp4")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--max-missing-ball-frames", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def initialize_court_calibrator() -> TennisCourtCalibrator:
    """Create the court calibrator with placeholder broadcast-frame corners."""

    # Swap these four pixel points with manually clicked singles-court corners
    # from test_data/sample_rally.mp4: top-left, top-right, bottom-right, bottom-left.
    source_pixel_corners = [
        (585.0, 260.0),
        (1335.0, 260.0),
        (1780.0, 1015.0),
        (140.0, 1015.0),
    ]
    LOGGER.info("Initializing court calibrator with hardcoded source corners.")
    return TennisCourtCalibrator(source_pixel_corners)


def initialize_pipeline(args: argparse.Namespace) -> tuple[
    TennisCourtCalibrator,
    TennisTracker,
    BallTracker,
    DataFusionEngine,
]:
    """Initialize calibrator, player tracker, ball tracker, and fusion engine."""

    device = getattr(args, "device", select_compute_device())
    log_compute_device(device)

    calibrator = initialize_court_calibrator()
    LOGGER.info("Initializing YOLO pose tracker: %s", args.yolo_model)
    player_tracker = TennisTracker(
        model_path=args.yolo_model,
        court_calibrator=calibrator,
        device=device,
    )

    LOGGER.info("Initializing ball tracker weights: %s", args.ball_weights)
    ball_tracker = BallTracker(weights_path=args.ball_weights, device=device)
    fusion_engine = DataFusionEngine(court_calibrator=calibrator)
    return calibrator, player_tracker, ball_tracker, fusion_engine


def run_video_fusion(
    video_path: Path,
    calibrator: TennisCourtCalibrator,
    player_tracker: TennisTracker,
    ball_tracker: BallTracker,
    fusion_engine: DataFusionEngine,
    annotated_video_path: Path,
    device: torch.device,
    max_frames: int | None = None,
) -> list[FusedFrameData]:
    """Run frame-by-frame tracking and return fused middle-frame records."""

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        LOGGER.warning("Video FPS was unavailable; defaulting annotated output to 30 FPS.")
        fps = 30.0
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Could not read video width/height for annotated output.")

    annotated_video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(annotated_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open VideoWriter: {annotated_video_path}")
    LOGGER.info(
        "Writing annotated video to %s at %.2f FPS (%sx%s).",
        annotated_video_path,
        fps,
        width,
        height,
    )

    frame_buffer: deque[
        tuple[int, np.ndarray, np.ndarray, list[PlayerPoseDetection]]
    ] = deque(maxlen=3)
    fused_frames: list[FusedFrameData] = []
    raw_frame_idx = 0
    first_edge_frame_written = False

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if max_frames is not None and raw_frame_idx >= max_frames:
                LOGGER.info("Stopping early at --max-frames=%s.", max_frames)
                break

            LOGGER.info("Processing frame %s", raw_frame_idx)
            player_detections = player_tracker.process_frame(frame)
            LOGGER.info(
                "Frame %s: retained %s player detections.",
                raw_frame_idx,
                len(player_detections),
            )

            annotated_frame = annotate_players(frame, player_detections)
            frame_buffer.append(
                (raw_frame_idx, frame.copy(), annotated_frame, player_detections)
            )
            if len(frame_buffer) == 3:
                if not first_edge_frame_written:
                    _, first_edge_frame, first_edge_detections = (
                        frame_buffer[0][0],
                        frame_buffer[0][2],
                        frame_buffer[0][3],
                    )
                    draw_minimap(
                        first_edge_frame,
                        calibrator,
                        ball_detection=None,
                        player_detections=first_edge_detections,
                    )
                    writer.write(first_edge_frame)
                    first_edge_frame_written = True

                middle_idx, _, middle_frame, middle_player_detections = frame_buffer[1]
                heatmap = ball_tracker.process_sequence(
                    frame_buffer[0][1],
                    frame_buffer[1][1],
                    frame_buffer[2][1],
                )
                ball_detection = ball_tracker.get_ball_coordinates(
                    heatmap,
                    original_size=(frame.shape[1], frame.shape[0]),
                )
                log_ball_detection(middle_idx, ball_detection)

                if ball_detection is not None:
                    draw_ball_marker(middle_frame, ball_detection)
                draw_minimap(
                    middle_frame,
                    calibrator=calibrator,
                    ball_detection=ball_detection,
                    player_detections=middle_player_detections,
                )

                fused_frame = fusion_engine.fuse_frame_data(
                    frame_idx=middle_idx,
                    ball_detection=ball_detection,
                    player_detections=middle_player_detections,
                )
                fused_frames.append(fused_frame)
                writer.write(middle_frame)
                LOGGER.info("Frame %s: fused data appended.", middle_idx)

            if device.type == "cuda" and raw_frame_idx > 0 and raw_frame_idx % 100 == 0:
                clear_cuda_cache(device)
                LOGGER.info("Frame %s: CUDA cache cleanup complete.", raw_frame_idx)

            raw_frame_idx += 1
    finally:
        flush_edge_frames(
            frame_buffer=frame_buffer,
            writer=writer,
            calibrator=calibrator,
            first_edge_frame_written=first_edge_frame_written,
        )
        writer.release()
        capture.release()
        clear_cuda_cache(device)

    LOGGER.info("Video fusion complete: %s fused frames.", len(fused_frames))
    LOGGER.info("Annotated video saved to %s.", annotated_video_path)
    return fused_frames


def annotate_players(
    frame: np.ndarray,
    detections: list[PlayerPoseDetection],
) -> np.ndarray:
    """Draw player bounding boxes, track IDs, and pose skeletons."""

    annotated = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = np.rint(detection.bbox_xyxy).astype(int)
        color = (80, 220, 80)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)
        cv2.putText(
            annotated,
            f"ID {detection.track_id}",
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        draw_pose_keypoints(annotated, detection.keypoints_xyc)

    return annotated


def draw_pose_keypoints(frame: np.ndarray, keypoints_xyc: np.ndarray) -> None:
    """Draw COCO 17-keypoint pose joints and skeleton edges."""

    if keypoints_xyc.ndim != 2 or keypoints_xyc.shape[1] < 2:
        return

    visible: dict[int, tuple[int, int]] = {}
    for index, keypoint in enumerate(keypoints_xyc):
        confidence = float(keypoint[2]) if keypoints_xyc.shape[1] >= 3 else 1.0
        if confidence < KEYPOINT_CONFIDENCE_THRESHOLD:
            continue

        x_coord, y_coord = np.rint(keypoint[:2]).astype(int)
        visible[index] = (x_coord, y_coord)
        cv2.circle(frame, (x_coord, y_coord), radius=3, color=(255, 180, 40), thickness=-1)

    for start, end in COCO_SKELETON_EDGES:
        if start in visible and end in visible:
            cv2.line(frame, visible[start], visible[end], color=(255, 120, 20), thickness=2)


def draw_ball_marker(frame: np.ndarray, detection: BallDetection) -> None:
    """Draw a highly visible ball marker at pixel coordinates."""

    center = (int(round(detection.x)), int(round(detection.y)))
    cv2.circle(frame, center, radius=12, color=(0, 0, 255), thickness=3)
    cv2.circle(frame, center, radius=5, color=(0, 255, 255), thickness=-1)
    cv2.putText(
        frame,
        f"Ball {detection.confidence:.2f}",
        (center[0] + 14, max(center[1] - 14, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_minimap(
    frame: np.ndarray,
    calibrator: TennisCourtCalibrator,
    ball_detection: BallDetection | None,
    player_detections: list[PlayerPoseDetection],
) -> None:
    """Draw a top-down court projection in the frame corner."""

    minimap_width = 150
    minimap_height = 260
    margin = 24
    top_left = (frame.shape[1] - minimap_width - margin, margin)
    x0, y0 = top_left
    court = frame[y0 : y0 + minimap_height, x0 : x0 + minimap_width]
    if court.shape[0] != minimap_height or court.shape[1] != minimap_width:
        return

    overlay = court.copy()
    cv2.rectangle(overlay, (0, 0), (minimap_width - 1, minimap_height - 1), (30, 90, 30), -1)
    cv2.addWeighted(overlay, 0.72, court, 0.28, 0, court)
    cv2.rectangle(court, (8, 8), (minimap_width - 9, minimap_height - 9), (230, 230, 230), 1)
    cv2.line(
        court,
        (8, minimap_height // 2),
        (minimap_width - 9, minimap_height // 2),
        (230, 230, 230),
        1,
    )
    cv2.putText(
        court,
        "Court",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )

    for detection in player_detections:
        foot_x = (detection.bbox_xyxy[0] + detection.bbox_xyxy[2]) / 2.0
        foot_y = detection.bbox_xyxy[3]
        court_point = calibrator.transform_pixel_to_court(foot_x, foot_y)
        mini_point = court_to_minimap_point(court_point.x, court_point.y, minimap_width, minimap_height)
        cv2.circle(court, mini_point, radius=4, color=(80, 220, 80), thickness=-1)

    if ball_detection is not None:
        court_point = calibrator.transform_pixel_to_court(ball_detection.x, ball_detection.y)
        mini_point = court_to_minimap_point(court_point.x, court_point.y, minimap_width, minimap_height)
        cv2.circle(court, mini_point, radius=5, color=(0, 0, 255), thickness=-1)


def court_to_minimap_point(
    court_x_m: float,
    court_y_m: float,
    minimap_width: int,
    minimap_height: int,
) -> tuple[int, int]:
    """Map calibrated court meters to minimap pixels."""

    court_width_m = 8.23
    court_length_m = 23.77
    pad = 8
    x_coord = pad + (court_x_m / court_width_m) * (minimap_width - 2 * pad)
    y_coord = pad + (court_y_m / court_length_m) * (minimap_height - 2 * pad)
    return int(round(x_coord)), int(round(y_coord))


def flush_edge_frames(
    frame_buffer: deque[tuple[int, np.ndarray, np.ndarray, list[PlayerPoseDetection]]],
    writer: cv2.VideoWriter,
    calibrator: TennisCourtCalibrator,
    first_edge_frame_written: bool,
) -> None:
    """Write edge frames that cannot receive centered 3-frame ball inference."""

    if not frame_buffer:
        return

    if len(frame_buffer) < 3:
        for _, _, frame, detections in frame_buffer:
            draw_minimap(frame, calibrator, ball_detection=None, player_detections=detections)
            writer.write(frame)
        return

    if not first_edge_frame_written:
        _, _, frame, detections = frame_buffer[0]
        draw_minimap(frame, calibrator, ball_detection=None, player_detections=detections)
        writer.write(frame)

    _, _, frame, detections = frame_buffer[-1]
    draw_minimap(frame, calibrator, ball_detection=None, player_detections=detections)
    writer.write(frame)


def log_ball_detection(frame_idx: int, detection: BallDetection | None) -> None:
    """Log ball-detection status for one fused frame."""

    if detection is None:
        LOGGER.info("Frame %s: ball not detected above threshold.", frame_idx)
        return

    LOGGER.info(
        "Frame %s: ball=(%.1f, %.1f), confidence=%.3f.",
        frame_idx,
        detection.x,
        detection.y,
        detection.confidence,
    )


def build_and_export_sequences(
    fused_frames: list[FusedFrameData],
    output_path: Path,
    window_size: int,
    max_missing_ball_frames: int,
) -> Path:
    """Build sliding windows from fused frames and export them as compressed NPZ."""

    if len(fused_frames) < window_size:
        raise ValueError(
            f"Need at least {window_size} fused frames to build one sequence; "
            f"got {len(fused_frames)}."
        )

    builder = SequenceBuilder(
        window_size=window_size,
        max_missing_ball_frames=max_missing_ball_frames,
    )
    sequences = []
    for start_idx in range(0, len(fused_frames) - window_size + 1):
        window = fused_frames[start_idx : start_idx + window_size]
        sequences.append(builder.build_shot_sequence(window))

    LOGGER.info("Built %s sliding-window sequences.", len(sequences))
    exported_path = builder.export_sequences(
        sequences=sequences,
        output_name=output_path.name,
        output_dir=output_path.parent,
    )
    LOGGER.info("Exported fused sequences to %s.", exported_path)
    return exported_path


def attach_random_labels(npz_path: Path) -> None:
    """Attach random binary labels to the generated NPZ for training smoke tests."""

    with np.load(npz_path, allow_pickle=True) as data:
        sequences = np.asarray(data["sequences"], dtype=np.float32)
        feature_columns = np.asarray(data["feature_columns"], dtype=object)

    if sequences.shape[0] == 0:
        raise ValueError("At least one sequence is required for the training smoke test.")

    if sequences.shape[0] == 1:
        LOGGER.warning(
            "Only one sequence was extracted; duplicating it for the train/validation "
            "smoke test."
        )
        sequences = np.concatenate([sequences, sequences.copy()], axis=0)

    labels = np.random.default_rng(seed=42).integers(
        low=0,
        high=2,
        size=sequences.shape[0],
        dtype=np.int64,
    ).astype(np.float32)
    labels[0] = 0.0
    labels[1] = 1.0
    np.savez_compressed(
        npz_path,
        sequences=sequences,
        feature_columns=feature_columns,
        labels=labels,
    )
    LOGGER.info("Attached %s random binary labels to %s.", len(labels), npz_path)


def run_training_smoke_test(npz_path: Path, epochs: int, batch_size: int) -> Path:
    """Trigger the prototype trainer programmatically against the generated NPZ."""

    LOGGER.info("Starting prototype classifier smoke training for %s epochs.", epochs)
    train_args = SimpleNamespace(
        data_path=str(npz_path),
        labels_path=None,
        label_column=None,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=0.001,
        validation_split=0.2,
        hidden_dimension=64,
        backbone="lstm",
        output_dir="models",
    )
    best_path = train_prototype(train_args)
    LOGGER.info("Training smoke test complete: %s", best_path)
    return best_path


def main() -> None:
    configure_logging()
    args = parse_args()
    device = select_compute_device()
    args.device = device

    try:
        calibrator, player_tracker, ball_tracker, fusion_engine = initialize_pipeline(args)
        fused_frames = run_video_fusion(
            video_path=Path(args.video_path),
            calibrator=calibrator,
            player_tracker=player_tracker,
            ball_tracker=ball_tracker,
            fusion_engine=fusion_engine,
            annotated_video_path=Path(args.annotated_video_path),
            device=device,
            max_frames=args.max_frames,
        )
        output_path = build_and_export_sequences(
            fused_frames=fused_frames,
            output_path=Path(args.output_path),
            window_size=args.window_size,
            max_missing_ball_frames=args.max_missing_ball_frames,
        )

        if args.skip_training:
            LOGGER.info("Skipping training smoke test because --skip-training was set.")
            return

        attach_random_labels(output_path)
        run_training_smoke_test(
            npz_path=output_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    except Exception:
        LOGGER.exception("Validation spike failed.")
        raise
    finally:
        clear_cuda_cache(device)


if __name__ == "__main__":
    main()
