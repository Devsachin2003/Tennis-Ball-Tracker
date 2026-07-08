from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics


def detect_device() -> str:
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps (Apple Silicon GPU)"

    return "cpu"


def verify_opencv_ffmpeg() -> None:
    output_dir = Path("test_data")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "test.mp4"

    fps = 30
    width = 320
    height = 240
    frame_count = fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an MP4 VideoWriter. FFmpeg may be unavailable.")

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(frame_count):
        writer.write(frame)
    writer.release()

    reader = cv2.VideoCapture(str(output_path))
    if not reader.isOpened():
        raise RuntimeError("OpenCV could not read the generated MP4 file.")

    ok, _ = reader.read()
    reader.release()

    if not ok:
        raise RuntimeError("OpenCV opened the generated MP4 but failed to decode the first frame.")

    output_path.unlink(missing_ok=True)
    print("OpenCV FFmpeg MP4 write/read test: OK")


def main() -> None:
    print("Environment diagnostic")
    print("======================")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Ultralytics version: {ultralytics.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Active device: {detect_device()}")
    verify_opencv_ffmpeg()
    print("Environment validation completed successfully.")


if __name__ == "__main__":
    main()
