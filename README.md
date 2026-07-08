# Tennis Ball Tracker

A validation-spike computer vision pipeline for tennis rally analysis. The project
tracks the ball with a TrackNetV2-style PyTorch model, tracks player pose with
YOLOv8-pose, calibrates court coordinates with homography, fuses detections into
time-series features, and exports sequences for prototype shot classification.

## What This Spike Proves

- Ball heatmap inference over 3-frame windows.
- Player bounding box and pose tracking with Ultralytics YOLOv8-pose.
- Pixel-to-court coordinate mapping via tennis court homography.
- Time-series fusion of ball, player, and pose features.
- Annotated MP4 output for visual QA.
- LSTM/Transformer-ready `.npz` sequence export.

## Project Structure

```text
.
├── run_spike.py              # End-to-end orchestration script
├── verify_env.py             # Runtime environment diagnostic
├── requirements.txt
├── src/
│   ├── ball_tracker.py       # TrackNetV2-style ball detector
│   ├── court_calibrator.py   # Homography and tactical overlay utilities
│   ├── data_fusion.py        # Frame fusion and sequence export
│   ├── download_assets.py    # Runtime asset downloader
│   ├── models.py             # Prototype shot classifier
│   ├── trackers.py           # YOLOv8 pose/player tracker
│   └── train_prototype.py    # Smoke-test training loop
├── models/                   # Local model weights, ignored by Git
├── test_data/                # Local videos, ignored by Git
└── processed_data/           # Generated outputs, ignored by Git
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Validate the environment:

```bash
python3 verify_env.py
```

## Runtime Assets

Model weights and sample videos are intentionally not committed to GitHub.
Download or place them locally:

```bash
python3 src/download_assets.py
```

Expected local paths:

```text
models/tracknetv2.pt
test_data/sample_rally.mp4
```

Ultralytics may also download `yolov8n-pose.pt` on first use if it is not
already present.

## Run The Pipeline

Short smoke run:

```bash
python3 run_spike.py --max-frames 180 --skip-training
```

Full video run:

```bash
python3 run_spike.py --skip-training
```

Run with the prototype classifier smoke test:

```bash
python3 run_spike.py --epochs 5
```

## Outputs

The pipeline writes:

```text
processed_data/spike_output.mp4  # Annotated video with ball/player overlays
processed_data/spike_test.npz    # Fused time-series sequences
models/prototype_classifier.pt   # Best smoke-test classifier weights
```

Open the annotated video:

```bash
open processed_data/spike_output.mp4
```

## GPU Notes

The inference scripts select CUDA automatically when available:

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

On macOS without CUDA, the pipeline runs on CPU and can be slow for high
resolution clips. The full sample run processed 1,729 frames and produced 1,697
sliding-window sequences.

## Calibration Notes

`run_spike.py` currently uses hardcoded court corner pixels:

```python
source_pixel_corners = [
    (585.0, 260.0),
    (1335.0, 260.0),
    (1780.0, 1015.0),
    (140.0, 1015.0),
]
```

For a new broadcast angle, replace those with manually clicked singles-court
corners in this order:

```text
top-left, top-right, bottom-right, bottom-left
```

## Visual QA Checklist

Before trusting exported features, inspect `processed_data/spike_output.mp4`:

- The red/yellow ball marker should follow the actual ball.
- Player boxes and skeletons should stay on tennis players, not spectators.
- The minimap should move plausibly as players/ball move.
- Court calibration should be updated if minimap projection looks warped.

## Git Hygiene

The repository ignores generated and heavy local assets:

- `.venv/`
- `__pycache__/`
- `.env`
- `*.pt`
- `test_data/*.mp4`
- `models/*.pt`
- `processed_data/`

This keeps GitHub focused on reproducible source code while assets remain local.
