"""TrackNetV2-style tennis ball detection over three-frame sequences.

The implementation here is intentionally lightweight for a validation spike:
it accepts three adjacent video frames, stacks them as a 9-channel tensor, and
predicts a single-channel 256x256 ball heatmap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


MODEL_INPUT_SIZE = (512, 288)  # width, height
HEATMAP_SIZE = (256, 256)  # width, height
LOW_CONFIDENCE_THRESHOLD = 0.5
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BallDetection:
    """Ball coordinate prediction in the original frame coordinate system."""

    x: float
    y: float
    confidence: float


class KerasWidthBatchNorm(nn.Module):
    """BatchNorm over the final spatial dimension to match converted Keras weights."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))
        self.register_buffer("running_mean", torch.zeros(width))
        self.register_buffer("running_var", torch.ones(width))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.eps = 1e-5
        self.momentum = 0.1

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize ``N x C x H x W`` tensors across the width dimension."""

        batch_size, channels, height, width = tensor.shape
        if self.training:
            self.num_batches_tracked.add_(1)

        normalized = F.batch_norm(
            tensor.reshape(batch_size * channels * height, width),
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training,
            self.momentum,
            self.eps,
        )
        return normalized.reshape(batch_size, channels, height, width)


class TrackNetConv(nn.Module):
    """TrackNetV2 convolution block matching the downloaded checkpoint keys."""

    def __init__(self, in_channels: int, out_channels: int, batchnorm_width: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")
        self.bn = KerasWidthBatchNorm(batchnorm_width)
        self.act = nn.ReLU()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply Conv2D, ReLU, then Keras-style width BatchNorm."""

        return self.bn(self.act(self.conv(tensor)))


class TrackNetArchitecture(nn.Module):
    """TrackNetV2-style encoder-decoder for 9-channel ball heatmaps.

    The model expects three consecutive RGB frames stacked along the channel
    axis, yielding an input tensor shaped ``B x 9 x 288 x 512``. The downloaded
    TrackNetV2 checkpoint predicts 3 heatmaps, one for each input frame; this
    module returns the middle-frame heatmap resized to ``B x 1 x 256 x 256``.
    """

    def __init__(
        self,
        input_channels: int = 9,
        output_channels: int = 3,
        heatmap_size: tuple[int, int] = HEATMAP_SIZE,
    ) -> None:
        super().__init__()
        self.heatmap_size = heatmap_size

        self.conv2d_1 = TrackNetConv(input_channels, 64, batchnorm_width=512)
        self.conv2d_2 = TrackNetConv(64, 64, batchnorm_width=512)
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2d_3 = TrackNetConv(64, 128, batchnorm_width=256)
        self.conv2d_4 = TrackNetConv(128, 128, batchnorm_width=256)
        self.max_pooling_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2d_5 = TrackNetConv(128, 256, batchnorm_width=128)
        self.conv2d_6 = TrackNetConv(256, 256, batchnorm_width=128)
        self.conv2d_7 = TrackNetConv(256, 256, batchnorm_width=128)
        self.max_pooling_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2d_8 = TrackNetConv(256, 512, batchnorm_width=64)
        self.conv2d_9 = TrackNetConv(512, 512, batchnorm_width=64)
        self.conv2d_10 = TrackNetConv(512, 512, batchnorm_width=64)

        self.up_sampling_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv2d_11 = TrackNetConv(768, 256, batchnorm_width=128)
        self.conv2d_12 = TrackNetConv(256, 256, batchnorm_width=128)
        self.conv2d_13 = TrackNetConv(256, 256, batchnorm_width=128)

        self.up_sampling_2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv2d_14 = TrackNetConv(384, 128, batchnorm_width=256)
        self.conv2d_15 = TrackNetConv(128, 128, batchnorm_width=256)

        self.up_sampling_3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv2d_16 = TrackNetConv(192, 64, batchnorm_width=512)
        self.conv2d_17 = TrackNetConv(64, 64, batchnorm_width=512)
        self.conv2d_18 = nn.Conv2d(64, output_channels, kernel_size=1, padding="same")

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return a 256x256 probability heatmap."""

        tensor = self.conv2d_1(tensor)
        skip_1 = self.conv2d_2(tensor)
        tensor = self.max_pooling_1(skip_1)

        tensor = self.conv2d_3(tensor)
        skip_2 = self.conv2d_4(tensor)
        tensor = self.max_pooling_2(skip_2)

        tensor = self.conv2d_5(tensor)
        tensor = self.conv2d_6(tensor)
        skip_3 = self.conv2d_7(tensor)
        tensor = self.max_pooling_3(skip_3)

        tensor = self.conv2d_8(tensor)
        tensor = self.conv2d_9(tensor)
        tensor = self.conv2d_10(tensor)

        tensor = self.up_sampling_1(tensor)
        tensor = torch.cat([tensor, skip_3], dim=1)
        tensor = self.conv2d_11(tensor)
        tensor = self.conv2d_12(tensor)
        tensor = self.conv2d_13(tensor)

        tensor = self.up_sampling_2(tensor)
        tensor = torch.cat([tensor, skip_2], dim=1)
        tensor = self.conv2d_14(tensor)
        tensor = self.conv2d_15(tensor)

        tensor = self.up_sampling_3(tensor)
        tensor = torch.cat([tensor, skip_1], dim=1)
        tensor = self.conv2d_16(tensor)
        tensor = self.conv2d_17(tensor)

        heatmaps = torch.sigmoid(self.conv2d_18(tensor))
        heatmap = heatmaps[:, 1:2, :, :]
        return nn.functional.interpolate(
            heatmap,
            size=(self.heatmap_size[1], self.heatmap_size[0]),
            mode="bilinear",
            align_corners=False,
        )


class BallTracker:
    """Run TrackNet-style tennis ball inference on three-frame windows.

    Args:
        weights_path: Path to a ``.pt`` file containing model weights. The file
            may be either a plain state dict or a checkpoint containing one of
            ``state_dict``, ``model_state_dict``, or ``model``.
        confidence_threshold: Minimum heatmap probability required to emit a
            ball coordinate.
        input_size: Preprocessing resize target as ``(width, height)``.
        heatmap_size: Output heatmap size as ``(width, height)``.
        device: Optional explicit device string. When omitted, CUDA is selected
            when available; otherwise CPU is used.
    """

    def __init__(
        self,
        weights_path: str | Path,
        confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
        input_size: tuple[int, int] = MODEL_INPUT_SIZE,
        heatmap_size: tuple[int, int] = HEATMAP_SIZE,
        device: str | torch.device | None = None,
    ) -> None:
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.confidence_threshold = confidence_threshold
        self.device = torch.device(device) if device is not None else self._select_device()

        self.model = TrackNetArchitecture(heatmap_size=heatmap_size).to(self.device)
        self._load_weights(Path(weights_path))
        self.model.eval()
        LOGGER.info("TrackNet ball tracker active device: %s", self.device)

        self._last_original_size: tuple[int, int] | None = None

    def process_sequence(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        frame3: np.ndarray,
    ) -> np.ndarray:
        """Predict a ball heatmap from three consecutive BGR video frames.

        Args:
            frame1: Previous frame, ``t-1``, as a BGR OpenCV image.
            frame2: Current frame, ``t``, as a BGR OpenCV image.
            frame3: Next frame, ``t+1``, as a BGR OpenCV image.

        Returns:
            A ``256 x 256`` NumPy heatmap on CPU.
        """

        self._validate_matching_frames((frame1, frame2, frame3))
        self._last_original_size = (frame2.shape[1], frame2.shape[0])
        input_tensor = self._preprocess_frames((frame1, frame2, frame3))
        if self.device.type == "cuda":
            input_tensor = input_tensor.pin_memory()
        input_tensor = input_tensor.to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=self.device.type == "cuda",
        )

        with torch.inference_mode():
            heatmap = self.model(input_tensor)

        heatmap_cpu = heatmap.squeeze(0).squeeze(0).detach().cpu().float().numpy()
        del heatmap, input_tensor
        return heatmap_cpu

    def get_ball_coordinates(
        self,
        heatmap: np.ndarray | torch.Tensor,
        original_size: tuple[int, int] | None = None,
    ) -> BallDetection | None:
        """Extract the highest-confidence ball coordinate from a heatmap.

        Args:
            heatmap: Model output heatmap. Accepts ``H x W``, ``1 x H x W``, or
                ``1 x 1 x H x W`` tensors/arrays.
            original_size: Optional original frame size as ``(width, height)``.
                When omitted, the size from the most recent ``process_sequence``
                call is used.

        Returns:
            ``BallDetection`` in original frame pixels, or ``None`` when the
            maximum heatmap probability is below ``confidence_threshold``.
        """

        heatmap_array = self._heatmap_to_numpy(heatmap)
        confidence = float(np.max(heatmap_array))
        if confidence < self.confidence_threshold:
            return None

        max_index = int(np.argmax(heatmap_array))
        heatmap_y, heatmap_x = np.unravel_index(max_index, heatmap_array.shape)

        frame_size = original_size or self._last_original_size
        if frame_size is None:
            raise ValueError(
                "original_size is required before process_sequence has been called."
            )

        original_width, original_height = frame_size
        heatmap_height, heatmap_width = heatmap_array.shape
        x = (float(heatmap_x) + 0.5) * (original_width / heatmap_width)
        y = (float(heatmap_y) + 0.5) * (original_height / heatmap_height)
        return BallDetection(x=x, y=y, confidence=confidence)

    def _preprocess_frames(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        processed_frames: list[np.ndarray] = []
        for frame in frames:
            resized = cv2.resize(frame, self.input_size, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb.astype(np.float32) / 255.0
            processed_frames.append(normalized)

        stacked = np.concatenate(processed_frames, axis=2)
        tensor = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0)
        return tensor.contiguous().float()

    def _load_weights(self, weights_path: Path) -> None:
        if not weights_path.exists():
            raise FileNotFoundError(f"Ball tracker weights not found: {weights_path}")

        checkpoint = torch.load(
            weights_path,
            map_location=self.device,
            weights_only=False,
        )
        state_dict = self._extract_state_dict(checkpoint)
        state_dict = self._strip_module_prefix(state_dict)
        self.model.load_state_dict(state_dict, strict=True)

    @staticmethod
    def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
        if isinstance(checkpoint, nn.Module):
            return checkpoint.state_dict()

        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                value = checkpoint.get(key)
                if isinstance(value, nn.Module):
                    return value.state_dict()
                if isinstance(value, dict):
                    return value

            if all(
                isinstance(key, str) and isinstance(value, torch.Tensor)
                for key, value in checkpoint.items()
            ):
                return checkpoint  # type: ignore[return-value]

        raise ValueError("Unsupported weights file format for BallTracker.")

    @staticmethod
    def _strip_module_prefix(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    @staticmethod
    def _heatmap_to_numpy(heatmap: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(heatmap, torch.Tensor):
            heatmap_array = heatmap.detach().cpu().float().numpy()
        else:
            heatmap_array = np.asarray(heatmap, dtype=np.float32)

        heatmap_array = np.squeeze(heatmap_array)
        if heatmap_array.ndim != 2:
            raise ValueError("heatmap must resolve to a 2D array after squeezing.")

        if heatmap_array.size == 0 or not np.isfinite(heatmap_array).all():
            raise ValueError("heatmap must be non-empty and finite.")

        return heatmap_array

    @staticmethod
    def _validate_matching_frames(frames: Sequence[np.ndarray]) -> None:
        if len(frames) != 3:
            raise ValueError("Exactly three frames are required: t-1, t, and t+1.")

        first_shape = None
        for index, frame in enumerate(frames, start=1):
            if frame is None:
                raise ValueError(f"frame{index} cannot be None.")

            if not isinstance(frame, np.ndarray):
                raise TypeError(f"frame{index} must be a NumPy ndarray.")

            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"frame{index} must have shape H x W x 3.")

            if frame.size == 0:
                raise ValueError(f"frame{index} cannot be empty.")

            if first_shape is None:
                first_shape = frame.shape
            elif frame.shape != first_shape:
                raise ValueError("All three frames must have the same shape.")

    @staticmethod
    def _select_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    def clear_device_cache(self) -> None:
        """Release cached CUDA allocator memory after long inference loops."""

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
