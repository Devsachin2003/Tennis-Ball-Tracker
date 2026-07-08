"""Court calibration utilities for tennis broadcast frames.

This module provides a small, production-oriented homography pipeline for
mapping between image pixels and physical singles-court coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np


SINGLES_COURT_WIDTH_M = 8.23
SINGLES_COURT_LENGTH_M = 23.77
SERVICE_LINE_FROM_NET_M = 6.40


class CalibrationError(ValueError):
    """Raised when a court calibration cannot produce a valid homography."""


@dataclass(frozen=True)
class CourtPoint:
    """Physical court coordinate in meters."""

    x: float
    y: float


class TennisCourtCalibrator:
    """Map points between broadcast frame pixels and court coordinates.

    The source points must be the visible singles-court corners in this exact
    order: top-left, top-right, bottom-right, bottom-left. The destination court
    coordinate system uses meters, with origin ``(0, 0)`` at the top-left corner,
    X increasing left-to-right, and Y increasing from the far baseline to the
    near baseline.

    Args:
        source_pixel_points: Four ``(x, y)`` pixel coordinates from the frame.

    Raises:
        CalibrationError: If the point set is malformed, degenerate, or cannot
            produce a stable homography matrix.
    """

    destination_court_points = np.array(
        [
            [0.0, 0.0],
            [SINGLES_COURT_WIDTH_M, 0.0],
            [SINGLES_COURT_WIDTH_M, SINGLES_COURT_LENGTH_M],
            [0.0, SINGLES_COURT_LENGTH_M],
        ],
        dtype=np.float32,
    )

    def __init__(self, source_pixel_points: Sequence[Sequence[float]]) -> None:
        self.source_pixel_points = self._validate_points(
            source_pixel_points,
            name="source_pixel_points",
        )

        self.pixel_to_court_matrix = self._compute_homography(
            self.source_pixel_points,
            self.destination_court_points,
        )
        self.court_to_pixel_matrix = self._compute_homography(
            self.destination_court_points,
            self.source_pixel_points,
        )

    def transform_pixel_to_court(self, x: float, y: float) -> CourtPoint:
        """Transform a frame pixel coordinate into a physical court position.

        Args:
            x: Pixel X coordinate in the broadcast frame.
            y: Pixel Y coordinate in the broadcast frame.

        Returns:
            CourtPoint: Court position in meters.
        """

        transformed = self._transform_points(
            np.array([[x, y]], dtype=np.float32),
            self.pixel_to_court_matrix,
        )[0]
        return CourtPoint(float(transformed[0]), float(transformed[1]))

    def transform_court_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        """Transform a physical court coordinate back into frame pixels.

        Args:
            x: Court X coordinate in meters.
            y: Court Y coordinate in meters.

        Returns:
            A ``(pixel_x, pixel_y)`` tuple.
        """

        transformed = self._transform_points(
            np.array([[x, y]], dtype=np.float32),
            self.court_to_pixel_matrix,
        )[0]
        return float(transformed[0]), float(transformed[1])

    def transform_pixels_to_court(
        self,
        pixel_coords: Iterable[Sequence[float]],
    ) -> np.ndarray:
        """Vectorized pixel-to-court transform for trajectories or detections.

        Args:
            pixel_coords: Iterable of ``(x, y)`` frame coordinates.

        Returns:
            ``N x 2`` NumPy array containing court coordinates in meters.
        """

        points = self._validate_points_array(pixel_coords, name="pixel_coords")
        return self._transform_points(points, self.pixel_to_court_matrix)

    def transform_court_points_to_pixel(
        self,
        court_coords: Iterable[Sequence[float]],
    ) -> np.ndarray:
        """Vectorized court-to-pixel transform for drawing overlays.

        Args:
            court_coords: Iterable of ``(x, y)`` court coordinates in meters.

        Returns:
            ``N x 2`` NumPy array containing frame pixel coordinates.
        """

        points = self._validate_points_array(court_coords, name="court_coords")
        return self._transform_points(points, self.court_to_pixel_matrix)

    def draw_tactical_overlay(
        self,
        image: np.ndarray,
        pixel_coords: Iterable[Sequence[float]],
    ) -> np.ndarray:
        """Draw tactical depth labels using this calibrated court geometry.

        Args:
            image: BGR video frame to annotate.
            pixel_coords: Iterable of frame coordinates in ``(x, y)`` format.

        Returns:
            Annotated image copy.
        """

        return draw_tactical_overlay(image, pixel_coords, calibrator=self)

    @staticmethod
    def _validate_points(
        points: Sequence[Sequence[float]],
        name: str,
    ) -> np.ndarray:
        point_array = TennisCourtCalibrator._validate_points_array(points, name)

        if point_array.shape != (4, 2):
            raise CalibrationError(f"{name} must contain exactly four (x, y) points.")

        contour_area = abs(cv2.contourArea(point_array.reshape((-1, 1, 2))))
        if contour_area <= 1.0:
            raise CalibrationError(f"{name} are degenerate or nearly collinear.")

        return point_array

    @staticmethod
    def _validate_points_array(
        points: Iterable[Sequence[float]],
        name: str,
    ) -> np.ndarray:
        point_array = np.asarray(list(points), dtype=np.float32)

        if point_array.ndim != 2 or point_array.shape[1] != 2:
            raise CalibrationError(f"{name} must be shaped as N x 2 coordinates.")

        if point_array.shape[0] == 0:
            raise CalibrationError(f"{name} must contain at least one point.")

        if not np.isfinite(point_array).all():
            raise CalibrationError(f"{name} contains non-finite values.")

        return point_array

    @staticmethod
    def _compute_homography(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
        matrix, _ = cv2.findHomography(source, destination, method=0)

        if matrix is None:
            raise CalibrationError("cv2.findHomography failed to compute a matrix.")

        if not np.isfinite(matrix).all():
            raise CalibrationError("Homography matrix contains non-finite values.")

        determinant = np.linalg.det(matrix)
        if abs(determinant) < 1e-10:
            raise CalibrationError("Homography matrix is singular or unstable.")

        condition_number = np.linalg.cond(matrix)
        if not np.isfinite(condition_number) or condition_number > 1e12:
            raise CalibrationError("Homography matrix is poorly conditioned.")

        return matrix.astype(np.float64)

    @staticmethod
    def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        homogeneous = np.column_stack(
            [points.astype(np.float64), np.ones(points.shape[0], dtype=np.float64)]
        )
        transformed = homogeneous @ matrix.T
        scale = transformed[:, 2:3]

        if np.any(np.isclose(scale, 0.0)):
            raise CalibrationError("Homogeneous transform produced a zero scale value.")

        return (transformed[:, :2] / scale).astype(np.float64)


def classify_depth(court_y: float) -> str:
    """Classify relative court depth from a Y coordinate in meters.

    The labels are intentionally coarse for Step 1 diagnostics. They separate
    near-net positions, service-box depth, and baseline-depth positions.
    """

    net_y = SINGLES_COURT_LENGTH_M / 2.0
    near_service_line_y = net_y + SERVICE_LINE_FROM_NET_M
    far_service_line_y = net_y - SERVICE_LINE_FROM_NET_M

    distance_from_net = abs(court_y - net_y)
    if distance_from_net <= 1.5:
        return "Net Short"

    if far_service_line_y <= court_y <= near_service_line_y:
        return "Service Box"

    return "Baseline Deep"


def draw_tactical_overlay(
    image: np.ndarray,
    pixel_coords: Iterable[Sequence[float]],
    calibrator: TennisCourtCalibrator | None = None,
) -> np.ndarray:
    """Draw and print tactical depth labels for pixel detections.

    Args:
        image: BGR video frame to annotate. The frame is copied before drawing.
        pixel_coords: Iterable of pixel coordinates, such as ball trajectory
            points, in ``(x, y)`` format.
        calibrator: Calibrated ``TennisCourtCalibrator`` instance. Use
            ``TennisCourtCalibrator.draw_tactical_overlay`` when calling from an
            existing calibrator object.

    Returns:
        Annotated image copy. Depth labels are also printed for quick CLI use.
    """

    if image is None or image.ndim != 3:
        raise CalibrationError("image must be a valid BGR frame with shape H x W x C.")

    if calibrator is None:
        raise CalibrationError("calibrator is required to transform pixel coordinates.")

    points = TennisCourtCalibrator._validate_points_array(pixel_coords, "pixel_coords")
    court_points = calibrator.transform_pixels_to_court(points)
    annotated = image.copy()

    for index, (pixel_point, court_point) in enumerate(zip(points, court_points), start=1):
        court_x, court_y = court_point
        depth = classify_depth(float(court_y))
        pixel_x, pixel_y = np.rint(pixel_point).astype(int)

        print(
            f"Point {index}: court=({court_x:.2f}m, {court_y:.2f}m), depth={depth}"
        )

        cv2.circle(annotated, (pixel_x, pixel_y), radius=5, color=(0, 255, 255), thickness=-1)
        cv2.putText(
            annotated,
            depth,
            (pixel_x + 8, pixel_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated
