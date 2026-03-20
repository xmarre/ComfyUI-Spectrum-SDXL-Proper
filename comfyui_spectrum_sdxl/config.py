"""Configuration model for the native SDXL Spectrum runtime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpectrumSDXLConfig:
    """Validated configuration for SDXL Spectrum scheduling and forecasting."""

    enabled: bool = True
    blend_weight: float = 0.50
    degree: int = 4
    ridge_lambda: float = 0.10
    window_size: float = 2.0
    flex_window: float = 0.75
    warmup_steps: int = 5
    tail_actual_steps: int = 3
    history_size: int = 100
    debug: bool = False
    min_fit_points: int = 6

    def validated(self) -> "SpectrumSDXLConfig":
        """Validate the configuration in place and return ``self``."""
        if not (0.0 <= float(self.blend_weight) <= 1.0):
            raise ValueError("blend_weight must be in [0, 1].")
        if int(self.degree) < 1:
            raise ValueError("degree must be >= 1.")
        if float(self.ridge_lambda) < 0.0:
            raise ValueError("ridge_lambda must be >= 0.")
        if float(self.window_size) < 1.0:
            raise ValueError("window_size must be >= 1.")
        if float(self.flex_window) < 0.0:
            raise ValueError("flex_window must be >= 0.")
        if int(self.warmup_steps) < 0:
            raise ValueError("warmup_steps must be >= 0.")
        if int(self.tail_actual_steps) < 0:
            raise ValueError("tail_actual_steps must be >= 0.")
        if int(self.min_fit_points) < 1:
            raise ValueError("min_fit_points must be >= 1.")
        required_history_size = max(3, int(self.degree) + 1, int(self.min_fit_points))
        if int(self.history_size) < required_history_size:
            raise ValueError("history_size must be >= max(3, degree + 1, min_fit_points).")
        return self
