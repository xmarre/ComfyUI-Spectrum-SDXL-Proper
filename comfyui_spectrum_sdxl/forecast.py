"""Online feature forecasting utilities for Spectrum SDXL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class _FitCache:
    """Cached regression fit for a fixed schedule coordinate sequence."""

    coeff: torch.Tensor
    feature_shape: torch.Size
    feature_dtype: torch.dtype
    schedule_coords: Tuple[float, ...]


class ChebyshevFeatureForecaster:
    """
    Forecast the final hidden SDXL U-Net feature.

    The predictor fits a Chebyshev basis with ridge regularization over
    observed hidden features, then blends that forecast with a local
    first-order extrapolation. Time is normalized against the current denoiser
    schedule coordinates rather than ordinal step indices.
    """

    def __init__(self, degree: int, ridge_lambda: float, blend_weight: float, history_size: int = 100):
        """Initialize the online forecaster."""
        self.degree = int(degree)
        self.ridge_lambda = float(ridge_lambda)
        self.blend_weight = float(blend_weight)
        self.history_size = int(history_size)
        self.reset()

    def reset(self) -> None:
        """Clear observed features and any cached regression fit."""
        self.history: List[Tuple[float, torch.Tensor]] = []
        self._feature_shape: Optional[torch.Size] = None
        self._feature_dtype: Optional[torch.dtype] = None
        self._feature_device: Optional[torch.device] = None
        self._fit_cache: Optional[_FitCache] = None

    def ready(self) -> bool:
        """Return whether enough history exists to attempt a forecast."""
        return len(self.history) >= 2

    def update(self, time_coord: float, feature: torch.Tensor) -> None:
        """Append an observed feature for one solver-step time coordinate."""
        feat = feature.detach()
        if self._feature_shape is None:
            self._feature_shape = feat.shape
            self._feature_dtype = feat.dtype
            self._feature_device = feat.device
        elif feat.shape != self._feature_shape:
            self.reset()
            self._feature_shape = feat.shape
            self._feature_dtype = feat.dtype
            self._feature_device = feat.device

        self.history.append((float(time_coord), feat))
        if len(self.history) > self.history_size:
            self.history.pop(0)
        self._fit_cache = None

    def _tau(self, time_values: torch.Tensor, total_steps: int) -> torch.Tensor:
        """Map solver-step coordinates to the Chebyshev domain ``[-1, 1]``."""
        denom = max(int(total_steps) - 1, 1)
        return (time_values / float(denom)) * 2.0 - 1.0

    def _design(self, taus: torch.Tensor, degree: int) -> torch.Tensor:
        """Construct the Chebyshev design matrix up to the requested degree."""
        taus = taus.reshape(-1, 1)
        cols = [torch.ones((taus.shape[0], 1), device=taus.device, dtype=torch.float32)]
        if degree >= 1:
            cols.append(taus.to(torch.float32))
        for _ in range(2, degree + 1):
            cols.append(2.0 * taus.to(torch.float32) * cols[-1] - cols[-2])
        return torch.cat(cols[: degree + 1], dim=1)

    def _fit_if_needed(self, schedule_coords: Tuple[float, ...]) -> None:
        """Fit ridge-regression coefficients if the cache is stale."""
        cached_coords = tuple(float(v) for v in schedule_coords)
        if self._fit_cache is not None and self._fit_cache.schedule_coords == cached_coords:
            return
        if not self.history:
            raise RuntimeError("Spectrum forecaster was asked to fit without history.")
        assert self._feature_shape is not None
        assert self._feature_dtype is not None
        assert self._feature_device is not None

        device = self._feature_device
        time_tensor = torch.tensor([t for t, _ in self.history], device=device, dtype=torch.float32)
        taus = self._tau(time_tensor, total_steps)
        x_mat = self._design(taus, self.degree)
        h_mat = torch.stack([feat.reshape(-1).to(torch.float32) for _, feat in self.history], dim=0)

        p = x_mat.shape[1]
        reg = self.ridge_lambda * torch.eye(p, device=device, dtype=torch.float32)
        lhs = x_mat.transpose(0, 1) @ x_mat + reg
        rhs = x_mat.transpose(0, 1) @ h_mat

        try:
            chol = torch.linalg.cholesky(lhs)
        except RuntimeError:
            jitter = float(lhs.diag().mean().item()) if lhs.numel() > 0 else 1.0
            jitter = max(jitter * 1e-6, 1e-6)
            chol = torch.linalg.cholesky(lhs + jitter * torch.eye(p, device=device, dtype=torch.float32))

        coeff = torch.cholesky_solve(rhs, chol)
        self._fit_cache = _FitCache(
            coeff=coeff,
            feature_shape=self._feature_shape,
            feature_dtype=self._feature_dtype,
            schedule_coords=cached_coords,
        )

    def _predict_chebyshev(self, time_coord: float, total_steps: int) -> torch.Tensor:
        """Predict a feature using only the fitted Chebyshev basis."""
        assert self._fit_cache is not None
        tau_star = self._tau(
            torch.tensor([float(time_coord)], device=self._fit_cache.coeff.device, dtype=torch.float32),
            total_steps,
        )
        x_star = self._design(tau_star, self.degree)
        pred = (x_star @ self._fit_cache.coeff).reshape(self._fit_cache.feature_shape)
        return pred

    def _predict_linear(self, time_coord: float) -> torch.Tensor:
        """Predict a feature using local first-order extrapolation."""
        assert self.history
        last_coord, last_feat = self.history[-1]
        if len(self.history) < 2:
            return last_feat.to(torch.float32)

        prev_coord, prev_feat = self.history[-2]
        prev_feat = prev_feat.to(torch.float32)
        last_feat = last_feat.to(torch.float32)

        dt = max(float(last_step - prev_step), 1e-6)
        k = (float(time_coord) - float(last_step)) / dt
        return last_feat + k * (last_feat - prev_feat)

    def predict(self, time_coord: float, total_steps: int) -> torch.Tensor:
        """Predict the hidden feature for a future solver-step coordinate."""
        if not self.history:
            raise RuntimeError("Spectrum forecaster was asked to predict without history.")
        if len(self.history) == 1:
            return self.history[-1][1]

        self._fit_if_needed(total_steps)
        cheb = self._predict_chebyshev(time_coord, total_steps)
        lin = self._predict_linear(time_coord)
        out = (1.0 - self.blend_weight) * lin + self.blend_weight * cheb

        latest = self.history[-1][1]
        if not torch.isfinite(out).all():
            return latest
        return out.to(dtype=latest.dtype, device=latest.device)
