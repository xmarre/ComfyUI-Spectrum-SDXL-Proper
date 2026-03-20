"""Online feature forecasting utilities for Spectrum SDXL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class _FitCache:
    """Cached regression fit for the current sigma-coordinate history."""

    coeff: torch.Tensor
    feature_shape: torch.Size
    feature_dtype: torch.dtype
    coord_min: float
    coord_max: float


class ChebyshevFeatureForecaster:
    """
    Forecast the configured SDXL target tensor.

    The predictor fits a Chebyshev basis with ridge regularization over
    observed target tensors, then blends that forecast with a local
    first-order extrapolation. The provided ``time_coord`` values are raw
    sigma-space coordinates and are normalized internally for the fit.
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
        self._coord_bounds: Optional[Tuple[float, float]] = None

    def ready(self) -> bool:
        """Return whether enough history exists to attempt a forecast."""
        return len(self.history) >= 2

    def set_coord_bounds(self, coord_min: float, coord_max: float) -> None:
        """Persist the active schedule-wide sigma span for future fits."""
        lo = float(min(coord_min, coord_max))
        hi = float(max(coord_min, coord_max))
        new_bounds = (lo, hi)
        if self._coord_bounds == new_bounds:
            return
        self._coord_bounds = new_bounds
        self._fit_cache = None

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

    def _tau(self, time_values: torch.Tensor, coord_min: float, coord_max: float) -> torch.Tensor:
        """Map raw sigma coordinates to the Chebyshev domain ``[-1, 1]``."""
        values = time_values.to(torch.float32)
        denom = float(coord_max) - float(coord_min)
        if abs(denom) < 1e-12:
            return torch.zeros_like(values, dtype=torch.float32)
        return ((values - float(coord_min)) / denom) * 2.0 - 1.0

    def _design(self, taus: torch.Tensor, degree: int) -> torch.Tensor:
        """Construct the Chebyshev design matrix up to the requested degree."""
        taus = taus.reshape(-1, 1)
        cols = [torch.ones((taus.shape[0], 1), device=taus.device, dtype=torch.float32)]
        if degree >= 1:
            cols.append(taus.to(torch.float32))
        for _ in range(2, degree + 1):
            cols.append(2.0 * taus.to(torch.float32) * cols[-1] - cols[-2])
        return torch.cat(cols[: degree + 1], dim=1)

    def _fit_if_needed(self) -> None:
        """Fit ridge-regression coefficients if the cache is stale."""
        if self._fit_cache is not None:
            return
        if not self.history:
            raise RuntimeError("Spectrum forecaster was asked to fit without history.")
        assert self._feature_shape is not None
        assert self._feature_dtype is not None
        assert self._feature_device is not None

        device = self._feature_device
        time_tensor = torch.tensor([t for t, _ in self.history], device=device, dtype=torch.float32)
        if self._coord_bounds is not None:
            coord_min, coord_max = self._coord_bounds
        else:
            coord_min = float(time_tensor.min().item())
            coord_max = float(time_tensor.max().item())
        taus = self._tau(time_tensor, coord_min, coord_max)
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
            coord_min=coord_min,
            coord_max=coord_max,
        )

    def _predict_chebyshev(self, time_coord: float) -> torch.Tensor:
        """Predict a feature using only the fitted Chebyshev basis."""
        assert self._fit_cache is not None
        tau_star = self._tau(
            torch.tensor([float(time_coord)], device=self._fit_cache.coeff.device, dtype=torch.float32),
            self._fit_cache.coord_min,
            self._fit_cache.coord_max,
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

        dt = float(last_coord - prev_coord)
        if abs(dt) < 1e-6:
            return last_feat
        k = (float(time_coord) - float(last_coord)) / dt
        return last_feat + k * (last_feat - prev_feat)

    def predict(self, time_coord: float, total_steps: int) -> torch.Tensor:
        """Predict the configured target tensor for a future solver-step coordinate."""
        del total_steps
        if not self.history:
            raise RuntimeError("Spectrum forecaster was asked to predict without history.")
        if len(self.history) == 1:
            return self.history[-1][1]

        self._fit_if_needed()
        cheb = self._predict_chebyshev(time_coord)
        lin = self._predict_linear(time_coord)
        out = (1.0 - self.blend_weight) * lin + self.blend_weight * cheb

        latest = self.history[-1][1]
        if not torch.isfinite(out).all():
            return latest
        return out.to(dtype=latest.dtype, device=latest.device)
