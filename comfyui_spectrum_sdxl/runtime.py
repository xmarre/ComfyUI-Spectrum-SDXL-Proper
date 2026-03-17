from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

import torch

from .config import SpectrumSDXLConfig
from .forecast import ChebyshevFeatureForecaster


class SpectrumSDXLRuntime:
    def __init__(self, cfg: SpectrumSDXLConfig):
        self.cfg = cfg.validated()
        self.forecaster = ChebyshevFeatureForecaster(
            degree=self.cfg.degree,
            ridge_lambda=self.cfg.ridge_lambda,
            blend_weight=self.cfg.blend_weight,
            history_size=self.cfg.history_size,
        )
        self.run_id = 0
        self._last_schedule_signature = None
        self.reset_all()

    def reset_cycle(self) -> None:
        self.step_idx = 0
        self.curr_ws = float(self.cfg.window_size)
        self.num_consecutive_cached_steps = 0
        self.decisions_by_sigma: Dict[float, Dict[str, Any]] = {}
        self.seen_sigmas: List[float] = []
        self.cycle_finished = False
        self.forecaster.reset()

    def reset_all(self) -> None:
        self.reset_cycle()
        self.last_info = {
            "enabled": self.cfg.enabled,
            "patched": False,
            "hook_target": None,
            "forecasted_passes": 0,
            "actual_forward_count": 0,
            "curr_ws": float(self.cfg.window_size),
            "last_sigma": None,
            "num_steps": 0,
            "run_id": self.run_id,
            "config": asdict(self.cfg),
        }

    def update_cfg(self, cfg: SpectrumSDXLConfig) -> None:
        self.cfg = cfg.validated()
        self.forecaster.degree = self.cfg.degree
        self.forecaster.ridge_lambda = self.cfg.ridge_lambda
        self.forecaster.blend_weight = self.cfg.blend_weight
        self.forecaster.history_size = self.cfg.history_size
        self.reset_all()

    def _schedule_signature(self, transformer_options: Dict[str, Any]):
        sample_sigmas = transformer_options.get("sample_sigmas", None)
        if sample_sigmas is None:
            return None
        try:
            values = sample_sigmas.detach().float().cpu().flatten().tolist()
            return tuple(round(float(v), 8) for v in values)
        except Exception:
            return None

    def _ensure_run_sync(self, transformer_options: Dict[str, Any]) -> None:
        sig = self._schedule_signature(transformer_options)
        if sig is None:
            return
        if self._last_schedule_signature is None:
            self._last_schedule_signature = sig
            self.last_info["num_steps"] = max(len(sig) - 1, 1)
            return
        if sig != self._last_schedule_signature:
            self.run_id += 1
            self._last_schedule_signature = sig
            self.reset_cycle()
            self.last_info["forecasted_passes"] = 0
            self.last_info["actual_forward_count"] = 0
            self.last_info["curr_ws"] = float(self.cfg.window_size)
            self.last_info["run_id"] = self.run_id
            self.last_info["num_steps"] = max(len(sig) - 1, 1)

    def num_steps(self) -> int:
        return max(int(self.last_info.get("num_steps", 0)), 1)

    def sigma_key(self, transformer_options: Dict[str, Any], timesteps: torch.Tensor) -> float:
        sigmas = transformer_options.get("sigmas", None)
        if sigmas is not None:
            try:
                return round(float(sigmas.detach().flatten()[0].item()), 8)
            except Exception:
                pass
        try:
            return round(float(timesteps.detach().flatten()[0].item()), 8)
        except Exception:
            return float(self.step_idx)

    def _finish_cycle_if_needed(self) -> None:
        if len(self.seen_sigmas) >= self.num_steps() and not self.cycle_finished:
            self.cycle_finished = True

    def _restart_cycle(self) -> None:
        self.run_id += 1
        self.reset_cycle()
        self.last_info["forecasted_passes"] = 0
        self.last_info["actual_forward_count"] = 0
        self.last_info["curr_ws"] = float(self.cfg.window_size)
        self.last_info["run_id"] = self.run_id

    def _should_restart_on_sigma(self, sigma: float) -> bool:
        if not self.seen_sigmas:
            return False
        if sigma != self.seen_sigmas[0]:
            return False
        return len(self.seen_sigmas) > 1

    def begin_step(self, transformer_options: Dict[str, Any], timesteps: torch.Tensor) -> Dict[str, Any]:
        self._ensure_run_sync(transformer_options)

        sigma = self.sigma_key(transformer_options, timesteps)
        self.last_info["last_sigma"] = sigma

        self._finish_cycle_if_needed()
        if self.cycle_finished:
            self._restart_cycle()
        if self._should_restart_on_sigma(sigma):
            self._restart_cycle()

        if sigma in self.decisions_by_sigma:
            return self.decisions_by_sigma[sigma]

        step_idx = len(self.seen_sigmas)
        self.seen_sigmas.append(sigma)

        actual_forward = True
        if step_idx >= self.cfg.warmup_steps:
            ws_floor = max(1, int(torch.floor(torch.tensor(self.curr_ws)).item()))
            actual_forward = ((self.num_consecutive_cached_steps + 1) % ws_floor) == 0

        if not self.forecaster.ready():
            actual_forward = True

        if actual_forward:
            if step_idx >= self.cfg.warmup_steps:
                self.curr_ws = round(self.curr_ws + float(self.cfg.flex_window), 3)
            self.num_consecutive_cached_steps = 0
            self.last_info["actual_forward_count"] += 1
        else:
            self.num_consecutive_cached_steps += 1
            self.last_info["forecasted_passes"] += 1

        self.step_idx = step_idx
        self.last_info["curr_ws"] = self.curr_ws

        decision = {
            "sigma": sigma,
            "step_idx": step_idx,
            "actual_forward": actual_forward,
            "run_id": self.run_id,
        }
        self.decisions_by_sigma[sigma] = decision
        return decision

    def observe_actual_feature(self, step_idx: int, feature: torch.Tensor) -> None:
        self.forecaster.update(step_idx, feature)

    def predict_feature(self, step_idx: int) -> torch.Tensor:
        return self.forecaster.predict(step_idx, self.num_steps())
