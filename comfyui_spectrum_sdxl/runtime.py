from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from .config import SpectrumSDXLConfig
from .forecast import ChebyshevFeatureForecaster


StreamKey = Tuple[Tuple[str, ...], Tuple[int, ...], Tuple[int, ...]]


@dataclass
class _StreamState:
    forecaster: ChebyshevFeatureForecaster
    curr_ws: float
    num_consecutive_cached_steps: int = 0
    decisions_by_global_step: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    observed_global_steps: Set[int] = field(default_factory=set)
    local_step_count: int = 0
    last_global_step_idx: Optional[int] = None


class SpectrumSDXLRuntime:
    def __init__(self, cfg: SpectrumSDXLConfig):
        self.cfg = cfg.validated()
        self.run_id = 0
        self._last_schedule_signature = None
        self.reset_all()

    def _make_forecaster(self) -> ChebyshevFeatureForecaster:
        return ChebyshevFeatureForecaster(
            degree=self.cfg.degree,
            ridge_lambda=self.cfg.ridge_lambda,
            blend_weight=self.cfg.blend_weight,
            history_size=self.cfg.history_size,
        )

    def reset_cycle(self) -> None:
        self.stream_states: Dict[StreamKey, _StreamState] = {}

    def reset_all(self) -> None:
        self._last_schedule_signature = None
        self._last_schedule_token = None
        self._pending_schedule_token = None
        self._sigma_to_step: Dict[float, int] = {}
        self._ambiguous_schedule_sigmas: Set[float] = set()
        self.reset_cycle()
        self.last_info = {
            "enabled": self.cfg.enabled,
            "patched": False,
            "hook_target": None,
            "forecasted_passes": 0,
            "actual_forward_count": 0,
            "curr_ws": float(self.cfg.window_size),
            "last_sigma": None,
            "last_stream_key": None,
            "active_streams": 0,
            "num_steps": 0,
            "run_id": self.run_id,
            "config": asdict(self.cfg),
        }

    def update_cfg(self, cfg: SpectrumSDXLConfig) -> None:
        self.cfg = cfg.validated()
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

    def _schedule_token(self, transformer_options: Dict[str, Any]):
        sample_sigmas = transformer_options.get("sample_sigmas", None)
        if sample_sigmas is None:
            return None
        try:
            return (
                int(sample_sigmas.data_ptr()),
                tuple(int(v) for v in sample_sigmas.shape),
                str(sample_sigmas.device),
                str(sample_sigmas.dtype),
            )
        except Exception:
            return id(sample_sigmas)

    def _rebuild_schedule_index(self, sig: Tuple[float, ...]) -> None:
        self._sigma_to_step: Dict[float, int] = {}
        self._ambiguous_schedule_sigmas: Set[float] = set()
        for idx, sigma in enumerate(sig[:-1]):
            if sigma in self._ambiguous_schedule_sigmas:
                continue
            if sigma in self._sigma_to_step:
                self._sigma_to_step.pop(sigma, None)
                self._ambiguous_schedule_sigmas.add(sigma)
                continue
            self._sigma_to_step[sigma] = idx

    def _reset_run_state(self) -> None:
        self.reset_cycle()
        self.last_info["forecasted_passes"] = 0
        self.last_info["actual_forward_count"] = 0
        self.last_info["curr_ws"] = float(self.cfg.window_size)
        self.last_info["last_sigma"] = None
        self.last_info["last_stream_key"] = None
        self.last_info["active_streams"] = 0
        self.last_info["run_id"] = self.run_id

    def _ensure_run_sync(self, transformer_options: Dict[str, Any]) -> None:
        sig = self._schedule_signature(transformer_options)
        token = self._schedule_token(transformer_options)
        if sig is None:
            return
        if self._last_schedule_signature is None:
            self._last_schedule_signature = sig
            self._last_schedule_token = token
            self._pending_schedule_token = None
            self._rebuild_schedule_index(sig)
            self.last_info["num_steps"] = max(len(sig) - 1, 1)
            return
        if sig != self._last_schedule_signature:
            self.run_id += 1
            self._last_schedule_signature = sig
            self._last_schedule_token = token
            self._pending_schedule_token = None
            self._rebuild_schedule_index(sig)
            self._reset_run_state()
            self.last_info["num_steps"] = max(len(sig) - 1, 1)
            return
        if token != self._last_schedule_token:
            self._pending_schedule_token = token
        else:
            self._pending_schedule_token = None

    def num_steps(self) -> int:
        return max(int(self.last_info.get("num_steps", 0)), 1)

    def stream_key(
        self,
        transformer_options: Dict[str, Any],
        input_shape: Optional[Tuple[int, ...]],
    ) -> Optional[StreamKey]:
        if input_shape is None:
            return None

        uuids = transformer_options.get("uuids", None)
        cond_or_uncond = transformer_options.get("cond_or_uncond", None)
        if uuids is None or cond_or_uncond is None:
            return None

        try:
            uuid_key = tuple(str(u) for u in uuids)
            cond_key = tuple(int(v) for v in cond_or_uncond)
            shape_key = tuple(int(v) for v in input_shape)
        except Exception:
            return None

        if not shape_key or (not uuid_key and not cond_key):
            return None
        return (uuid_key, cond_key, shape_key)

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
            return 0.0

    def global_step_index(self, transformer_options: Dict[str, Any], sigma: float) -> Optional[int]:
        sig = self._schedule_signature(transformer_options)
        if sig is None:
            return None
        if sig != self._last_schedule_signature:
            return None
        if sigma in getattr(self, "_ambiguous_schedule_sigmas", set()):
            return None
        return getattr(self, "_sigma_to_step", {}).get(sigma)

    def _ensure_stream_state(self, stream_key: StreamKey) -> _StreamState:
        state = self.stream_states.get(stream_key)
        if state is None:
            state = self._reset_stream_state(stream_key)
        return state

    def _reset_stream_state(self, stream_key: StreamKey) -> _StreamState:
        state = _StreamState(
            forecaster=self._make_forecaster(),
            curr_ws=float(self.cfg.window_size),
        )
        self.stream_states[stream_key] = state
        self.last_info["active_streams"] = len(self.stream_states)
        return state

    def begin_step(
        self,
        transformer_options: Dict[str, Any],
        timesteps: torch.Tensor,
        input_shape: Optional[Tuple[int, ...]] = None,
    ) -> Dict[str, Any]:
        self._ensure_run_sync(transformer_options)

        sigma = self.sigma_key(transformer_options, timesteps)
        stream_key = self.stream_key(transformer_options, input_shape)
        global_step_idx = self.global_step_index(transformer_options, sigma)

        self.last_info["last_sigma"] = sigma
        self.last_info["last_stream_key"] = stream_key

        if stream_key is None or global_step_idx is None:
            self.last_info["actual_forward_count"] += 1
            return {
                "sigma": sigma,
                "step_idx": global_step_idx if global_step_idx is not None else 0,
                "global_step_idx": global_step_idx,
                "local_step_count": 0,
                "actual_forward": True,
                "run_id": self.run_id,
                "stream_key": stream_key,
                "forecast_safe": False,
            }

        state = None
        if self._pending_schedule_token is not None:
            existing_state = self.stream_states.get(stream_key)
            existing_step_zero = existing_state is not None and (
                0 in existing_state.decisions_by_global_step or 0 in existing_state.observed_global_steps
            )
            if global_step_idx == 0 and existing_step_zero:
                self._last_schedule_token = self._pending_schedule_token
                self._pending_schedule_token = None
                state = self._reset_stream_state(stream_key)
            elif global_step_idx >= 0:
                self._last_schedule_token = self._pending_schedule_token
                self._pending_schedule_token = None

        if state is None:
            state = self._ensure_stream_state(stream_key)
        if state.last_global_step_idx is not None and global_step_idx < state.last_global_step_idx:
            state = self._reset_stream_state(stream_key)
        if global_step_idx in state.decisions_by_global_step:
            decision = state.decisions_by_global_step[global_step_idx]
            self.last_info["curr_ws"] = state.curr_ws
            return decision

        local_step_count = state.local_step_count
        state.local_step_count += 1

        actual_forward = True
        if local_step_count >= self.cfg.warmup_steps:
            ws_floor = max(1, int(torch.floor(torch.tensor(state.curr_ws)).item()))
            actual_forward = ((state.num_consecutive_cached_steps + 1) % ws_floor) == 0

        if not state.forecaster.ready():
            actual_forward = True

        if actual_forward:
            if local_step_count >= self.cfg.warmup_steps:
                state.curr_ws = round(state.curr_ws + float(self.cfg.flex_window), 3)
            state.num_consecutive_cached_steps = 0
            self.last_info["actual_forward_count"] += 1
        else:
            state.num_consecutive_cached_steps += 1
            self.last_info["forecasted_passes"] += 1

        self.last_info["curr_ws"] = state.curr_ws

        decision = {
            "sigma": sigma,
            "step_idx": global_step_idx,
            "global_step_idx": global_step_idx,
            "local_step_count": local_step_count,
            "actual_forward": actual_forward,
            "run_id": self.run_id,
            "stream_key": stream_key,
            "forecast_safe": True,
        }
        state.decisions_by_global_step[global_step_idx] = decision
        state.last_global_step_idx = global_step_idx
        return decision

    def observe_actual_feature(self, stream_key: Optional[StreamKey], global_step_idx: Optional[int], feature: torch.Tensor) -> None:
        if stream_key is None:
            return
        if global_step_idx is None:
            return
        state = self.stream_states.get(stream_key)
        if state is None:
            return
        if global_step_idx in state.observed_global_steps:
            return
        state.observed_global_steps.add(global_step_idx)
        state.forecaster.update(global_step_idx, feature)

    def predict_feature(self, stream_key: StreamKey, global_step_idx: int) -> torch.Tensor:
        state = self.stream_states.get(stream_key)
        if state is None:
            raise RuntimeError("Spectrum runtime has no state for the requested stream.")
        return state.forecaster.predict(global_step_idx, self.num_steps())
