"""Per-stream runtime state management for native SDXL Spectrum sampling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, Optional, Set, Tuple

import torch

from .config import SpectrumSDXLConfig
from .forecast import ChebyshevFeatureForecaster


StreamKey = Tuple[Tuple[str, ...], Tuple[int, ...], Tuple[int, ...]]

_RUN_ID_KEY = "spectrum_run_id"
_SOLVER_STEP_ID_KEY = "spectrum_solver_step_id"
_TIME_COORD_KEY = "spectrum_time_coord"
_ACTUAL_FORWARD_KEY = "spectrum_actual_forward"
_TOTAL_STEPS_KEY = "spectrum_total_steps"


@dataclass
class _StreamState:
    """Mutable Spectrum state for one logical ComfyUI sampling stream."""

    forecaster: ChebyshevFeatureForecaster
    curr_ws: float
    run_id: Any = None
    num_consecutive_cached_steps: int = 0
    decisions_by_solver_step: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    observed_solver_steps: Set[int] = field(default_factory=set)
    local_step_count: int = 0


class SpectrumSDXLRuntime:
    """
    Coordinate Spectrum forecasting for one explicit outer-step context.

    The runtime no longer reconstructs solver steps from low-level model calls.
    Forecasting is allowed only when the sampler/guider has already attached an
    explicit outer-step context to ``transformer_options``.
    """

    def __init__(self, cfg: SpectrumSDXLConfig):
        """Initialize the runtime and clear all run-local state."""
        self.cfg = cfg.validated()
        self.reset_all()

    def _make_forecaster(self) -> ChebyshevFeatureForecaster:
        """Create a new forecaster from the current runtime configuration."""
        return ChebyshevFeatureForecaster(
            degree=self.cfg.degree,
            ridge_lambda=self.cfg.ridge_lambda,
            blend_weight=self.cfg.blend_weight,
            history_size=self.cfg.history_size,
        )

    def reset_cycle(self) -> None:
        """Clear all per-stream state for the current logical run."""
        self.stream_states: Dict[StreamKey, _StreamState] = {}

    def reset_all(self) -> None:
        """Reset run state and fail-open guard state to a fresh runtime."""
        self._active_run_id = None
        self._forecast_disabled = False
        self._forecast_disable_reason: Optional[str] = None
        self.reset_cycle()
        self.last_info = {
            "enabled": self.cfg.enabled,
            "patched": False,
            "hook_target": None,
            "forecasted_passes": 0,
            "actual_forward_count": 0,
            "curr_ws": float(self.cfg.window_size),
            "last_stream_key": None,
            "forecast_disabled": False,
            "forecast_disable_reason": None,
            "active_streams": 0,
            "num_steps": 0,
            "run_id": None,
            "config": asdict(self.cfg),
        }

    def update_cfg(self, cfg: SpectrumSDXLConfig) -> None:
        """Replace the runtime configuration and reset all runtime state."""
        self.cfg = cfg.validated()
        self.reset_all()

    def _reset_run_state(self, run_id: Any) -> None:
        """Reset all state scoped to one logical sampling run."""
        self.reset_cycle()
        self._active_run_id = run_id
        self._forecast_disabled = False
        self._forecast_disable_reason = None
        self.last_info["forecasted_passes"] = 0
        self.last_info["actual_forward_count"] = 0
        self.last_info["curr_ws"] = float(self.cfg.window_size)
        self.last_info["last_stream_key"] = None
        self.last_info["forecast_disabled"] = False
        self.last_info["forecast_disable_reason"] = None
        self.last_info["active_streams"] = 0
        self.last_info["num_steps"] = 0
        self.last_info["run_id"] = run_id

    def _ensure_run_sync(self, run_id: Any) -> None:
        """Reset stream state when the explicit outer run identifier changes."""
        if self._active_run_id != run_id:
            self._reset_run_state(run_id)

    def _disable_forecasting(self, reason: str) -> None:
        """Fail open for the current run."""
        self._forecast_disabled = True
        self._forecast_disable_reason = reason
        self.last_info["forecast_disabled"] = True
        self.last_info["forecast_disable_reason"] = reason

    def _extract_total_steps(self, transformer_options: Dict[str, Any]) -> int:
        """Return the sampler length when it is provided explicitly."""
        raw_total_steps = transformer_options.get(_TOTAL_STEPS_KEY, None)
        if raw_total_steps is not None:
            try:
                total_steps = int(raw_total_steps)
                if total_steps >= 1:
                    return total_steps
            except (TypeError, ValueError):
                pass

        sample_sigmas = transformer_options.get("sample_sigmas", None)
        if sample_sigmas is not None:
            try:
                return max(int(sample_sigmas.numel()) - 1, 1)
            except Exception:
                pass

        # Detect run boundary to avoid leaking previous run's step count
        incoming_run_id = transformer_options.get(_RUN_ID_KEY, None)
        if incoming_run_id is not None and incoming_run_id != self._active_run_id:
            # New run detected, don't use stale last_info
            return 1

        return max(int(self.last_info.get("num_steps", 0)), 1)

    def _expected_time_coord(self, transformer_options: Dict[str, Any], solver_step_id: int) -> Optional[float]:
        """Return the normalized schedule coordinate for one solver step when sample sigmas are available."""
        sample_sigmas = transformer_options.get("sample_sigmas", None)
        if sample_sigmas is None:
            return None
        try:
            values = tuple(float(v) for v in sample_sigmas.detach().flatten().tolist()[:-1])
        except Exception:
            return None
        if not values:
            return None
        idx = min(max(int(solver_step_id), 0), len(values) - 1)
        start = values[0]
        end = values[-1]
        denom = end - start
        if abs(denom) < 1e-12:
            return 0.0
        return float(((values[idx] - start) / denom) * 2.0 - 1.0)

    def _solver_step_context(self, transformer_options: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Validate and extract an explicit outer-step Spectrum context."""
        run_id = transformer_options.get(_RUN_ID_KEY, None)
        solver_step_id = transformer_options.get(_SOLVER_STEP_ID_KEY, None)
        time_coord = transformer_options.get(_TIME_COORD_KEY, None)
        actual_forward = transformer_options.get(_ACTUAL_FORWARD_KEY, None)

        if run_id is None and solver_step_id is None and time_coord is None and actual_forward is None:
            return None, "missing_solver_step_context"
        if run_id is None or solver_step_id is None or time_coord is None:
            return None, "invalid_solver_step_context"

        try:
            solver_step_id = int(solver_step_id)
        except (TypeError, ValueError):
            return None, "invalid_solver_step_context"
        if solver_step_id < 0:
            return None, "invalid_solver_step_context"

        try:
            time_coord = float(time_coord)
        except (TypeError, ValueError):
            return None, "invalid_solver_step_context"
        if not math.isfinite(time_coord):
            return None, "invalid_solver_step_context"

        if actual_forward is not None and not isinstance(actual_forward, bool):
            return None, "invalid_solver_step_context"

        return {
            "run_id": run_id,
            "solver_step_id": solver_step_id,
            "time_coord": time_coord,
            "actual_forward": actual_forward,
            "total_steps": self._extract_total_steps(transformer_options),
        }, None

    def num_steps(self) -> int:
        """Return the current sampler length in diffusion steps."""
        return max(int(self.last_info.get("num_steps", 0)), 1)

    def _is_tail_actual_step(self, solver_step_id: int, total_steps: int) -> bool:
        """Return whether this solver step is forced to stay on the real path."""
        tail_actual_steps = int(self.cfg.tail_actual_steps)
        if tail_actual_steps <= 0:
            return False
        tail_start = max(0, int(total_steps) - tail_actual_steps)
        return int(solver_step_id) >= tail_start

    def stream_key(
        self,
        transformer_options: Dict[str, Any],
        input_shape: Optional[Tuple[int, ...]],
    ) -> Optional[StreamKey]:
        """Build a stable logical stream key for the current sampler call."""
        if input_shape is None:
            return None

        uuids = transformer_options.get("uuids", None)
        cond_or_uncond = transformer_options.get("cond_or_uncond", None)
        if uuids is None or cond_or_uncond is None:
            return None

        try:
            raw_uuid_key = tuple(uuids)
            if any(u is None for u in raw_uuid_key):
                return None
            uuid_key = tuple(str(u) for u in raw_uuid_key)
            cond_key = tuple(int(v) for v in cond_or_uncond)
            shape_key = tuple(int(v) for v in input_shape)
        except (TypeError, ValueError):
            return None

        if not shape_key:
            return None
        if not uuid_key or not cond_key:
            return None
        if len(uuid_key) != len(cond_key):
            return None
        return (uuid_key, cond_key, shape_key)

    def _ensure_stream_state(self, stream_key: StreamKey, run_id: Any) -> _StreamState:
        """Return the existing stream state or create a fresh one for this run."""
        state = self.stream_states.get(stream_key)
        if state is None or state.run_id != run_id:
            state = _StreamState(
                forecaster=self._make_forecaster(),
                curr_ws=float(self.cfg.window_size),
                run_id=run_id,
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
        """
        Bind one model call to an already-decided outer solver step.

        If no explicit outer-step context exists, the runtime fails open and the
        wrapper must execute a real denoiser forward.
        """
        del timesteps

        stream_key = self.stream_key(transformer_options, input_shape)
        self.last_info["last_stream_key"] = stream_key

        ctx, reason = self._solver_step_context(transformer_options)
        if ctx is None:
            self._disable_forecasting(reason or "invalid_solver_step_context")
            if self.cfg.debug:
                print(
                    f"[Spectrum SDXL] fail-open: reason={self._forecast_disable_reason} "
                    f"stream_key_present={stream_key is not None}"
                )
            self.last_info["actual_forward_count"] += 1
            return {
                "global_step_idx": None,
                "solver_step_id": None,
                "local_step_count": 0,
                "actual_forward": True,
                "run_id": None,
                "stream_key": stream_key,
                "forecast_safe": False,
                "finalized": False,
                "time_coord": None,
                "total_steps": self.num_steps(),
            }

        self._ensure_run_sync(ctx["run_id"])
        self.last_info["num_steps"] = ctx["total_steps"]
        expected_time_coord = self._expected_time_coord(transformer_options, ctx["solver_step_id"])
        if expected_time_coord is not None and not math.isclose(
            float(ctx["time_coord"]), float(expected_time_coord), rel_tol=0.0, abs_tol=1e-8
        ):
            self._disable_forecasting("solver-step time_coord did not match the active schedule")

        if stream_key is None:
            self._disable_forecasting("missing_stream_identity")
            if self.cfg.debug:
                print(
                    f"[Spectrum SDXL] fail-open: reason=missing_stream_identity "
                    f"run={ctx['run_id']} step={ctx['solver_step_id']}"
                )
            self.last_info["actual_forward_count"] += 1
            return {
                "global_step_idx": ctx["solver_step_id"],
                "solver_step_id": ctx["solver_step_id"],
                "local_step_count": 0,
                "actual_forward": True,
                "run_id": ctx["run_id"],
                "stream_key": None,
                "forecast_safe": False,
                "finalized": False,
                "time_coord": ctx["time_coord"],
                "total_steps": ctx["total_steps"],
            }

        state = self._ensure_stream_state(stream_key, ctx["run_id"])
        existing = state.decisions_by_solver_step.get(ctx["solver_step_id"])
        if existing is not None:
            if self._forecast_disabled and not existing.get("finalized", False):
                existing["forecast_safe"] = False
                existing["actual_forward"] = True
            self.last_info["curr_ws"] = state.curr_ws
            return existing

        local_step_count = state.local_step_count
        state.local_step_count += 1

        actual_forward = ctx["actual_forward"]
        tail_actual_only = self._is_tail_actual_step(ctx["solver_step_id"], ctx["total_steps"])
        if actual_forward is None:
            actual_forward = True
            if (
                ctx["solver_step_id"] >= self.cfg.warmup_steps
                and not tail_actual_only
            ):
                ws_floor = max(1, int(math.floor(state.curr_ws)))
                actual_forward = ((state.num_consecutive_cached_steps + 1) % ws_floor) == 0

        if self._forecast_disabled or tail_actual_only:
            actual_forward = True

        forecast_safe = not self._forecast_disabled and not tail_actual_only
        if (not actual_forward) and (not state.forecaster.ready()):
            actual_forward = True
            forecast_safe = False

        decision = {
            "global_step_idx": ctx["solver_step_id"],
            "solver_step_id": ctx["solver_step_id"],
            "local_step_count": local_step_count,
            "actual_forward": actual_forward,
            "run_id": ctx["run_id"],
            "stream_key": stream_key,
            "forecast_safe": forecast_safe,
            "finalized": False,
            "time_coord": ctx["time_coord"],
            "total_steps": ctx["total_steps"],
        }
        state.decisions_by_solver_step[ctx["solver_step_id"]] = decision
        self.last_info["curr_ws"] = state.curr_ws
        if self.cfg.debug:
            print(
                f"[Spectrum SDXL] begin "
                f"run={ctx['run_id']} step={ctx['solver_step_id']} "
                f"local={local_step_count} actual={actual_forward} "
                f"forecast_safe={forecast_safe} ready={state.forecaster.ready()} "
                f"ws={state.curr_ws:.3f} cached={state.num_consecutive_cached_steps} "
                f"stream={stream_key}"
            )
        return decision

    def finalize_step(self, stream_key: Optional[StreamKey], solver_step_id: Optional[int], used_forecast: bool) -> None:
        """Commit bookkeeping after the wrapper knows which path actually ran."""
        if stream_key is None or solver_step_id is None:
            return

        state = self.stream_states.get(stream_key)
        if state is None:
            return

        decision = state.decisions_by_solver_step.get(solver_step_id)
        if decision is None or decision.get("finalized", False):
            return

        if used_forecast:
            state.num_consecutive_cached_steps += 1
            self.last_info["forecasted_passes"] += 1
            decision["actual_forward"] = False
        else:
            if solver_step_id >= self.cfg.warmup_steps:
                state.curr_ws = round(state.curr_ws + float(self.cfg.flex_window), 3)
            state.num_consecutive_cached_steps = 0
            self.last_info["actual_forward_count"] += 1
            decision["actual_forward"] = True

        decision["finalized"] = True
        self.last_info["curr_ws"] = state.curr_ws
        if self.cfg.debug:
            print(
                f"[Spectrum SDXL] finalize "
                f"step={solver_step_id} used_forecast={used_forecast} "
                f"curr_ws={state.curr_ws:.3f} "
                f"forecasted_passes={self.last_info['forecasted_passes']} "
                f"actual_forward_count={self.last_info['actual_forward_count']}"
            )

    def observe_actual_feature(self, stream_key: Optional[StreamKey], solver_step_id: Optional[int], feature: torch.Tensor) -> None:
        """Record a real hidden feature for one explicit solver step once."""
        if stream_key is None or solver_step_id is None:
            return
        state = self.stream_states.get(stream_key)
        if state is None:
            return
        if solver_step_id in state.observed_solver_steps:
            return

        decision = state.decisions_by_solver_step.get(solver_step_id)
        if decision is None:
            return

        state.forecaster.update(float(decision["time_coord"]), feature)
        self.finalize_step(stream_key, solver_step_id, used_forecast=False)
        state.observed_solver_steps.add(solver_step_id)

    def predict_feature(self, stream_key: StreamKey, solver_step_id: int) -> torch.Tensor:
        """Predict the hidden feature for a specific stream and solver step."""
        state = self.stream_states.get(stream_key)
        if state is None:
            raise RuntimeError("Spectrum runtime has no state for the requested stream.")

        decision = state.decisions_by_solver_step.get(int(solver_step_id))
        if decision is None:
            raise RuntimeError("Spectrum runtime has no decision for the requested solver step.")

        return state.forecaster.predict(float(decision["time_coord"]), int(decision["total_steps"]))
