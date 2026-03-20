"""Lightweight runtime regression tests for the SDXL Spectrum scheduler."""

from __future__ import annotations

import sys
import types

import torch

from comfyui_spectrum_sdxl.config import SpectrumSDXLConfig
from comfyui_spectrum_sdxl.forecast import ChebyshevFeatureForecaster
from comfyui_spectrum_sdxl.runtime import SpectrumSDXLRuntime
from comfyui_spectrum_sdxl.sdxl import (
    _RUNTIME_KEY,
    _MODEL_TIME_COORD_KEY,
    _SpectrumModelFunctionWrapper,
    _SpectrumOuterStepController,
    _wrap_sdxl_unet_forward,
)


def _make_cfg() -> SpectrumSDXLConfig:
    """Create the default test configuration."""
    return SpectrumSDXLConfig(
        blend_weight=0.5,
        degree=4,
        ridge_lambda=0.1,
        window_size=2.0,
        flex_window=0.75,
        warmup_steps=5,
        min_fit_points=6,
    ).validated()


def _make_relaxed_cfg() -> SpectrumSDXLConfig:
    """Create a permissive configuration for tests that need forecastability."""
    return SpectrumSDXLConfig(
        blend_weight=0.5,
        degree=1,
        ridge_lambda=0.1,
        window_size=2.0,
        flex_window=0.75,
        warmup_steps=0,
        tail_actual_steps=0,
        min_fit_points=3,
    ).validated()


def _step_options(
    run_id,
    solver_step_id: int,
    time_coord: float,
    actual_forward: bool | None = None,
    total_steps: int = 6,
    uuid: str = "stream-a",
    cond: int = 0,
    model_time_coord: float | None = None,
    sample_sigmas: torch.Tensor | None = None,
):
    """Create explicit outer-step context for one logical stream."""
    options = {
        "spectrum_run_id": run_id,
        "spectrum_solver_step_id": solver_step_id,
        "spectrum_time_coord": time_coord,
        "spectrum_total_steps": total_steps,
        "uuids": [uuid],
        "cond_or_uncond": [cond],
    }
    if model_time_coord is not None:
        options[_MODEL_TIME_COORD_KEY] = model_time_coord
    if sample_sigmas is not None:
        options["sample_sigmas"] = sample_sigmas
    if actual_forward is not None:
        options["spectrum_actual_forward"] = actual_forward
    return options


def test_missing_solver_step_context_fails_open() -> None:
    """Without explicit outer-step context, forecasting must stay disabled."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    decision = runtime.begin_step({}, torch.tensor([0.0]), (2, 8, 4, 4))
    assert decision["actual_forward"] is True
    assert decision["forecast_safe"] is False
    assert decision["solver_step_id"] is None
    assert runtime.last_info["forecast_disabled"] is True
    assert runtime.last_info["forecast_disable_reason"] == "missing_solver_step_context"


def test_invalid_solver_step_context_fails_open() -> None:
    """Partially specified solver-step context must not be trusted."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    decision = runtime.begin_step(
        {
            "spectrum_run_id": "run-a",
            "spectrum_solver_step_id": 0,
            "spectrum_time_coord": 0.0,
            "spectrum_actual_forward": 0,
        },
        torch.tensor([0.0]),
        (2, 8, 4, 4),
    )
    assert decision["actual_forward"] is True
    assert decision["forecast_safe"] is False
    assert runtime.last_info["forecast_disable_reason"] == "invalid_solver_step_context"


def test_missing_stream_identity_fails_open() -> None:
    """Explicit step context still needs a stable logical stream identity."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    decision = runtime.begin_step(
        {
            "spectrum_run_id": "run-a",
            "spectrum_solver_step_id": 0,
            "spectrum_time_coord": 0.0,
            "spectrum_actual_forward": False,
            "spectrum_total_steps": 6,
        },
        torch.tensor([0.0]),
        (2, 8, 4, 4),
    )
    assert decision["actual_forward"] is True
    assert decision["forecast_safe"] is False
    assert decision["stream_key"] is None
    assert runtime.last_info["forecast_disable_reason"] == "missing_stream_identity"


def test_explicit_solver_step_context_allows_forecast() -> None:
    """Forecasting should work once the outer solver step is explicit."""
    runtime = SpectrumSDXLRuntime(_make_relaxed_cfg())

    first = runtime.begin_step(_step_options("run-a", 0, 0.0, True), torch.tensor([0.0]), (2, 8, 4, 4))
    runtime.observe_actual_feature(
        first["stream_key"],
        first["solver_step_id"],
        torch.full((2, 8, 4, 4), 1.0, dtype=torch.float16),
    )

    second = runtime.begin_step(_step_options("run-a", 1, 1.0, True), torch.tensor([1.0]), (2, 8, 4, 4))
    runtime.observe_actual_feature(
        second["stream_key"],
        second["solver_step_id"],
        torch.full((2, 8, 4, 4), 2.0, dtype=torch.float16),
    )

    third = runtime.begin_step(_step_options("run-a", 2, 2.0, True), torch.tensor([2.0]), (2, 8, 4, 4))
    runtime.observe_actual_feature(
        third["stream_key"],
        third["solver_step_id"],
        torch.full((2, 8, 4, 4), 3.0, dtype=torch.float16),
    )

    forecast = runtime.begin_step(_step_options("run-a", 3, 3.0, False), torch.tensor([3.0]), (2, 8, 4, 4))
    assert forecast["actual_forward"] is False
    assert forecast["forecast_safe"] is True

    pred = runtime.predict_feature(forecast["stream_key"], forecast["solver_step_id"])
    assert pred.shape == (2, 8, 4, 4)
    assert torch.isfinite(pred).all()

    runtime.finalize_step(forecast["stream_key"], forecast["solver_step_id"], used_forecast=True)
    assert runtime.last_info["forecasted_passes"] == 1


def test_runtime_prefers_sigma_coord_for_forecasting_even_when_model_time_is_present() -> None:
    """The forecast axis should stay on raw sigma even when model-time metadata is present."""
    runtime = SpectrumSDXLRuntime(_make_relaxed_cfg())
    sample_sigmas = torch.tensor([14.0, 7.0, 3.0, 1.5, 0.75, 0.25, 0.0], dtype=torch.float32)

    first = runtime.begin_step(
        _step_options(
            "run-a",
            0,
            14.0,
            True,
            total_steps=8,
            model_time_coord=100.0,
            sample_sigmas=sample_sigmas,
        ),
        torch.tensor([100.0]),
        (2, 8, 4, 4),
    )
    runtime.observe_actual_feature(
        first["stream_key"],
        first["solver_step_id"],
        torch.full((2, 8, 4, 4), 1.0, dtype=torch.float16),
    )

    second = runtime.begin_step(
        _step_options(
            "run-a",
            1,
            7.0,
            True,
            total_steps=8,
            model_time_coord=200.0,
            sample_sigmas=sample_sigmas,
        ),
        torch.tensor([200.0]),
        (2, 8, 4, 4),
    )
    runtime.observe_actual_feature(
        second["stream_key"],
        second["solver_step_id"],
        torch.full((2, 8, 4, 4), 2.0, dtype=torch.float16),
    )

    third = runtime.begin_step(
        _step_options(
            "run-a",
            2,
            3.0,
            True,
            total_steps=8,
            model_time_coord=300.0,
            sample_sigmas=sample_sigmas,
        ),
        torch.tensor([300.0]),
        (2, 8, 4, 4),
    )
    runtime.observe_actual_feature(
        third["stream_key"],
        third["solver_step_id"],
        torch.full((2, 8, 4, 4), 3.0, dtype=torch.float16),
    )

    forecast = runtime.begin_step(
        _step_options(
            "run-a",
            3,
            1.5,
            False,
            total_steps=8,
            model_time_coord=400.0,
            sample_sigmas=sample_sigmas,
        ),
        torch.tensor([400.0]),
        (2, 8, 4, 4),
    )

    assert forecast["time_coord"] == 1.5
    assert forecast["model_time_coord"] == 400.0
    assert forecast["sigma_coord"] == 1.5
    assert forecast["actual_forward"] is False
    assert forecast["forecast_safe"] is True

    pred = runtime.predict_feature(forecast["stream_key"], forecast["solver_step_id"])
    assert pred.shape == (2, 8, 4, 4)
    assert torch.isfinite(pred).all()

    state = runtime.stream_states[forecast["stream_key"]]
    fit_cache = state.forecaster._fit_cache
    assert fit_cache is not None
    assert torch.allclose(torch.tensor(fit_cache.coord_min), torch.tensor(0.25), atol=1e-6)
    assert torch.allclose(torch.tensor(fit_cache.coord_max), torch.tensor(14.0), atol=1e-6)


def test_runtime_prefers_sigma_coord_for_forecasting_without_sample_sigmas() -> None:
    """Fallback bounds should stay on observed sigma history when schedule bounds are absent."""
    runtime = SpectrumSDXLRuntime(_make_relaxed_cfg())

    first = runtime.begin_step(
        _step_options(
            "run-a",
            0,
            14.0,
            True,
            total_steps=8,
            model_time_coord=100.0,
        ),
        torch.tensor([100.0]),
        (2, 8, 4, 4),
    )
    runtime.observe_actual_feature(
        first["stream_key"],
        first["solver_step_id"],
        torch.full((2, 8, 4, 4), 1.0, dtype=torch.float16),
    )

    second = runtime.begin_step(
        _step_options(
            "run-a",
            1,
            7.0,
            True,
            total_steps=8,
            model_time_coord=200.0,
        ),
        torch.tensor([200.0]),
        (2, 8, 4, 4),
    )
    runtime.observe_actual_feature(
        second["stream_key"],
        second["solver_step_id"],
        torch.full((2, 8, 4, 4), 2.0, dtype=torch.float16),
    )

    third = runtime.begin_step(
        _step_options(
            "run-a",
            2,
            3.0,
            True,
            total_steps=8,
            model_time_coord=300.0,
        ),
        torch.tensor([300.0]),
        (2, 8, 4, 4),
    )
    runtime.observe_actual_feature(
        third["stream_key"],
        third["solver_step_id"],
        torch.full((2, 8, 4, 4), 3.0, dtype=torch.float16),
    )

    forecast = runtime.begin_step(
        _step_options(
            "run-a",
            3,
            1.5,
            False,
            total_steps=8,
            model_time_coord=400.0,
        ),
        torch.tensor([400.0]),
        (2, 8, 4, 4),
    )

    assert forecast["time_coord"] == 1.5
    assert forecast["model_time_coord"] == 400.0
    assert forecast["sigma_coord"] == 1.5
    assert forecast["actual_forward"] is False
    assert forecast["forecast_safe"] is True

    pred = runtime.predict_feature(forecast["stream_key"], forecast["solver_step_id"])
    assert pred.shape == (2, 8, 4, 4)
    assert torch.isfinite(pred).all()

    state = runtime.stream_states[forecast["stream_key"]]
    fit_cache = state.forecaster._fit_cache
    assert fit_cache is not None
    assert torch.allclose(torch.tensor(fit_cache.coord_min), torch.tensor(3.0), atol=1e-6)
    assert torch.allclose(torch.tensor(fit_cache.coord_max), torch.tensor(14.0), atol=1e-6)


def test_explicit_solver_step_context_without_decision_still_schedules() -> None:
    """Outer-step context alone should be enough once the controller owns step ids."""
    runtime = SpectrumSDXLRuntime(_make_relaxed_cfg())

    first = runtime.begin_step(_step_options("run-a", 0, 0.0), torch.tensor([0.0]), (2, 8, 4, 4))
    assert first["actual_forward"] is True
    runtime.observe_actual_feature(
        first["stream_key"],
        first["solver_step_id"],
        torch.full((2, 8, 4, 4), 1.0, dtype=torch.float16),
    )

    second = runtime.begin_step(_step_options("run-a", 1, 1.0), torch.tensor([1.0]), (2, 8, 4, 4))
    assert second["actual_forward"] is True
    runtime.observe_actual_feature(
        second["stream_key"],
        second["solver_step_id"],
        torch.full((2, 8, 4, 4), 2.0, dtype=torch.float16),
    )

    third = runtime.begin_step(_step_options("run-a", 2, 2.0), torch.tensor([2.0]), (2, 8, 4, 4))
    assert third["actual_forward"] is True
    runtime.observe_actual_feature(
        third["stream_key"],
        third["solver_step_id"],
        torch.full((2, 8, 4, 4), 3.0, dtype=torch.float16),
    )

    fourth = runtime.begin_step(_step_options("run-a", 3, 3.0), torch.tensor([3.0]), (2, 8, 4, 4))
    assert fourth["actual_forward"] is False
    assert fourth["forecast_safe"] is True


def test_outer_step_controller_injects_context_and_resets_runs() -> None:
    """The sampler hook should stamp stable run and solver-step ids."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    controller = _SpectrumOuterStepController(
        runtime=runtime,
        delegate=lambda args: args["model_options"]["transformer_options"].copy(),
    )

    sample_sigmas = torch.linspace(1.0, 0.0, 4)
    model_options = {"transformer_options": {"sample_sigmas": sample_sigmas}}
    first = controller({"model_options": model_options, "sigma": torch.tensor([float(sample_sigmas[0].item())])})
    second = controller({"model_options": model_options, "sigma": torch.tensor([float(sample_sigmas[1].item())])})
    assert first["spectrum_run_id"] == second["spectrum_run_id"]
    assert first["spectrum_solver_step_id"] == 0
    assert second["spectrum_solver_step_id"] == 1
    assert second["spectrum_total_steps"] == 3

    next_sample_sigmas = torch.linspace(1.0, 0.0, 5)
    next_model_options = {"transformer_options": {"sample_sigmas": next_sample_sigmas}}
    restarted = controller(
        {"model_options": next_model_options, "sigma": torch.tensor([float(next_sample_sigmas[0].item())])}
    )
    assert restarted["spectrum_run_id"] != first["spectrum_run_id"]
    assert restarted["spectrum_solver_step_id"] == 0
    assert restarted["spectrum_total_steps"] == 4


def test_outer_step_controller_same_object_restart_resets_run_state() -> None:
    """Reusing the exact same schedule tensor for a new run must reset controller state."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    controller = _SpectrumOuterStepController(
        runtime=runtime,
        delegate=lambda args: args["model_options"]["transformer_options"].copy(),
    )

    sample_sigmas = torch.linspace(1.0, 0.0, 6)
    model_options = {"transformer_options": {"sample_sigmas": sample_sigmas}}

    first = controller({"model_options": model_options, "sigma": torch.tensor([float(sample_sigmas[0].item())])})
    second = controller({"model_options": model_options, "sigma": torch.tensor([float(sample_sigmas[1].item())])})
    restarted = controller({"model_options": model_options, "sigma": torch.tensor([float(sample_sigmas[0].item())])})

    assert first["spectrum_run_id"] == second["spectrum_run_id"]
    assert second["spectrum_solver_step_id"] == 1
    assert restarted["spectrum_run_id"] != first["spectrum_run_id"]
    assert restarted["spectrum_solver_step_id"] == 0
    assert torch.allclose(torch.tensor(restarted["spectrum_time_coord"]), torch.tensor(1.0), atol=1e-6)


def test_outer_step_controller_uses_raw_sigma_time_coord() -> None:
    """The controller must stamp raw sigma values as the forecast coordinate."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    controller = _SpectrumOuterStepController(
        runtime=runtime,
        delegate=lambda args: args["model_options"]["transformer_options"].copy(),
    )

    model_options = {
        "transformer_options": {
            "sample_sigmas": torch.tensor([14.0, 7.0, 1.5, 0.0], dtype=torch.float32),
        }
    }

    first = controller({"model_options": model_options, "sigma": torch.tensor([14.0])})
    second = controller({"model_options": model_options, "sigma": torch.tensor([7.0])})

    assert first["spectrum_solver_step_id"] == 0
    assert second["spectrum_solver_step_id"] == 1
    assert torch.allclose(torch.tensor(first["spectrum_time_coord"]), torch.tensor(14.0), atol=1e-6)
    assert torch.allclose(torch.tensor(second["spectrum_time_coord"]), torch.tensor(7.0), atol=1e-6)


def test_runtime_disables_forecasting_when_time_coord_does_not_match_schedule() -> None:
    """A mismatched raw sigma coordinate must fail open instead of forecasting."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    sample_sigmas = torch.tensor([14.0, 7.0, 1.5, 0.0], dtype=torch.float32)
    decision = runtime.begin_step(
        {
            "sample_sigmas": sample_sigmas,
            "uuids": ("u0", "u1"),
            "cond_or_uncond": (1, 0),
            "spectrum_run_id": "run-a",
            "spectrum_solver_step_id": 1,
            "spectrum_time_coord": 123.0,
            "spectrum_total_steps": 3,
        },
        torch.tensor([7.0]),
        (2, 8, 4, 4),
    )
    assert decision["actual_forward"] is True
    assert decision["forecast_safe"] is False
    assert runtime.last_info["forecast_disable_reason"] == "solver-step sigma_coord did not match the active schedule"


def test_model_function_wrapper_injects_context_for_bypassed_guider_path() -> None:
    """The compat wrapper must stamp context when the sampler hook is skipped."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    controller = _SpectrumOuterStepController(runtime=runtime)
    wrapper = _SpectrumModelFunctionWrapper(controller=controller)

    def apply_model(input_x, timestep, **c):
        del input_x, timestep
        return c["transformer_options"].copy()

    sample_sigmas = torch.tensor([14.0, 7.0, 1.5, 0.0], dtype=torch.float32)
    first = wrapper(
        apply_model,
        {
            "c": {"transformer_options": {"sample_sigmas": sample_sigmas}},
            "timestep": torch.tensor([14.0]),
            "input": torch.zeros((1, 4, 8, 8)),
        },
    )
    second = wrapper(
        apply_model,
        {
            "c": {"transformer_options": {"sample_sigmas": sample_sigmas}},
            "timestep": torch.tensor([7.0]),
            "input": torch.zeros((1, 4, 8, 8)),
        },
    )

    assert first["spectrum_run_id"] == second["spectrum_run_id"]
    assert first["spectrum_solver_step_id"] == 0
    assert second["spectrum_solver_step_id"] == 1
    assert second["spectrum_total_steps"] == 3


def test_model_function_wrapper_reuses_solver_step_for_repeated_same_sigma_subcalls() -> None:
    """Repeated guider subcalls for one outer step must not advance the solver-step id."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    controller = _SpectrumOuterStepController(runtime=runtime)
    wrapper = _SpectrumModelFunctionWrapper(controller=controller)

    def apply_model(input_x, timestep, **c):
        del input_x, timestep
        return c["transformer_options"].copy()

    sample_sigmas = torch.tensor([14.0, 7.0, 1.5, 0.0], dtype=torch.float32)
    first = wrapper(
        apply_model,
        {
            "c": {"transformer_options": {"sample_sigmas": sample_sigmas}},
            "timestep": torch.tensor([14.0]),
            "input": torch.zeros((1, 4, 8, 8)),
        },
    )
    repeated = wrapper(
        apply_model,
        {
            "c": {"transformer_options": {"sample_sigmas": sample_sigmas}},
            "timestep": torch.tensor([14.0]),
            "input": torch.zeros((1, 4, 8, 8)),
        },
    )
    next_step = wrapper(
        apply_model,
        {
            "c": {"transformer_options": {"sample_sigmas": sample_sigmas}},
            "timestep": torch.tensor([7.0]),
            "input": torch.zeros((1, 4, 8, 8)),
        },
    )

    assert repeated["spectrum_run_id"] == first["spectrum_run_id"]
    assert repeated["spectrum_solver_step_id"] == first["spectrum_solver_step_id"] == 0
    assert next_step["spectrum_solver_step_id"] == 1


def test_model_function_wrapper_same_object_restart_resets_run_state() -> None:
    """Returning to the first sigma with the same schedule tensor must start a new run."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    controller = _SpectrumOuterStepController(runtime=runtime)
    wrapper = _SpectrumModelFunctionWrapper(controller=controller)

    def apply_model(input_x, timestep, **c):
        del input_x, timestep
        return c["transformer_options"].copy()

    sample_sigmas = torch.linspace(1.0, 0.0, 6)
    first = wrapper(
        apply_model,
        {
            "c": {"transformer_options": {"sample_sigmas": sample_sigmas}},
            "timestep": torch.tensor([float(sample_sigmas[0].item())]),
            "input": torch.zeros((1, 4, 8, 8)),
        },
    )
    second = wrapper(
        apply_model,
        {
            "c": {"transformer_options": {"sample_sigmas": sample_sigmas}},
            "timestep": torch.tensor([float(sample_sigmas[1].item())]),
            "input": torch.zeros((1, 4, 8, 8)),
        },
    )
    restarted = wrapper(
        apply_model,
        {
            "c": {"transformer_options": {"sample_sigmas": sample_sigmas}},
            "timestep": torch.tensor([float(sample_sigmas[0].item())]),
            "input": torch.zeros((1, 4, 8, 8)),
        },
    )

    assert first["spectrum_run_id"] == second["spectrum_run_id"]
    assert second["spectrum_solver_step_id"] == 1
    assert restarted["spectrum_run_id"] != first["spectrum_run_id"]
    assert restarted["spectrum_solver_step_id"] == 0
    assert torch.allclose(torch.tensor(restarted["spectrum_time_coord"]), torch.tensor(1.0), atol=1e-6)


def test_model_function_wrapper_preserves_existing_context() -> None:
    """The compat wrapper must not overwrite context from the sampler-layer hook."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    controller = _SpectrumOuterStepController(runtime=runtime)
    wrapper = _SpectrumModelFunctionWrapper(controller=controller)

    def apply_model(input_x, timestep, **c):
        del input_x, timestep
        return c["transformer_options"].copy()

    transformer_options = {
        "sample_sigmas": torch.tensor([14.0, 7.0, 1.5, 0.0], dtype=torch.float32),
        "spectrum_run_id": 99,
        "spectrum_solver_step_id": 4,
        "spectrum_time_coord": 4.0,
        "spectrum_total_steps": 12,
    }
    seen = wrapper(
        apply_model,
        {
            "c": {"transformer_options": transformer_options},
            "timestep": torch.tensor([7.0]),
            "input": torch.zeros((1, 4, 8, 8)),
        },
    )

    assert seen["spectrum_run_id"] == 99
    assert seen["spectrum_solver_step_id"] == 4
    assert seen["spectrum_time_coord"] == 4.0
    assert seen["spectrum_total_steps"] == 12


def test_model_function_wrapper_injects_context_for_delegate_internal_apply_calls() -> None:
    """Delegate-internal apply_model calls must still pass through Spectrum injection."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    controller = _SpectrumOuterStepController(runtime=runtime)

    captured = []

    def apply_model(input_x, timestep, **c):
        del input_x
        captured.append(
            {
                "timestep": float(timestep.flatten()[0].item()),
                "transformer_options": c["transformer_options"].copy(),
            }
        )
        return timestep

    def delegate(wrapped_apply_model, args):
        base = args["c"].get("transformer_options", {})
        t0 = args["timestep"]
        t1 = torch.tensor([7.0], dtype=t0.dtype)
        wrapped_apply_model(args["input"], t0, transformer_options={"sample_sigmas": base["sample_sigmas"]})
        wrapped_apply_model(args["input"], t0, transformer_options={"sample_sigmas": base["sample_sigmas"]})
        wrapped_apply_model(args["input"], t1, transformer_options={"sample_sigmas": base["sample_sigmas"]})
        return captured

    wrapper = _SpectrumModelFunctionWrapper(controller=controller, delegate=delegate)
    sample_sigmas = torch.tensor([14.0, 7.0, 1.5, 0.0], dtype=torch.float32)
    result = wrapper(
        apply_model,
        {
            "c": {"transformer_options": {"sample_sigmas": sample_sigmas}},
            "timestep": torch.tensor([14.0], dtype=torch.float32),
            "input": torch.zeros((1, 4, 8, 8)),
        },
    )

    assert len(result) == 3
    first = result[0]["transformer_options"]
    second = result[1]["transformer_options"]
    third = result[2]["transformer_options"]

    assert first["spectrum_solver_step_id"] == 0
    assert second["spectrum_solver_step_id"] == 0
    assert third["spectrum_solver_step_id"] == 1
    assert first["spectrum_run_id"] == second["spectrum_run_id"] == third["spectrum_run_id"]


def test_model_function_wrapper_delegate_forces_actual_forward_flag() -> None:
    """Delegated guider paths must be forced onto real forwards to avoid forecast aliasing."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    controller = _SpectrumOuterStepController(runtime=runtime)

    seen = {}

    def apply_model(input_x, timestep, **c):
        del input_x, timestep
        seen.update(c["transformer_options"])
        return 0

    def delegate(wrapped_apply_model, args):
        options = args["c"]["transformer_options"]
        return wrapped_apply_model(
            args["input"],
            args["timestep"],
            transformer_options={"sample_sigmas": options["sample_sigmas"]},
        )

    wrapper = _SpectrumModelFunctionWrapper(controller=controller, delegate=delegate)
    wrapper(
        apply_model,
        {
            "c": {"transformer_options": {"sample_sigmas": torch.tensor([14.0, 7.0, 1.5, 0.0])}},
            "timestep": torch.tensor([14.0]),
            "input": torch.zeros((1, 4, 8, 8)),
        },
    )

    assert seen["spectrum_actual_forward"] is True


def test_forecast_request_before_history_fails_open_per_step() -> None:
    """Outer code may request a forecast early; the runtime must still fail open."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    decision = runtime.begin_step(_step_options("run-a", 0, 0.0, False), torch.tensor([0.0]), (2, 8, 4, 4))
    assert decision["actual_forward"] is True
    assert decision["forecast_safe"] is False


def test_duplicate_actual_updates_are_deduped() -> None:
    """A repeated actual write for one stream/step must only be recorded once."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    decision = runtime.begin_step(_step_options("run-a", 0, 0.0, True), torch.tensor([0.0]), (2, 8, 4, 4))
    assert decision["actual_forward"] is True

    first_feature = torch.full((2, 8, 4, 4), 1.0, dtype=torch.float16)
    second_feature = torch.full((2, 8, 4, 4), 2.0, dtype=torch.float16)
    runtime.observe_actual_feature(decision["stream_key"], decision["solver_step_id"], first_feature)
    runtime.observe_actual_feature(decision["stream_key"], decision["solver_step_id"], second_feature)

    state = runtime.stream_states[decision["stream_key"]]
    assert len(state.forecaster.history) == 1
    _, stored_feature = state.forecaster.history[0]
    assert torch.equal(stored_feature, first_feature)


def test_forecast_fallback_commits_actual_bookkeeping() -> None:
    """A rejected forecast should not count as a cached pass."""
    runtime = SpectrumSDXLRuntime(_make_relaxed_cfg())

    for step_id, value in ((0, 1.0), (1, 2.0), (2, 3.0)):
        decision = runtime.begin_step(
            _step_options("run-a", step_id, float(step_id), True),
            torch.tensor([float(step_id)]),
            (2, 8, 4, 4),
        )
        runtime.observe_actual_feature(
            decision["stream_key"],
            decision["solver_step_id"],
            torch.full((2, 8, 4, 4), value, dtype=torch.float16),
        )

    forecast = runtime.begin_step(_step_options("run-a", 3, 3.0, False), torch.tensor([3.0]), (2, 8, 4, 4))
    assert forecast["actual_forward"] is False

    runtime.finalize_step(forecast["stream_key"], forecast["solver_step_id"], used_forecast=False)

    state = runtime.stream_states[forecast["stream_key"]]
    assert runtime.last_info["forecasted_passes"] == 0
    assert runtime.last_info["actual_forward_count"] == 4
    assert state.num_consecutive_cached_steps == 0
    assert state.decisions_by_solver_step[forecast["solver_step_id"]]["actual_forward"] is True
    assert state.decisions_by_solver_step[forecast["solver_step_id"]]["finalized"] is True


def test_run_id_switch_resets_stream_state() -> None:
    """A new explicit run id must clear stale per-stream history."""
    runtime = SpectrumSDXLRuntime(_make_cfg())

    first = runtime.begin_step(_step_options("run-a", 0, 0.0, True), torch.tensor([0.0]), (2, 8, 4, 4))
    runtime.observe_actual_feature(
        first["stream_key"],
        first["solver_step_id"],
        torch.full((2, 8, 4, 4), 4.0, dtype=torch.float16),
    )
    first_state = runtime.stream_states[first["stream_key"]]
    assert len(first_state.forecaster.history) == 1

    restarted = runtime.begin_step(_step_options("run-b", 0, 0.0, True), torch.tensor([0.0]), (2, 8, 4, 4))
    restarted_state = runtime.stream_states[restarted["stream_key"]]
    assert restarted_state is not first_state
    assert restarted_state.decisions_by_solver_step.keys() == {0}
    assert restarted_state.observed_solver_steps == set()
    assert restarted_state.forecaster.history == []
    assert runtime.last_info["run_id"] == "run-b"


def test_forecaster_chebyshev_fit_falls_back_to_observed_sigma_range() -> None:
    """Without an explicit schedule span, the fit must fall back to observed sigma bounds."""
    forecaster = ChebyshevFeatureForecaster(
        degree=2,
        ridge_lambda=0.1,
        blend_weight=1.0,
        history_size=10,
    )
    forecaster.update(10.0, torch.tensor([10.0]))
    forecaster.update(9.0, torch.tensor([9.0]))

    pred = forecaster.predict(1.0, 11)

    assert forecaster._fit_cache is not None
    assert torch.allclose(torch.tensor(forecaster._fit_cache.coord_min), torch.tensor(9.0), atol=1e-6)
    assert torch.allclose(torch.tensor(forecaster._fit_cache.coord_max), torch.tensor(10.0), atol=1e-6)
    assert torch.isfinite(pred).all()


def test_runtime_fallback_coord_bounds_ignore_unobserved_candidate_step() -> None:
    """Fallback bounds must come only from observed feature history, not pending steps."""
    runtime = SpectrumSDXLRuntime(_make_relaxed_cfg())

    first = runtime.begin_step(_step_options("run-a", 0, 10.0, True), torch.tensor([10.0]), (1, 1, 1, 1))
    runtime.observe_actual_feature(
        first["stream_key"],
        first["solver_step_id"],
        torch.full((1, 1, 1, 1), 10.0, dtype=torch.float16),
    )

    second = runtime.begin_step(_step_options("run-a", 1, 1.0, True), torch.tensor([1.0]), (1, 1, 1, 1))
    state = runtime.stream_states[second["stream_key"]]

    assert state.forecaster._coord_bounds is not None
    assert torch.allclose(torch.tensor(state.forecaster._coord_bounds[0]), torch.tensor(10.0), atol=1e-6)
    assert torch.allclose(torch.tensor(state.forecaster._coord_bounds[1]), torch.tensor(10.0), atol=1e-6)


def test_runtime_uses_schedule_wide_sigma_bounds_for_forecast_fit() -> None:
    """When sample_sigmas are available, the fit must use the full active schedule span."""
    cfg = _make_relaxed_cfg()
    cfg.tail_actual_steps = 0
    runtime = SpectrumSDXLRuntime(cfg.validated())
    sample_sigmas = torch.tensor([10.0, 9.0, 1.0, 0.0], dtype=torch.float32)

    for solver_step_id, time_coord, value in ((0, 10.0, 10.0), (1, 9.0, 9.0), (2, 1.0, 1.0)):
        decision = runtime.begin_step(
            {
                "sample_sigmas": sample_sigmas,
                "uuids": ("u0",),
                "cond_or_uncond": (0,),
                "spectrum_run_id": "run-a",
                "spectrum_solver_step_id": solver_step_id,
                "spectrum_time_coord": time_coord,
                "spectrum_actual_forward": True,
                "spectrum_total_steps": 3,
            },
            torch.tensor([time_coord]),
            (1, 1, 1, 1),
        )
        runtime.observe_actual_feature(
            decision["stream_key"],
            decision["solver_step_id"],
            torch.full((1, 1, 1, 1), value, dtype=torch.float16),
        )

    forecast = runtime.begin_step(
        {
            "sample_sigmas": sample_sigmas,
            "uuids": ("u0",),
            "cond_or_uncond": (0,),
            "spectrum_run_id": "run-a",
            "spectrum_solver_step_id": 3,
            "spectrum_time_coord": 1.0,
            "spectrum_actual_forward": False,
            "spectrum_total_steps": 4,
        },
        torch.tensor([1.0]),
        (1, 1, 1, 1),
    )
    assert forecast["actual_forward"] is False
    assert forecast["forecast_safe"] is True

    pred = runtime.predict_feature(forecast["stream_key"], forecast["solver_step_id"])
    state = runtime.stream_states[forecast["stream_key"]]
    assert state.forecaster._fit_cache is not None
    assert torch.allclose(torch.tensor(state.forecaster._fit_cache.coord_min), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(torch.tensor(state.forecaster._fit_cache.coord_max), torch.tensor(10.0), atol=1e-6)
    assert torch.isfinite(pred).all()


def test_runtime_holds_forecasting_until_conservative_min_fit_points() -> None:
    """The runtime must stay real until the configured conservative fit threshold is met."""
    runtime = SpectrumSDXLRuntime(
        SpectrumSDXLConfig(
            blend_weight=0.5,
            degree=1,
            ridge_lambda=0.1,
            window_size=2.0,
            flex_window=0.75,
            warmup_steps=0,
            tail_actual_steps=0,
            min_fit_points=6,
        ).validated()
    )

    for step_id in range(5):
        decision = runtime.begin_step(
            _step_options("run-a", step_id, float(step_id), False, total_steps=10),
            torch.tensor([float(step_id)]),
            (1, 1, 1, 1),
        )
        assert decision["actual_forward"] is True
        assert decision["forecast_safe"] is False
        runtime.observe_actual_feature(
            decision["stream_key"],
            decision["solver_step_id"],
            torch.full((1, 1, 1, 1), float(step_id), dtype=torch.float16),
        )

    blocked = runtime.begin_step(
        _step_options("run-a", 5, 5.0, False, total_steps=10),
        torch.tensor([5.0]),
        (1, 1, 1, 1),
    )
    assert blocked["actual_forward"] is True
    assert blocked["forecast_safe"] is False

    runtime.observe_actual_feature(
        blocked["stream_key"],
        blocked["solver_step_id"],
        torch.full((1, 1, 1, 1), 5.0, dtype=torch.float16),
    )

    ready = runtime.begin_step(
        _step_options("run-a", 6, 6.0, False, total_steps=10),
        torch.tensor([6.0]),
        (1, 1, 1, 1),
    )
    assert ready["actual_forward"] is False
    assert ready["forecast_safe"] is True


def test_runtime_blocks_forecast_when_recent_validation_error_is_bad() -> None:
    """A bad recent one-step holdout must fail open even after the count gate passes."""
    runtime = SpectrumSDXLRuntime(
        SpectrumSDXLConfig(
            blend_weight=0.5,
            degree=1,
            ridge_lambda=0.1,
            window_size=2.0,
            flex_window=0.75,
            warmup_steps=0,
            tail_actual_steps=0,
            min_fit_points=3,
        ).validated()
    )

    for step_id, value in ((0, 0.0), (1, 0.0), (2, 100.0)):
        decision = runtime.begin_step(
            _step_options("run-a", step_id, float(step_id), True, total_steps=10),
            torch.tensor([float(step_id)]),
            (1, 1, 1, 1),
        )
        runtime.observe_actual_feature(
            decision["stream_key"],
            decision["solver_step_id"],
            torch.full((1, 1, 1, 1), value, dtype=torch.float16),
        )

    blocked = runtime.begin_step(
        _step_options("run-a", 3, 3.0, False, total_steps=10),
        torch.tensor([3.0]),
        (1, 1, 1, 1),
    )
    assert blocked["actual_forward"] is True
    assert blocked["forecast_safe"] is False
    assert blocked["recent_validation_rel_l2"] is not None
    assert blocked["recent_validation_rel_l2"] > 0.35


def test_chebyshev_prediction_varies_with_time_coord() -> None:
    """The pure Chebyshev branch must not collapse all future steps to one value."""
    forecaster = ChebyshevFeatureForecaster(
        degree=1,
        ridge_lambda=0.0,
        blend_weight=1.0,
        history_size=10,
    )
    forecaster.update(0.0, torch.tensor([0.0]))
    forecaster.update(1.0, torch.tensor([1.0]))

    pred_two = forecaster.predict(2.0, 4)
    pred_three = forecaster.predict(3.0, 4)

    assert torch.allclose(pred_two, torch.tensor([2.0]), atol=1e-5)
    assert torch.allclose(pred_three, torch.tensor([3.0]), atol=1e-5)


def test_tail_actual_steps_force_real_tail_even_with_ready_history() -> None:
    """The configured tail must stay on the real path even when forecasting is ready."""
    cfg = _make_relaxed_cfg()
    cfg.tail_actual_steps = 1
    runtime = SpectrumSDXLRuntime(cfg.validated())
    coords = (10.0, 8.0, 6.0, 4.0, 2.0)

    first = runtime.begin_step(
        _step_options("run-a", 0, coords[0], True, total_steps=5),
        torch.tensor([coords[0]]),
        (2, 8, 4, 4),
    )
    runtime.observe_actual_feature(
        first["stream_key"],
        first["solver_step_id"],
        torch.full((2, 8, 4, 4), 1.0, dtype=torch.float16),
    )

    second = runtime.begin_step(
        _step_options("run-a", 1, coords[1], True, total_steps=5),
        torch.tensor([coords[1]]),
        (2, 8, 4, 4),
    )
    runtime.observe_actual_feature(
        second["stream_key"],
        second["solver_step_id"],
        torch.full((2, 8, 4, 4), 2.0, dtype=torch.float16),
    )

    third_actual = runtime.begin_step(
        _step_options("run-a", 2, coords[2], True, total_steps=5),
        torch.tensor([coords[2]]),
        (2, 8, 4, 4),
    )
    runtime.observe_actual_feature(
        third_actual["stream_key"],
        third_actual["solver_step_id"],
        torch.full((2, 8, 4, 4), 3.0, dtype=torch.float16),
    )

    third = runtime.begin_step(
        _step_options("run-a", 3, coords[3], False, total_steps=5),
        torch.tensor([coords[3]]),
        (2, 8, 4, 4),
    )
    assert third["solver_step_id"] == 3
    assert third["actual_forward"] is False
    assert third["forecast_safe"] is True
    runtime.finalize_step(third["stream_key"], third["solver_step_id"], used_forecast=True)

    fourth = runtime.begin_step(
        _step_options("run-a", 4, coords[4], False, total_steps=5),
        torch.tensor([coords[4]]),
        (2, 8, 4, 4),
    )
    assert fourth["solver_step_id"] == 4
    assert fourth["actual_forward"] is True
    assert fourth["forecast_safe"] is False
    runtime.observe_actual_feature(
        fourth["stream_key"],
        fourth["solver_step_id"],
        torch.full((2, 8, 4, 4), 4.0, dtype=torch.float16),
    )


def test_tail_actual_steps_zero_preserves_existing_scheduler_behavior() -> None:
    """Disabling the tail override must keep the old scheduling path intact."""
    cfg = _make_relaxed_cfg()
    cfg.tail_actual_steps = 0
    runtime = SpectrumSDXLRuntime(cfg.validated())
    coords = (10.0, 8.0, 6.0, 4.0, 2.0)

    first = runtime.begin_step(
        _step_options("run-a", 0, coords[0], True, total_steps=5),
        torch.tensor([coords[0]]),
        (2, 8, 4, 4),
    )
    runtime.observe_actual_feature(
        first["stream_key"],
        first["solver_step_id"],
        torch.full((2, 8, 4, 4), 1.0, dtype=torch.float16),
    )

    second = runtime.begin_step(
        _step_options("run-a", 1, coords[1], True, total_steps=5),
        torch.tensor([coords[1]]),
        (2, 8, 4, 4),
    )
    runtime.observe_actual_feature(
        second["stream_key"],
        second["solver_step_id"],
        torch.full((2, 8, 4, 4), 2.0, dtype=torch.float16),
    )

    third_actual = runtime.begin_step(
        _step_options("run-a", 2, coords[2], True, total_steps=5),
        torch.tensor([coords[2]]),
        (2, 8, 4, 4),
    )
    runtime.observe_actual_feature(
        third_actual["stream_key"],
        third_actual["solver_step_id"],
        torch.full((2, 8, 4, 4), 3.0, dtype=torch.float16),
    )

    third = runtime.begin_step(
        _step_options("run-a", 3, coords[3], False, total_steps=5),
        torch.tensor([coords[3]]),
        (2, 8, 4, 4),
    )
    assert third["solver_step_id"] == 3
    assert third["actual_forward"] is False
    assert third["forecast_safe"] is True
    runtime.finalize_step(third["stream_key"], third["solver_step_id"], used_forecast=True)

    fourth = runtime.begin_step(
        _step_options("run-a", 4, coords[4], False, total_steps=5),
        torch.tensor([coords[4]]),
        (2, 8, 4, 4),
    )
    assert fourth["solver_step_id"] == 4
    assert fourth["actual_forward"] is False
    assert fourth["forecast_safe"] is True
    runtime.finalize_step(fourth["stream_key"], fourth["solver_step_id"], used_forecast=True)


def test_tail_actual_steps_greater_than_total_steps_forces_all_steps_real() -> None:
    """A tail longer than the schedule must force every step onto the real path."""
    cfg = _make_cfg()
    cfg.tail_actual_steps = 10
    runtime = SpectrumSDXLRuntime(cfg.validated())
    coords = (10.0, 8.0, 6.0, 4.0, 2.0)

    for solver_step_id in range(5):
        step = runtime.begin_step(
            _step_options("run-a", solver_step_id, coords[solver_step_id], False, total_steps=5),
            torch.tensor([coords[solver_step_id]]),
            (2, 8, 4, 4),
        )
        assert step["solver_step_id"] == solver_step_id
        assert step["actual_forward"] is True
        assert step["forecast_safe"] is False
        runtime.observe_actual_feature(
            step["stream_key"],
            step["solver_step_id"],
            torch.full((2, 8, 4, 4), float(solver_step_id + 1), dtype=torch.float16),
        )


def test_sdxl_wrapper_forecasts_pre_final_projection_for_splittable_non_codebook_path() -> None:
    """The SDXL wrapper must forecast the pre-final-projection target when the head is splittable."""
    cfg = SpectrumSDXLConfig(
        blend_weight=0.0,
        degree=1,
        ridge_lambda=0.0,
        window_size=2.0,
        flex_window=0.75,
        warmup_steps=0,
        tail_actual_steps=0,
        min_fit_points=3,
    ).validated()
    runtime = SpectrumSDXLRuntime(cfg)

    module_names = (
        "comfy",
        "comfy.ldm",
        "comfy.ldm.modules",
        "comfy.ldm.modules.diffusionmodules",
        "comfy.ldm.modules.diffusionmodules.openaimodel",
    )
    saved_modules = {name: sys.modules.get(name) for name in module_names}

    try:
        comfy = types.ModuleType("comfy")
        ldm = types.ModuleType("comfy.ldm")
        modules = types.ModuleType("comfy.ldm.modules")
        diffusionmodules = types.ModuleType("comfy.ldm.modules.diffusionmodules")
        openaimodel = types.ModuleType("comfy.ldm.modules.diffusionmodules.openaimodel")

        def apply_control(h, control, where):
            del control, where
            return h

        def forward_timestep_embed(module, h, emb, context, transformer_options, output_shape=None, **kwargs):
            del emb, context, transformer_options, output_shape, kwargs
            return module(h) if callable(module) else h

        def timestep_embedding(timesteps, channels, repeat_only=False):
            del repeat_only
            return timesteps.reshape(-1, 1).to(torch.float32).repeat(1, channels)

        openaimodel.apply_control = apply_control
        openaimodel.forward_timestep_embed = forward_timestep_embed
        openaimodel.timestep_embedding = timestep_embedding
        diffusionmodules.openaimodel = openaimodel
        modules.diffusionmodules = diffusionmodules
        ldm.modules = modules
        comfy.ldm = ldm

        sys.modules["comfy"] = comfy
        sys.modules["comfy.ldm"] = ldm
        sys.modules["comfy.ldm.modules"] = modules
        sys.modules["comfy.ldm.modules.diffusionmodules"] = diffusionmodules
        sys.modules["comfy.ldm.modules.diffusionmodules.openaimodel"] = openaimodel

        class _AddOne(torch.nn.Module):
            def forward(self, x):
                return x + 1.0

        class _TimesTwo(torch.nn.Module):
            def forward(self, x):
                return x * 2.0

        class _FakeInner:
            def __init__(self):
                self.input_blocks = []
                self.output_blocks = []
                self.middle_block = None
                self.time_embed = lambda emb: emb
                self.out = torch.nn.Sequential(_AddOne(), _TimesTwo())
                self.model_channels = 1
                self.num_classes = None
                self.predict_codebook_ids = False
                self._forward = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unwrapped forward used"))

        inner = _FakeInner()
        _wrap_sdxl_unet_forward(inner)

        def _options(step_id: int, time_coord: float, actual_forward: bool) -> dict:
            return {
                _RUNTIME_KEY: runtime,
                "spectrum_run_id": "run-a",
                "spectrum_solver_step_id": step_id,
                "spectrum_time_coord": time_coord,
                "spectrum_total_steps": 4,
                "spectrum_actual_forward": actual_forward,
                "uuids": ["stream-a"],
                "cond_or_uncond": [0],
            }

        first = inner._forward(
            torch.full((1, 1, 1, 1), 1.0),
            timesteps=torch.tensor([0.0]),
            transformer_options=_options(0, 0.0, True),
        )
        second = inner._forward(
            torch.full((1, 1, 1, 1), 2.0),
            timesteps=torch.tensor([1.0]),
            transformer_options=_options(1, 1.0, True),
        )
        third = inner._forward(
            torch.full((1, 1, 1, 1), 3.0),
            timesteps=torch.tensor([2.0]),
            transformer_options=_options(2, 2.0, True),
        )
        forecast = inner._forward(
            torch.full((1, 1, 1, 1), 999.0),
            timesteps=torch.tensor([3.0]),
            transformer_options=_options(3, 3.0, False),
        )

        assert torch.allclose(first, torch.full((1, 1, 1, 1), 4.0))
        assert torch.allclose(second, torch.full((1, 1, 1, 1), 6.0))
        assert torch.allclose(third, torch.full((1, 1, 1, 1), 8.0))
        assert torch.allclose(forecast, torch.full((1, 1, 1, 1), 10.0), atol=1e-5)

        state = next(iter(runtime.stream_states.values()))
        assert len(state.forecaster.history) == 3
        assert torch.allclose(state.forecaster.history[0][1], torch.full((1, 1, 1, 1), 2.0))
        assert torch.allclose(state.forecaster.history[1][1], torch.full((1, 1, 1, 1), 3.0))
        assert torch.allclose(state.forecaster.history[2][1], torch.full((1, 1, 1, 1), 4.0))
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_sdxl_wrapper_falls_back_to_final_output_when_non_codebook_head_is_not_splittable() -> None:
    """Unsplittable non-codebook heads must preserve the existing final-output target behavior."""
    cfg = SpectrumSDXLConfig(
        blend_weight=0.0,
        degree=1,
        ridge_lambda=0.0,
        window_size=2.0,
        flex_window=0.75,
        warmup_steps=0,
        tail_actual_steps=0,
        min_fit_points=3,
    ).validated()
    runtime = SpectrumSDXLRuntime(cfg)

    module_names = (
        "comfy",
        "comfy.ldm",
        "comfy.ldm.modules",
        "comfy.ldm.modules.diffusionmodules",
        "comfy.ldm.modules.diffusionmodules.openaimodel",
    )
    saved_modules = {name: sys.modules.get(name) for name in module_names}

    try:
        comfy = types.ModuleType("comfy")
        ldm = types.ModuleType("comfy.ldm")
        modules = types.ModuleType("comfy.ldm.modules")
        diffusionmodules = types.ModuleType("comfy.ldm.modules.diffusionmodules")
        openaimodel = types.ModuleType("comfy.ldm.modules.diffusionmodules.openaimodel")

        def apply_control(h, control, where):
            del control, where
            return h

        def forward_timestep_embed(module, h, emb, context, transformer_options, output_shape=None, **kwargs):
            del emb, context, transformer_options, output_shape, kwargs
            return module(h) if callable(module) else h

        def timestep_embedding(timesteps, channels, repeat_only=False):
            del repeat_only
            return timesteps.reshape(-1, 1).to(torch.float32).repeat(1, channels)

        openaimodel.apply_control = apply_control
        openaimodel.forward_timestep_embed = forward_timestep_embed
        openaimodel.timestep_embedding = timestep_embedding
        diffusionmodules.openaimodel = openaimodel
        modules.diffusionmodules = diffusionmodules
        ldm.modules = modules
        comfy.ldm = ldm

        sys.modules["comfy"] = comfy
        sys.modules["comfy.ldm"] = ldm
        sys.modules["comfy.ldm.modules"] = modules
        sys.modules["comfy.ldm.modules.diffusionmodules"] = diffusionmodules
        sys.modules["comfy.ldm.modules.diffusionmodules.openaimodel"] = openaimodel

        class _FakeInner:
            def __init__(self):
                self.input_blocks = []
                self.output_blocks = []
                self.middle_block = None
                self.time_embed = lambda emb: emb
                self.out = lambda h: h * h
                self.model_channels = 1
                self.num_classes = None
                self.predict_codebook_ids = False
                self._forward = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unwrapped forward used"))

        inner = _FakeInner()
        _wrap_sdxl_unet_forward(inner)

        def _options(step_id: int, time_coord: float, actual_forward: bool) -> dict:
            return {
                _RUNTIME_KEY: runtime,
                "spectrum_run_id": "run-a",
                "spectrum_solver_step_id": step_id,
                "spectrum_time_coord": time_coord,
                "spectrum_total_steps": 4,
                "spectrum_actual_forward": actual_forward,
                "uuids": ["stream-a"],
                "cond_or_uncond": [0],
            }

        first = inner._forward(
            torch.full((1, 1, 1, 1), 1.0),
            timesteps=torch.tensor([0.0]),
            transformer_options=_options(0, 0.0, True),
        )
        second = inner._forward(
            torch.full((1, 1, 1, 1), 2.0),
            timesteps=torch.tensor([1.0]),
            transformer_options=_options(1, 1.0, True),
        )
        third = inner._forward(
            torch.full((1, 1, 1, 1), 3.0),
            timesteps=torch.tensor([2.0]),
            transformer_options=_options(2, 2.0, True),
        )
        forecast = inner._forward(
            torch.full((1, 1, 1, 1), 999.0),
            timesteps=torch.tensor([3.0]),
            transformer_options=_options(3, 3.0, False),
        )

        assert torch.allclose(first, torch.full((1, 1, 1, 1), 1.0))
        assert torch.allclose(second, torch.full((1, 1, 1, 1), 4.0))
        assert torch.allclose(third, torch.full((1, 1, 1, 1), 9.0))
        assert torch.allclose(forecast, torch.full((1, 1, 1, 1), 14.0), atol=1e-5)

        state = next(iter(runtime.stream_states.values()))
        assert len(state.forecaster.history) == 3
        assert torch.allclose(state.forecaster.history[0][1], torch.full((1, 1, 1, 1), 1.0))
        assert torch.allclose(state.forecaster.history[1][1], torch.full((1, 1, 1, 1), 4.0))
        assert torch.allclose(state.forecaster.history[2][1], torch.full((1, 1, 1, 1), 9.0))
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def main() -> None:
    """Run the lightweight regression suite without external test tooling."""
    test_missing_solver_step_context_fails_open()
    test_invalid_solver_step_context_fails_open()
    test_missing_stream_identity_fails_open()
    test_explicit_solver_step_context_allows_forecast()
    test_runtime_prefers_sigma_coord_for_forecasting_even_when_model_time_is_present()
    test_runtime_prefers_sigma_coord_for_forecasting_without_sample_sigmas()
    test_explicit_solver_step_context_without_decision_still_schedules()
    test_outer_step_controller_injects_context_and_resets_runs()
    test_outer_step_controller_same_object_restart_resets_run_state()
    test_outer_step_controller_uses_raw_sigma_time_coord()
    test_model_function_wrapper_injects_context_for_bypassed_guider_path()
    test_model_function_wrapper_reuses_solver_step_for_repeated_same_sigma_subcalls()
    test_model_function_wrapper_same_object_restart_resets_run_state()
    test_model_function_wrapper_preserves_existing_context()
    test_model_function_wrapper_injects_context_for_delegate_internal_apply_calls()
    test_model_function_wrapper_delegate_forces_actual_forward_flag()
    test_runtime_disables_forecasting_when_time_coord_does_not_match_schedule()
    test_forecast_request_before_history_fails_open_per_step()
    test_duplicate_actual_updates_are_deduped()
    test_forecast_fallback_commits_actual_bookkeeping()
    test_run_id_switch_resets_stream_state()
    test_forecaster_chebyshev_fit_falls_back_to_observed_sigma_range()
    test_runtime_fallback_coord_bounds_ignore_unobserved_candidate_step()
    test_runtime_uses_schedule_wide_sigma_bounds_for_forecast_fit()
    test_runtime_holds_forecasting_until_conservative_min_fit_points()
    test_runtime_blocks_forecast_when_recent_validation_error_is_bad()
    test_chebyshev_prediction_varies_with_time_coord()
    test_tail_actual_steps_force_real_tail_even_with_ready_history()
    test_tail_actual_steps_zero_preserves_existing_scheduler_behavior()
    test_tail_actual_steps_greater_than_total_steps_forces_all_steps_real()
    test_sdxl_wrapper_forecasts_pre_final_projection_for_splittable_non_codebook_path()
    test_sdxl_wrapper_falls_back_to_final_output_when_non_codebook_head_is_not_splittable()
    print("ok")


if __name__ == "__main__":
    main()
