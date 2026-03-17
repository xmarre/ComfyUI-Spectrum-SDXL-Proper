"""Lightweight runtime regression tests for the SDXL Spectrum scheduler."""

from __future__ import annotations

import torch

from comfyui_spectrum_sdxl.config import SpectrumSDXLConfig
from comfyui_spectrum_sdxl.forecast import ChebyshevFeatureForecaster
from comfyui_spectrum_sdxl.runtime import SpectrumSDXLRuntime
from comfyui_spectrum_sdxl.sdxl import _SpectrumOuterStepController


def _make_cfg() -> SpectrumSDXLConfig:
    """Create the default test configuration."""
    return SpectrumSDXLConfig(
        blend_weight=0.5,
        degree=4,
        ridge_lambda=0.1,
        window_size=2.0,
        flex_window=0.75,
        warmup_steps=2,
    ).validated()


def _step_options(
    run_id,
    solver_step_id: int,
    time_coord: float,
    actual_forward: bool | None = None,
    total_steps: int = 6,
    uuid: str = "stream-a",
    cond: int = 0,
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
    runtime = SpectrumSDXLRuntime(_make_cfg())

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

    forecast = runtime.begin_step(_step_options("run-a", 2, 2.0, False), torch.tensor([2.0]), (2, 8, 4, 4))
    assert forecast["actual_forward"] is False
    assert forecast["forecast_safe"] is True

    pred = runtime.predict_feature(forecast["stream_key"], forecast["solver_step_id"])
    assert pred.shape == (2, 8, 4, 4)
    assert torch.isfinite(pred).all()

    runtime.finalize_step(forecast["stream_key"], forecast["solver_step_id"], used_forecast=True)
    assert runtime.last_info["forecasted_passes"] == 1


def test_explicit_solver_step_context_without_decision_still_schedules() -> None:
    """Outer-step context alone should be enough once the controller owns step ids."""
    runtime = SpectrumSDXLRuntime(_make_cfg())

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
    assert third["actual_forward"] is False
    assert third["forecast_safe"] is True


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
    assert restarted["spectrum_time_coord"] == 0.0


def test_outer_step_controller_uses_ordinal_time_coord() -> None:
    """The controller must keep time_coord in the same ordinal step space as the forecaster."""
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
    assert first["spectrum_time_coord"] == 0.0
    assert second["spectrum_time_coord"] == 1.0


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
    runtime = SpectrumSDXLRuntime(_make_cfg())

    for step_id, value in ((0, 1.0), (1, 2.0)):
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

    forecast = runtime.begin_step(_step_options("run-a", 2, 2.0, False), torch.tensor([2.0]), (2, 8, 4, 4))
    assert forecast["actual_forward"] is False

    runtime.finalize_step(forecast["stream_key"], forecast["solver_step_id"], used_forecast=False)

    state = runtime.stream_states[forecast["stream_key"]]
    assert runtime.last_info["forecasted_passes"] == 0
    assert runtime.last_info["actual_forward_count"] == 3
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


def test_forecaster_uses_schedule_coordinates_not_ordinal_step_index() -> None:
    """Direct forecaster predictions should accept the runtime total-steps contract."""
    forecaster = ChebyshevFeatureForecaster(
        degree=2,
        ridge_lambda=0.1,
        blend_weight=0.0,
        history_size=10,
    )
    forecaster.update(10.0, torch.tensor([10.0]))
    forecaster.update(9.0, torch.tensor([9.0]))

    pred = forecaster.predict(1.0, 11)

    assert torch.allclose(pred, torch.tensor([1.0]), atol=1e-5)


def main() -> None:
    """Run the lightweight regression suite without external test tooling."""
    test_missing_solver_step_context_fails_open()
    test_invalid_solver_step_context_fails_open()
    test_missing_stream_identity_fails_open()
    test_explicit_solver_step_context_allows_forecast()
    test_explicit_solver_step_context_without_decision_still_schedules()
    test_outer_step_controller_injects_context_and_resets_runs()
    test_outer_step_controller_same_object_restart_resets_run_state()
    test_outer_step_controller_uses_ordinal_time_coord()
    test_forecast_request_before_history_fails_open_per_step()
    test_duplicate_actual_updates_are_deduped()
    test_forecast_fallback_commits_actual_bookkeeping()
    test_run_id_switch_resets_stream_state()
    test_forecaster_uses_schedule_coordinates_not_ordinal_step_index()
    print("ok")


if __name__ == "__main__":
    main()
