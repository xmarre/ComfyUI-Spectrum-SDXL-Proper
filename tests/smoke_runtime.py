"""Lightweight runtime regression tests for the SDXL Spectrum scheduler."""

from __future__ import annotations

import torch

from comfyui_spectrum_sdxl.config import SpectrumSDXLConfig
from comfyui_spectrum_sdxl.runtime import SpectrumSDXLRuntime


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


def test_single_stream_forecasts() -> None:
    """A single stream should warm up, then eventually forecast."""
    runtime = SpectrumSDXLRuntime(_make_cfg())

    sample_sigmas = torch.linspace(1.0, 0.0, 6)
    seen_actual = 0
    seen_forecast = 0
    for i in range(5):
        sigma = float(sample_sigmas[i].item())
        transformer_options = {
            "sample_sigmas": sample_sigmas,
            "sigmas": torch.tensor([sigma]),
            "uuids": ["stream-a"],
            "cond_or_uncond": [0],
        }
        timesteps = torch.tensor([float(i)])
        decision = runtime.begin_step(transformer_options, timesteps, (2, 8, 4, 4))
        assert decision["global_step_idx"] == i
        assert decision["local_step_count"] == i
        if decision["actual_forward"]:
            seen_actual += 1
            feature = torch.full((2, 8, 4, 4), float(i), dtype=torch.float16)
            runtime.observe_actual_feature(decision["stream_key"], decision["global_step_idx"], feature)
        else:
            seen_forecast += 1
            runtime.finalize_step(decision["stream_key"], decision["global_step_idx"], used_forecast=True)
            pred = runtime.predict_feature(decision["stream_key"], decision["global_step_idx"])
            assert pred.shape == (2, 8, 4, 4)
            assert torch.isfinite(pred).all()

    assert seen_actual >= 2
    assert seen_forecast >= 1


def test_multi_stream_warmup_disables_forecasting() -> None:
    """Multiple warmup streams at one solver step must fail open."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    sample_sigmas = torch.linspace(1.0, 0.0, 6)

    for i in range(2):
        sigma = float(sample_sigmas[i].item())
        for uuid, cond in (("stream-a", 0), ("stream-b", 1)):
            transformer_options = {
                "sample_sigmas": sample_sigmas,
                "sigmas": torch.tensor([sigma]),
                "uuids": [uuid],
                "cond_or_uncond": [cond],
            }
            decision = runtime.begin_step(transformer_options, torch.tensor([float(i)]), (2, 8, 4, 4))
            assert decision["actual_forward"] is True
            assert decision["global_step_idx"] == i
            runtime.observe_actual_feature(
                decision["stream_key"],
                decision["global_step_idx"],
                torch.full((2, 8, 4, 4), float(i + cond), dtype=torch.float16),
            )

    decision = runtime.begin_step(
        {
            "sample_sigmas": sample_sigmas,
            "sigmas": torch.tensor([float(sample_sigmas[2].item())]),
            "uuids": ["stream-a"],
            "cond_or_uncond": [0],
        },
        torch.tensor([2.0]),
        (2, 8, 4, 4),
    )
    assert runtime.last_info["forecast_disabled"] is True
    assert runtime.last_info["forecast_disable_reason"] == "multi_stream_warmup"
    assert decision["forecast_safe"] is False
    assert decision["actual_forward"] is True


def test_cfg_streams_keep_isolated_actual_state() -> None:
    """Two same-step streams must not share runtime state."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    sample_sigmas = torch.linspace(1.0, 0.0, 6)

    decisions = {}
    for uuid, cond, value in (("stream-a", 0, 10.0), ("stream-b", 1, 20.0)):
        decision = runtime.begin_step(
            {
                "sample_sigmas": sample_sigmas,
                "sigmas": torch.tensor([float(sample_sigmas[0].item())]),
                "uuids": [uuid],
                "cond_or_uncond": [cond],
            },
            torch.tensor([0.0]),
            (2, 8, 4, 4),
        )
        assert decision["global_step_idx"] == 0
        assert decision["actual_forward"] is True
        runtime.observe_actual_feature(
            decision["stream_key"],
            decision["global_step_idx"],
            torch.full((2, 8, 4, 4), value, dtype=torch.float16),
        )
        decisions[uuid] = decision

    state_a = runtime.stream_states[decisions["stream-a"]["stream_key"]]
    state_b = runtime.stream_states[decisions["stream-b"]["stream_key"]]
    assert state_a is not state_b
    assert state_a.observed_global_steps == {0}
    assert state_b.observed_global_steps == {0}
    assert len(state_a.forecaster.history) == 1
    assert len(state_b.forecaster.history) == 1

    _, feature_a = state_a.forecaster.history[0]
    _, feature_b = state_b.forecaster.history[0]
    assert torch.equal(feature_a, torch.full((2, 8, 4, 4), 10.0, dtype=torch.float16))
    assert torch.equal(feature_b, torch.full((2, 8, 4, 4), 20.0, dtype=torch.float16))


def test_fail_open_reconciles_unfinalized_cached_decision() -> None:
    """Disabling forecasting must override stale unfinalized cached decisions."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    sample_sigmas = torch.linspace(1.0, 0.0, 6)

    for i, value in enumerate((1.0, 2.0)):
        decision = runtime.begin_step(
            {
                "sample_sigmas": sample_sigmas,
                "sigmas": torch.tensor([float(sample_sigmas[i].item())]),
                "uuids": ["stream-a"],
                "cond_or_uncond": [0],
            },
            torch.tensor([float(i)]),
            (2, 8, 4, 4),
        )
        runtime.observe_actual_feature(
            decision["stream_key"],
            decision["global_step_idx"],
            torch.full((2, 8, 4, 4), value, dtype=torch.float16),
        )

    forecast_candidate = runtime.begin_step(
        {
            "sample_sigmas": sample_sigmas,
            "sigmas": torch.tensor([float(sample_sigmas[2].item())]),
            "uuids": ["stream-a"],
            "cond_or_uncond": [0],
        },
        torch.tensor([2.0]),
        (2, 8, 4, 4),
    )
    assert forecast_candidate["actual_forward"] is False
    assert forecast_candidate["forecast_safe"] is True
    assert forecast_candidate["finalized"] is False

    warmup_other_stream = runtime.begin_step(
        {
            "sample_sigmas": sample_sigmas,
            "sigmas": torch.tensor([float(sample_sigmas[0].item())]),
            "uuids": ["stream-b"],
            "cond_or_uncond": [1],
        },
        torch.tensor([0.0]),
        (2, 8, 4, 4),
    )
    assert runtime.last_info["forecast_disabled"] is True
    assert runtime.last_info["forecast_disable_reason"] == "multi_stream_warmup"
    assert warmup_other_stream["actual_forward"] is True

    retried = runtime.begin_step(
        {
            "sample_sigmas": sample_sigmas,
            "sigmas": torch.tensor([float(sample_sigmas[2].item())]),
            "uuids": ["stream-a"],
            "cond_or_uncond": [0],
        },
        torch.tensor([2.0]),
        (2, 8, 4, 4),
    )
    assert retried is forecast_candidate
    assert retried["forecast_safe"] is False
    assert retried["actual_forward"] is True
    assert retried["finalized"] is False


def test_same_signature_restart_clears_stale_multistream_guard() -> None:
    """A same-signature step-0 restart must clear stale run-scoped warmup layout state."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    sample_sigmas_a = torch.linspace(1.0, 0.0, 6)
    sample_sigmas_b = torch.linspace(1.0, 0.0, 6)
    sample_sigmas_c = torch.linspace(1.0, 0.0, 6)

    first_a = runtime.begin_step(
        {
            "sample_sigmas": sample_sigmas_a,
            "sigmas": torch.tensor([float(sample_sigmas_a[0].item())]),
            "uuids": ["stream-a"],
            "cond_or_uncond": [0],
        },
        torch.tensor([0.0]),
        (2, 8, 4, 4),
    )
    runtime.observe_actual_feature(
        first_a["stream_key"],
        first_a["global_step_idx"],
        torch.full((2, 8, 4, 4), 1.0, dtype=torch.float16),
    )

    first_b = runtime.begin_step(
        {
            "sample_sigmas": sample_sigmas_b,
            "sigmas": torch.tensor([float(sample_sigmas_b[0].item())]),
            "uuids": ["stream-b"],
            "cond_or_uncond": [1],
        },
        torch.tensor([0.0]),
        (2, 8, 4, 4),
    )
    runtime.observe_actual_feature(
        first_b["stream_key"],
        first_b["global_step_idx"],
        torch.full((2, 8, 4, 4), 2.0, dtype=torch.float16),
    )

    restarted_a = runtime.begin_step(
        {
            "sample_sigmas": sample_sigmas_c,
            "sigmas": torch.tensor([float(sample_sigmas_c[0].item())]),
            "uuids": ["stream-a"],
            "cond_or_uncond": [0],
        },
        torch.tensor([0.0]),
        (2, 8, 4, 4),
    )
    assert runtime.last_info["forecast_disabled"] is False
    assert runtime.last_info["forecast_disable_reason"] is None
    runtime.observe_actual_feature(
        restarted_a["stream_key"],
        restarted_a["global_step_idx"],
        torch.full((2, 8, 4, 4), 3.0, dtype=torch.float16),
    )

    step1 = runtime.begin_step(
        {
            "sample_sigmas": sample_sigmas_c,
            "sigmas": torch.tensor([float(sample_sigmas_c[1].item())]),
            "uuids": ["stream-a"],
            "cond_or_uncond": [0],
        },
        torch.tensor([1.0]),
        (2, 8, 4, 4),
    )
    runtime.observe_actual_feature(
        step1["stream_key"],
        step1["global_step_idx"],
        torch.full((2, 8, 4, 4), 4.0, dtype=torch.float16),
    )

    step2 = runtime.begin_step(
        {
            "sample_sigmas": sample_sigmas_c,
            "sigmas": torch.tensor([float(sample_sigmas_c[2].item())]),
            "uuids": ["stream-a"],
            "cond_or_uncond": [0],
        },
        torch.tensor([2.0]),
        (2, 8, 4, 4),
    )
    assert runtime.last_info["forecast_disabled"] is False
    assert runtime.last_info["forecast_disable_reason"] is None
    assert step2["forecast_safe"] is True
    assert step2["actual_forward"] is False


def test_duplicate_actual_updates_are_deduped() -> None:
    """A repeated actual write for one stream/step must only be recorded once."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    sample_sigmas = torch.linspace(1.0, 0.0, 4)
    decision = runtime.begin_step(
        {
            "sample_sigmas": sample_sigmas,
            "sigmas": torch.tensor([float(sample_sigmas[0].item())]),
            "uuids": ["stream-a"],
            "cond_or_uncond": [0],
        },
        torch.tensor([0.0]),
        (2, 8, 4, 4),
    )
    assert decision["actual_forward"] is True

    first_feature = torch.full((2, 8, 4, 4), 1.0, dtype=torch.float16)
    second_feature = torch.full((2, 8, 4, 4), 2.0, dtype=torch.float16)
    runtime.observe_actual_feature(decision["stream_key"], decision["global_step_idx"], first_feature)
    runtime.observe_actual_feature(decision["stream_key"], decision["global_step_idx"], second_feature)

    state = runtime.stream_states[decision["stream_key"]]
    assert len(state.forecaster.history) == 1
    _, stored_feature = state.forecaster.history[0]
    assert torch.equal(stored_feature, first_feature)


def test_forecast_fallback_commits_actual_bookkeeping() -> None:
    """A rejected forecast should not count as a cached pass."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    sample_sigmas = torch.linspace(1.0, 0.0, 6)

    for i, value in enumerate((1.0, 2.0)):
        decision = runtime.begin_step(
            {
                "sample_sigmas": sample_sigmas,
                "sigmas": torch.tensor([float(sample_sigmas[i].item())]),
                "uuids": ["stream-a"],
                "cond_or_uncond": [0],
            },
            torch.tensor([float(i)]),
            (2, 8, 4, 4),
        )
        runtime.observe_actual_feature(
            decision["stream_key"],
            decision["global_step_idx"],
            torch.full((2, 8, 4, 4), value, dtype=torch.float16),
        )

    decision = runtime.begin_step(
        {
            "sample_sigmas": sample_sigmas,
            "sigmas": torch.tensor([float(sample_sigmas[2].item())]),
            "uuids": ["stream-a"],
            "cond_or_uncond": [0],
        },
        torch.tensor([2.0]),
        (2, 8, 4, 4),
    )
    assert decision["actual_forward"] is False

    runtime.finalize_step(decision["stream_key"], decision["global_step_idx"], used_forecast=False)

    state = runtime.stream_states[decision["stream_key"]]
    assert runtime.last_info["forecasted_passes"] == 0
    assert runtime.last_info["actual_forward_count"] == 3
    assert state.num_consecutive_cached_steps == 0
    assert state.decisions_by_global_step[decision["global_step_idx"]]["actual_forward"] is True
    assert state.decisions_by_global_step[decision["global_step_idx"]]["finalized"] is True


def test_observe_retry_after_update_failure() -> None:
    """A failed update should not poison retries for that step."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    sample_sigmas = torch.linspace(1.0, 0.0, 4)
    decision = runtime.begin_step(
        {
            "sample_sigmas": sample_sigmas,
            "sigmas": torch.tensor([float(sample_sigmas[0].item())]),
            "uuids": ["stream-a"],
            "cond_or_uncond": [0],
        },
        torch.tensor([0.0]),
        (2, 8, 4, 4),
    )
    state = runtime.stream_states[decision["stream_key"]]
    original_update = state.forecaster.update

    def failing_update(step_idx, feature):
        raise RuntimeError("boom")

    state.forecaster.update = failing_update
    try:
        try:
            runtime.observe_actual_feature(
                decision["stream_key"],
                decision["global_step_idx"],
                torch.full((2, 8, 4, 4), 1.0, dtype=torch.float16),
            )
            raise AssertionError("expected update failure")
        except RuntimeError as exc:
            assert str(exc) == "boom"
        assert state.observed_global_steps == set()
        assert state.forecaster.history == []
        assert runtime.last_info["actual_forward_count"] == 0
        assert state.decisions_by_global_step[decision["global_step_idx"]]["finalized"] is False
    finally:
        state.forecaster.update = original_update

    runtime.observe_actual_feature(
        decision["stream_key"],
        decision["global_step_idx"],
        torch.full((2, 8, 4, 4), 2.0, dtype=torch.float16),
    )
    assert state.observed_global_steps == {0}
    assert len(state.forecaster.history) == 1
    assert runtime.last_info["actual_forward_count"] == 1
    assert state.decisions_by_global_step[decision["global_step_idx"]]["finalized"] is True


def test_same_schedule_restart_resets_stream_state() -> None:
    """A lower-step revisit on the same schedule should reset that stream."""
    runtime = SpectrumSDXLRuntime(
        SpectrumSDXLConfig(
            blend_weight=0.5,
            degree=4,
            ridge_lambda=0.1,
            window_size=2.0,
            flex_window=0.75,
            warmup_steps=10,
        ).validated()
    )
    sample_sigmas = torch.linspace(1.0, 0.0, 4)
    transformer_options = {
        "sample_sigmas": sample_sigmas,
        "uuids": ["stream-a"],
        "cond_or_uncond": [0],
    }

    first_run_features = []
    for i in range(3):
        sigma = float(sample_sigmas[i].item())
        decision = runtime.begin_step(
            {**transformer_options, "sigmas": torch.tensor([sigma])},
            torch.tensor([float(i)]),
            (2, 8, 4, 4),
        )
        assert decision["global_step_idx"] == i
        assert decision["local_step_count"] == i
        assert decision["actual_forward"] is True
        feature = torch.full((2, 8, 4, 4), float(10 + i), dtype=torch.float16)
        runtime.observe_actual_feature(decision["stream_key"], decision["global_step_idx"], feature)
        first_run_features.append(feature)

    initial_state = runtime.stream_states[decision["stream_key"]]
    assert len(initial_state.forecaster.history) == 3

    restarted = runtime.begin_step(
        {**transformer_options, "sigmas": torch.tensor([float(sample_sigmas[0].item())])},
        torch.tensor([0.0]),
        (2, 8, 4, 4),
    )
    assert restarted["global_step_idx"] == 0
    assert restarted["local_step_count"] == 0
    restarted_state = runtime.stream_states[restarted["stream_key"]]
    assert restarted_state is not initial_state
    assert restarted_state.decisions_by_global_step.keys() == {0}
    assert restarted_state.observed_global_steps == set()
    assert restarted_state.forecaster.history == []

    new_feature = torch.full((2, 8, 4, 4), 99.0, dtype=torch.float16)
    runtime.observe_actual_feature(restarted["stream_key"], restarted["global_step_idx"], new_feature)
    assert restarted_state.observed_global_steps == {0}
    assert len(restarted_state.forecaster.history) == 1
    _, stored_feature = restarted_state.forecaster.history[0]
    assert torch.equal(stored_feature, new_feature)
    assert not torch.equal(stored_feature, first_run_features[0])


def test_same_schedule_new_token_resets_after_step_zero_abort() -> None:
    """Step-0 token churn after an abort should reset only the stale stream."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    sample_sigmas_a = torch.linspace(1.0, 0.0, 4)
    sample_sigmas_b = torch.linspace(1.0, 0.0, 4)

    first = runtime.begin_step(
        {
            "sample_sigmas": sample_sigmas_a,
            "sigmas": torch.tensor([float(sample_sigmas_a[0].item())]),
            "uuids": ["stream-a"],
            "cond_or_uncond": [0],
        },
        torch.tensor([0.0]),
        (2, 8, 4, 4),
    )
    assert first["actual_forward"] is True
    runtime.observe_actual_feature(
        first["stream_key"],
        first["global_step_idx"],
        torch.full((2, 8, 4, 4), 7.0, dtype=torch.float16),
    )

    first_state = runtime.stream_states[first["stream_key"]]
    assert first_state.observed_global_steps == {0}
    assert len(first_state.forecaster.history) == 1

    restarted = runtime.begin_step(
        {
            "sample_sigmas": sample_sigmas_b,
            "sigmas": torch.tensor([float(sample_sigmas_b[0].item())]),
            "uuids": ["stream-a"],
            "cond_or_uncond": [0],
        },
        torch.tensor([0.0]),
        (2, 8, 4, 4),
    )
    assert restarted["run_id"] == 0
    assert runtime.run_id == 0
    assert restarted["global_step_idx"] == 0
    assert restarted["local_step_count"] == 0

    restarted_state = runtime.stream_states[restarted["stream_key"]]
    assert restarted_state is not first_state
    assert restarted_state.decisions_by_global_step.keys() == {0}
    assert restarted_state.observed_global_steps == set()
    assert restarted_state.forecaster.history == []

    runtime.observe_actual_feature(
        restarted["stream_key"],
        restarted["global_step_idx"],
        torch.full((2, 8, 4, 4), 8.0, dtype=torch.float16),
    )
    assert restarted_state.observed_global_steps == {0}
    assert len(restarted_state.forecaster.history) == 1


def test_step_zero_token_churn_across_streams_does_not_reset() -> None:
    """Step-0 token churn across different streams must not reset the run."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    sample_sigmas_a = torch.linspace(1.0, 0.0, 4)
    sample_sigmas_b = torch.linspace(1.0, 0.0, 4)

    first = runtime.begin_step(
        {
            "sample_sigmas": sample_sigmas_a,
            "sigmas": torch.tensor([float(sample_sigmas_a[0].item())]),
            "uuids": ["stream-a"],
            "cond_or_uncond": [0],
        },
        torch.tensor([0.0]),
        (2, 8, 4, 4),
    )
    assert first["run_id"] == 0
    runtime.observe_actual_feature(
        first["stream_key"],
        first["global_step_idx"],
        torch.full((2, 8, 4, 4), 3.0, dtype=torch.float16),
    )
    first_state = runtime.stream_states[first["stream_key"]]

    second = runtime.begin_step(
        {
            "sample_sigmas": sample_sigmas_b,
            "sigmas": torch.tensor([float(sample_sigmas_b[0].item())]),
            "uuids": ["stream-b"],
            "cond_or_uncond": [1],
        },
        torch.tensor([0.0]),
        (2, 8, 4, 4),
    )
    assert second["run_id"] == 0
    assert runtime.run_id == 0
    assert second["local_step_count"] == 0

    second_state = runtime.stream_states[second["stream_key"]]
    assert second_state is not first_state
    assert first["stream_key"] in runtime.stream_states
    assert len(first_state.forecaster.history) == 1
    assert second_state.forecaster.history == []


def test_step_zero_same_stream_token_churn_resets_only_that_stream() -> None:
    """Same-stream step-0 token churn should reset only that stream's state."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    sample_sigmas_a = torch.linspace(1.0, 0.0, 4)
    sample_sigmas_b = torch.linspace(1.0, 0.0, 4)
    sample_sigmas_c = torch.linspace(1.0, 0.0, 4)

    first_a = runtime.begin_step(
        {
            "sample_sigmas": sample_sigmas_a,
            "sigmas": torch.tensor([float(sample_sigmas_a[0].item())]),
            "uuids": ["stream-a"],
            "cond_or_uncond": [0],
        },
        torch.tensor([0.0]),
        (2, 8, 4, 4),
    )
    runtime.observe_actual_feature(
        first_a["stream_key"],
        first_a["global_step_idx"],
        torch.full((2, 8, 4, 4), 4.0, dtype=torch.float16),
    )
    first_a_state = runtime.stream_states[first_a["stream_key"]]
    assert len(first_a_state.forecaster.history) == 1

    first_b = runtime.begin_step(
        {
            "sample_sigmas": sample_sigmas_b,
            "sigmas": torch.tensor([float(sample_sigmas_b[0].item())]),
            "uuids": ["stream-b"],
            "cond_or_uncond": [1],
        },
        torch.tensor([0.0]),
        (2, 8, 4, 4),
    )
    runtime.observe_actual_feature(
        first_b["stream_key"],
        first_b["global_step_idx"],
        torch.full((2, 8, 4, 4), 9.0, dtype=torch.float16),
    )
    first_b_state = runtime.stream_states[first_b["stream_key"]]
    assert len(first_b_state.forecaster.history) == 1

    second_a = runtime.begin_step(
        {
            "sample_sigmas": sample_sigmas_c,
            "sigmas": torch.tensor([float(sample_sigmas_c[0].item())]),
            "uuids": ["stream-a"],
            "cond_or_uncond": [0],
        },
        torch.tensor([0.0]),
        (2, 8, 4, 4),
    )
    assert second_a["run_id"] == 0
    assert runtime.run_id == 0
    assert second_a["global_step_idx"] == 0
    assert second_a["local_step_count"] == 0
    assert second_a["actual_forward"] is True

    second_a_state = runtime.stream_states[second_a["stream_key"]]
    assert second_a_state is not first_a_state
    assert runtime.stream_states[first_b["stream_key"]] is first_b_state
    assert len(first_b_state.forecaster.history) == 1
    assert second_a_state.forecaster.history == []
    assert second_a_state.observed_global_steps == set()
    assert second_a_state.decisions_by_global_step.keys() == {0}


def test_mid_run_token_change_does_not_reset() -> None:
    """Identical-value token churn after step 0 must not flush active state."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    sample_sigmas_a = torch.linspace(1.0, 0.0, 4)
    sample_sigmas_b = torch.linspace(1.0, 0.0, 4)

    first = runtime.begin_step(
        {
            "sample_sigmas": sample_sigmas_a,
            "sigmas": torch.tensor([float(sample_sigmas_a[0].item())]),
            "uuids": ["stream-a"],
            "cond_or_uncond": [0],
        },
        torch.tensor([0.0]),
        (2, 8, 4, 4),
    )
    assert first["run_id"] == 0
    assert first["local_step_count"] == 0
    runtime.observe_actual_feature(
        first["stream_key"],
        first["global_step_idx"],
        torch.full((2, 8, 4, 4), 5.0, dtype=torch.float16),
    )

    first_state = runtime.stream_states[first["stream_key"]]
    assert len(first_state.forecaster.history) == 1

    second = runtime.begin_step(
        {
            "sample_sigmas": sample_sigmas_b,
            "sigmas": torch.tensor([float(sample_sigmas_b[1].item())]),
            "uuids": ["stream-a"],
            "cond_or_uncond": [0],
        },
        torch.tensor([1.0]),
        (2, 8, 4, 4),
    )
    assert second["run_id"] == 0
    assert second["global_step_idx"] == 1
    assert second["local_step_count"] == 1

    second_state = runtime.stream_states[second["stream_key"]]
    assert second_state is first_state
    assert len(second_state.forecaster.history) == 1

    runtime.observe_actual_feature(
        second["stream_key"],
        second["global_step_idx"],
        torch.full((2, 8, 4, 4), 6.0, dtype=torch.float16),
    )
    assert second_state.observed_global_steps == {0, 1}
    assert len(second_state.forecaster.history) == 2


def test_missing_stream_identity_fails_open() -> None:
    """Missing stream identity should conservatively force an actual forward."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    sample_sigmas = torch.linspace(1.0, 0.0, 4)
    decision = runtime.begin_step(
        {"sample_sigmas": sample_sigmas, "sigmas": torch.tensor([1.0])},
        torch.tensor([0.0]),
        (2, 8, 4, 4),
    )
    assert decision["actual_forward"] is True
    assert decision["forecast_safe"] is False
    assert decision["stream_key"] is None


def test_malformed_stream_identity_fails_open() -> None:
    """Partially invalid stream metadata must not produce a cacheable stream key."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    sample_sigmas = torch.linspace(1.0, 0.0, 4)

    decision = runtime.begin_step(
        {
            "sample_sigmas": sample_sigmas,
            "sigmas": torch.tensor([1.0]),
            "uuids": ["stream-a"],
            "cond_or_uncond": [0, 1],
        },
        torch.tensor([0.0]),
        (2, 8, 4, 4),
    )
    assert decision["actual_forward"] is True
    assert decision["forecast_safe"] is False
    assert decision["stream_key"] is None


def test_non_iterable_stream_identity_fails_open() -> None:
    """Non-iterable stream metadata should fail open instead of raising."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    sample_sigmas = torch.linspace(1.0, 0.0, 4)

    decision = runtime.begin_step(
        {
            "sample_sigmas": sample_sigmas,
            "sigmas": torch.tensor([1.0]),
            "uuids": 123,
            "cond_or_uncond": [0],
        },
        torch.tensor([0.0]),
        (1, 8, 4, 4),
    )
    assert decision["actual_forward"] is True
    assert decision["forecast_safe"] is False
    assert decision["stream_key"] is None


def test_global_step_index_uses_cached_schedule_metadata() -> None:
    """Step lookup should not rebuild the schedule signature after sync."""
    runtime = SpectrumSDXLRuntime(_make_cfg())
    sample_sigmas = torch.linspace(1.0, 0.0, 6)
    runtime._ensure_run_sync({"sample_sigmas": sample_sigmas})
    # This patches a private method on purpose: after _ensure_run_sync()
    # populates cached metadata, global_step_index() should resolve from that
    # cache alone and must not re-read transformer_options.
    runtime._schedule_signature = lambda _transformer_options: (_ for _ in ()).throw(
        AssertionError("global_step_index should use cached schedule metadata")
    )

    assert runtime.global_step_index(round(float(sample_sigmas[2].item()), 8)) == 2


def main() -> None:
    """Run the lightweight regression suite without external test tooling."""
    test_single_stream_forecasts()
    test_multi_stream_warmup_disables_forecasting()
    test_cfg_streams_keep_isolated_actual_state()
    test_fail_open_reconciles_unfinalized_cached_decision()
    test_same_signature_restart_clears_stale_multistream_guard()
    test_duplicate_actual_updates_are_deduped()
    test_forecast_fallback_commits_actual_bookkeeping()
    test_observe_retry_after_update_failure()
    test_same_schedule_restart_resets_stream_state()
    test_same_schedule_new_token_resets_after_step_zero_abort()
    test_step_zero_token_churn_across_streams_does_not_reset()
    test_step_zero_same_stream_token_churn_resets_only_that_stream()
    test_mid_run_token_change_does_not_reset()
    test_missing_stream_identity_fails_open()
    test_malformed_stream_identity_fails_open()
    test_non_iterable_stream_identity_fails_open()
    test_global_step_index_uses_cached_schedule_metadata()
    print("ok")


if __name__ == "__main__":
    main()
