# ComfyUI Spectrum SDXL Proper

Native ComfyUI custom node for the SDXL U-Net path. It patches the native `UNetModel._forward(...)` boundary and targets the final hidden feature before the SDXL output head.

## Current contract

Spectrum's scheduling invariant lives at the outer solver-step level, not inside repeated low-level model calls. This repository now enforces that conservatively:

- the node installs an outer-step controller at ComfyUI's `sampler_calc_cond_batch_function` hook
- the model hook does not reconstruct solver steps from `sigmas`, call order, or repeated timesteps
- forecasting is allowed only when `transformer_options` contains explicit outer-step context
- otherwise the runtime fails open and executes a real denoiser forward

Required outer-step keys:

- `spectrum_run_id` - unique identifier for the current sampling run
- `spectrum_solver_step_id` - ordinal solver step index (0, 1, 2, ...)
- `spectrum_time_coord` - raw sigma value (sigma-space coordinate)

Optional key:

- `spectrum_actual_forward` - explicit per-step decision override
- `spectrum_total_steps` - total number of diffusion steps

The built-in outer-step controller injects `run_id`, `solver_step_id`, `time_coord`, and `total_steps` automatically for normal ComfyUI sampling. The `time_coord` represents the raw sigma value from the noise schedule, not an ordinal step index. External controllers should supply raw sigma values in `spectrum_time_coord` to maintain compatibility with the sigma-space forecasting coordinate system. `spectrum_actual_forward` can still be supplied by an external controller if it wants to force the per-step decision explicitly.

## What the model hook does

The SDXL wrapper is now only a thin execution layer:

- on actual steps, it runs the real U-Net path and records the final hidden feature
- on forecast steps, it reads the outer-step context and asks the runtime for a prediction
- if context is missing, malformed, or stream identity is missing, it falls back to an actual forward

The hook does not own solver-step identities. When `spectrum_actual_forward` is absent, the runtime schedules from the explicit outer step id instead of reconstructing steps from repeated low-level calls.

This still assumes the hooked `sampler_calc_cond_batch_function` path is called once per forecastable denoiser evaluation in the samplers you care about. That boundary is better than the U-Net hook, but it is not a universal proof for every sampler layout.

## Node

`Spectrum Apply SDXL` returns a patched `MODEL`.

Inputs:

- `model`
- `enabled`
- `blend_weight`
- `degree`
- `ridge_lambda`
- `window_size`
- `flex_window`
- `warmup_steps`
- `tail_actual_steps`
- `min_fit_points`
- `debug`

`window_size`, `flex_window`, and `warmup_steps` are now consumed by the built-in outer-step controller path. `min_fit_points` sets the minimum number of actual observations required before the forecaster is even eligible; the runtime still requires at least `max(3, degree + 1, min_fit_points)` actual history points and also requires the recent one-step validation error to stay below a conservative internal threshold before a forecast can be used. `tail_actual_steps` reserves the last `N` solver steps for real forwards only, even if forecasting history is ready. Set it to `0` to preserve the older behavior with no protected real tail. External controllers can also provide explicit per-step decisions through `spectrum_actual_forward`, but the runtime still forces the protected tail onto the real path.

## Usage

Typical placement:

```text
CheckpointLoaderSimple
  -> LoRA / model patches
  -> Spectrum Apply SDXL
  -> sampler / guider
```

For normal ComfyUI sampling, the node installs its own outer-step controller and stamps explicit Spectrum step metadata before the U-Net hook runs.

## Implementation notes

- Forecast target: the final hidden U-Net feature before `self.out(h)`
- Forecast history: populated only from actual forwards
- Forecast coordinate: `spectrum_time_coord` in sigma-space (raw sigma values)
- Step identity: `spectrum_solver_step_id` (ordinal step index)
- Run boundary: `spectrum_run_id`
- Stream isolation: `uuids` + `cond_or_uncond` + input shape

The forecaster normalizes sigma values to the Chebyshev domain using the actual min/max range of observed sigma coordinates. This allows the predictor to correctly handle arbitrary continuous sigma schedules without assuming ordinal step spacing. The runtime uses `sample_sigmas` length as a fallback for `spectrum_total_steps` when not explicitly provided.

## Smoke test

Run:

```bash
python tests/smoke_runtime.py
```

The smoke suite covers:

- fail-open behavior without explicit solver-step context
- explicit-context forecasting
- outer-step controller context injection
- run-id-based state reset
- protected real-tail enforcement, including `tail_actual_steps >= total_steps`

Expected output:

```text
ok
```

## References

- Paper: [Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](https://arxiv.org/abs/2603.01623)
- Spectrum project page: [Spectrum](https://hanjq17.github.io/Spectrum/)
- Official implementation: [hanjq17/Spectrum](https://github.com/hanjq17/Spectrum)
