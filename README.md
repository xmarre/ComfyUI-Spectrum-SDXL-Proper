# ComfyUI Spectrum SDXL Proper

Native ComfyUI custom node for the SDXL U-Net path. It patches the native `UNetModel._forward(...)` boundary and targets the final hidden feature before the SDXL output head.

## Current contract

Spectrum's scheduling invariant lives at the outer solver-step level, not inside repeated low-level model calls. This repository now enforces that conservatively:

- the node installs an outer-step controller at ComfyUI's `sampler_calc_cond_batch_function` hook
- the model hook does not reconstruct solver steps from `sigmas`, call order, or repeated timesteps
- forecasting is allowed only when `transformer_options` contains explicit outer-step context
- otherwise the runtime fails open and executes a real denoiser forward

Required outer-step keys:

- `spectrum_run_id`
- `spectrum_solver_step_id`
- `spectrum_time_coord`

Optional key:

- `spectrum_actual_forward`
- `spectrum_total_steps`

The built-in outer-step controller injects `run_id`, `solver_step_id`, `time_coord`, and `total_steps` automatically for normal ComfyUI sampling. In the current implementation, `time_coord` is the same ordinal outer-step index used by the forecaster, not the raw sigma value. `spectrum_actual_forward` can still be supplied by an external controller if it wants to force the per-step decision explicitly.

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
- `debug`

`window_size`, `flex_window`, and `warmup_steps` are now consumed by the built-in outer-step controller path. External controllers can also provide explicit per-step decisions through `spectrum_actual_forward`.

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
- Forecast coordinate: `spectrum_time_coord` in ordinal outer-step space
- Step identity: `spectrum_solver_step_id`
- Run boundary: `spectrum_run_id`
- Stream isolation: `uuids` + `cond_or_uncond` + input shape

The runtime may still use `sample_sigmas` length as a fallback for normalization when `spectrum_total_steps` is absent, but it does not use sigma values to decide solver steps or forecast eligibility. Raw sigma-space forecasting is not implemented in this tree.

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

Expected output:

```text
ok
```

## References

- Paper: [Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](https://arxiv.org/abs/2603.01623)
- Spectrum project page: [Spectrum](https://hanjq17.github.io/Spectrum/)
- Official implementation: [hanjq17/Spectrum](https://github.com/hanjq17/Spectrum)
