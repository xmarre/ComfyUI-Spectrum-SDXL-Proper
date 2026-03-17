# ComfyUI Spectrum SDXL Native

Native ComfyUI custom node that ports **Spectrum** to the **SDXL / U-Net** execution path.

This repo is intentionally narrower than a generic “works on every model” wrapper. It targets the native ComfyUI U-Net path used by SDXL-family checkpoints and hooks at the **final hidden feature before the SDXL output head**, which is the correct analogue of the official SDXL implementation.

## What this node does

Spectrum is a **training-free feature forecaster** for diffusion sampling acceleration.

Instead of running the SDXL U-Net on every step, it:

1. performs real U-Net forwards on selected steps,
2. caches the **final hidden feature** from the native U-Net path,
3. fits a small **Chebyshev + ridge regression** forecaster online,
4. predicts the hidden feature on skipped steps,
5. runs the normal SDXL output head on that predicted hidden feature.

This follows the paper’s core design:

- **Chebyshev polynomial** forecasting over time
- **ridge regression** coefficient fitting
- **last-block / last-hidden-only** caching instead of per-module caching
- **adaptive scheduling** via `window_size` + `flex_window`

## Why this repo exists separately

The existing public ComfyUI Spectrum SDXL node is useful as a proof of concept, but it is not a proper native port of the SDXL path.

The main issues are:

- it accelerates through a **top-level U-Net wrapper** instead of patching the **native internal SDXL U-Net path**,
- it forecasts the **model wrapper output** rather than the final hidden feature at the correct internal boundary,
- it relies on pass-reset heuristics instead of synchronizing directly with ComfyUI’s sigma schedule,
- it duplicates logic and broadens scope in ways that make behavior harder to reason about.

This repo instead patches the native ComfyUI `UNetModel._forward(...)` path and keeps the implementation minimal.

## Installation

Clone or copy this repository into your ComfyUI `custom_nodes` directory:

```bash
git clone <this-repo> ComfyUI/custom_nodes/ComfyUI-Spectrum-SDXL-Native
```

Then restart ComfyUI.

No additional Python dependencies are required beyond the normal ComfyUI stack.

## Node

### `Spectrum Apply SDXL`

**Inputs**

- `model` — ComfyUI `MODEL`
- `enabled` — enable/disable the patch
- `blend_weight` — `w` in the official implementation; mixes local linear extrapolation with the Chebyshev predictor
- `degree` — `M`, the Chebyshev degree
- `ridge_lambda` — `λ`, ridge regularization strength
- `window_size` — initial adaptive scheduling window
- `flex_window` — growth added to the scheduling window after each real forward
- `warmup_steps` — number of initial real forwards before forecasting is allowed
- `debug` — currently stored in runtime metadata only

**Output**

- patched `MODEL`

## Parameter guidance

### `blend_weight`

Controls the mixture:

```text
pred = (1 - blend_weight) * local_linear + blend_weight * chebyshev
```

- `1.0` = pure Chebyshev forecast
- `0.5` = official practical recommendation range center
- lower values bias toward local extrapolation

Recommended default: `0.50`

### `degree`

Chebyshev polynomial degree `M`.

Recommended default: `4`

The paper’s ablation found `M = 4` to be a strong practical choice.

### `ridge_lambda`

Regularization strength `λ` for the ridge solve.

Recommended default: `0.10`

Too small can become numerically unstable. Too large can underfit the feature trajectory.

### `window_size`

Initial number controlling how aggressively the method starts skipping steps.

Recommended default: `2.0`

### `flex_window`

Amount added to the adaptive window after each real forward.

Recommended default: `0.75`

Higher values are more aggressive and usually faster, but quality drops sooner.

### `warmup_steps`

Initial real steps before forecasting begins.

Recommended default: `5`

This matches the official repo’s base config.

## Usage

Typical placement:

```text
CheckpointLoaderSimple
  -> LoRA / model patches
  -> Spectrum Apply SDXL
  -> sampler / guider
```

It should sit **after** model-altering nodes such as LoRA loaders and **before** the sampler.

## Example settings

Balanced / official-style:

```text
blend_weight = 0.50
degree = 4
ridge_lambda = 0.10
window_size = 2.0
flex_window = 0.75
warmup_steps = 5
```

More conservative:

```text
blend_weight = 0.50
degree = 4
ridge_lambda = 0.10
window_size = 2.0
flex_window = 0.25
warmup_steps = 5
```

More aggressive:

```text
blend_weight = 0.50
degree = 4
ridge_lambda = 0.10
window_size = 2.0
flex_window = 1.5
warmup_steps = 5
```

## Implementation notes

### Hook point

The patch targets the native ComfyUI U-Net path after all `output_blocks` and before `self.out(h)`.

That is the correct SDXL analogue of the official SDXL implementation, which caches the final hidden feature before the output head.

### Forecast target

This repo forecasts the **final hidden U-Net feature** `h`, not the denoised output tensor.

That matters because Spectrum is defined as feature forecasting inside the denoiser, and the output head is still applied normally on forecasted steps.

### Schedule tracking

The runtime synchronizes with the sampler via `transformer_options["sample_sigmas"]` / `transformer_options["sigmas"]` when available, instead of trying to detect new runs purely from monotonic timestep changes.

### Step normalization

The official public code often assumes a 50-step run when mapping timesteps into the Chebyshev domain. This port instead normalizes against the **actual current sampler step count** derived from the sigma schedule.

That is a deliberate compatibility fix for normal ComfyUI usage, where users frequently run non-50-step schedules.

## Assumptions, caveats, and known limitations

### Scope

This node is designed for **native ComfyUI SDXL-family U-Net models**.

It may also work on some related U-Net checkpoints that use the same internal `UNetModel` path, but that is incidental rather than promised.

It is **not** intended for FLUX / SD3 / Wan / Hunyuan; those need backend-specific integrations.

### Custom block patches on skipped steps

On a forecasted step, the expensive internal U-Net blocks are skipped by design.

That means custom nodes that rely on mutating **internal block execution on every step** will not see those block-level effects on skipped steps. This is inherent to Spectrum’s acceleration strategy, not a bug in the port.

### Dynamic control modules

ControlNet-style residuals applied inside the U-Net are naturally only injected on actual forward steps. Forecasted steps instead use the predicted final hidden feature. In practice this can still work well, but very aggressive acceleration may degrade control fidelity earlier than plain sampling.

### Approximation relative to the paper

This repo is faithful to the core method, but one practical approximation remains explicit:

- the online forecaster uses the **discrete sampler step index** as the time variable for fitting/prediction, matching the official code path more closely than a reconstructed continuous-time parameterization.

### Compatibility maintenance

This repo reproduces ComfyUI’s current internal SDXL `UNetModel._forward(...)` structure closely. If ComfyUI changes that internal method in the future, this node may need a maintenance update.

## Repository structure

```text
ComfyUI-Spectrum-SDXL-Native/
├── __init__.py
├── nodes.py
├── pyproject.toml
├── LICENSE
├── README.md
├── comfyui_spectrum_sdxl/
│   ├── __init__.py
│   ├── config.py
│   ├── forecast.py
│   ├── runtime.py
│   └── sdxl.py
└── tests/
    └── smoke_runtime.py
```

## Smoke test

This repo includes a lightweight non-ComfyUI smoke test for the forecaster/runtime pieces:

```bash
python tests/smoke_runtime.py
```

Expected output:

```text
ok
```

## References

- Paper: `Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration`
- Project page: `https://hanjq17.github.io/Spectrum/`
- Official code: `https://github.com/hanjq17/Spectrum`
- Existing community SDXL node reviewed as reference: `https://github.com/ruwwww/comfyui-spectrum-sdxl`
