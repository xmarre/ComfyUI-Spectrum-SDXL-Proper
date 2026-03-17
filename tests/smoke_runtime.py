from __future__ import annotations

import torch

from comfyui_spectrum_sdxl.config import SpectrumSDXLConfig
from comfyui_spectrum_sdxl.runtime import SpectrumSDXLRuntime


def main() -> None:
    cfg = SpectrumSDXLConfig(
        blend_weight=0.5,
        degree=4,
        ridge_lambda=0.1,
        window_size=2.0,
        flex_window=0.75,
        warmup_steps=2,
    ).validated()
    runtime = SpectrumSDXLRuntime(cfg)

    sample_sigmas = torch.linspace(1.0, 0.0, 6)
    transformer_options = {"sample_sigmas": sample_sigmas}

    seen_actual = 0
    seen_forecast = 0
    for i in range(5):
        timesteps = torch.tensor([float(i)])
        decision = runtime.begin_step(transformer_options, timesteps)
        if decision["actual_forward"]:
            seen_actual += 1
            feature = torch.full((2, 8, 4, 4), float(i), dtype=torch.float16)
            runtime.observe_actual_feature(i, feature)
        else:
            seen_forecast += 1
            pred = runtime.predict_feature(i)
            assert pred.shape == (2, 8, 4, 4)
            assert torch.isfinite(pred).all()

    assert seen_actual >= 2
    assert seen_forecast >= 1
    print("ok")


if __name__ == "__main__":
    main()
