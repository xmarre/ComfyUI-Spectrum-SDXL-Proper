from __future__ import annotations

from .comfyui_spectrum_sdxl.config import SpectrumSDXLConfig
from .comfyui_spectrum_sdxl.sdxl import SDXLSpectrumPatcher


class SpectrumApplySDXL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "blend_weight": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "degree": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
                "ridge_lambda": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 10.0, "step": 0.01}),
                "window_size": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 32.0, "step": 0.05}),
                "flex_window": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 16.0, "step": 0.05}),
                "warmup_steps": ("INT", {"default": 5, "min": 0, "max": 64, "step": 1}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "sampling/spectrum"

    def apply(
        self,
        model,
        enabled,
        blend_weight,
        degree,
        ridge_lambda,
        window_size,
        flex_window,
        warmup_steps,
        debug,
    ):
        if not enabled:
            return (model,)

        cfg = SpectrumSDXLConfig(
            enabled=enabled,
            blend_weight=blend_weight,
            degree=degree,
            ridge_lambda=ridge_lambda,
            window_size=window_size,
            flex_window=flex_window,
            warmup_steps=warmup_steps,
            debug=debug,
        ).validated()
        return (SDXLSpectrumPatcher.patch(model, cfg),)


NODE_CLASS_MAPPINGS = {
    "SpectrumApplySDXL": SpectrumApplySDXL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpectrumApplySDXL": "Spectrum Apply SDXL",
}
