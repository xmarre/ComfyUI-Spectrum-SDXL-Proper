from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import torch

from .config import SpectrumSDXLConfig
from .runtime import SpectrumSDXLRuntime

_RUNTIME_KEY = "spectrum_sdxl_runtime"
_CFG_KEY = "spectrum_sdxl_cfg"
_ENABLED_KEY = "spectrum_sdxl_enabled"
_BACKEND_KEY = "spectrum_backend"


def _clone_model(model: Any) -> Any:
    return model.clone() if hasattr(model, "clone") else model


def _ensure_model_options(model: Any) -> Dict[str, Any]:
    if not hasattr(model, "model_options") or model.model_options is None:
        model.model_options = {}
    return model.model_options


def _ensure_transformer_options(model: Any) -> Dict[str, Any]:
    opts = _ensure_model_options(model)
    if "transformer_options" not in opts or opts["transformer_options"] is None:
        opts["transformer_options"] = {}
    return opts["transformer_options"]


def _locate_unet_inner_model(model: Any) -> Tuple[Optional[Any], Optional[str]]:
    outer = getattr(model, "model", None)
    if outer is not None and hasattr(outer, "diffusion_model"):
        return outer.diffusion_model, "model.diffusion_model"
    if hasattr(model, "diffusion_model"):
        return model.diffusion_model, "diffusion_model"
    return None, None


def _looks_like_comfy_unet(inner: Any) -> bool:
    required = (
        "input_blocks",
        "output_blocks",
        "time_embed",
        "out",
        "model_channels",
    )
    return all(hasattr(inner, name) for name in required) and not hasattr(inner, "double_blocks")


def _resolve_runtime(transformer_options: Dict[str, Any]) -> Optional[SpectrumSDXLRuntime]:
    runtime = transformer_options.get(_RUNTIME_KEY)
    if isinstance(runtime, SpectrumSDXLRuntime):
        return runtime
    return None


def _wrap_sdxl_unet_forward(inner: Any) -> None:
    if getattr(inner, "_spectrum_sdxl_wrapped", False):
        return

    original_forward = inner._forward

    def wrapped_forward(
        x,
        timesteps=None,
        context=None,
        y=None,
        control=None,
        transformer_options={},
        **kwargs,
    ):
        runtime = _resolve_runtime(transformer_options)
        if runtime is None or not runtime.cfg.enabled:
            return original_forward(x, timesteps, context, y, control, transformer_options, **kwargs)

        if transformer_options is None:
            transformer_options = {}

        # Lazy imports keep the repository importable outside a ComfyUI runtime.
        from comfy.ldm.modules.diffusionmodules.openaimodel import (
            apply_control,
            forward_timestep_embed,
            timestep_embedding,
        )
        import torch as th

        decision = runtime.begin_step(transformer_options, timesteps, tuple(int(v) for v in x.shape))
        global_step_idx = decision["global_step_idx"]
        actual_forward = decision["actual_forward"]
        stream_key = decision["stream_key"]

        transformer_options["original_shape"] = list(x.shape)
        transformer_options["transformer_index"] = 0

        if not actual_forward:
            try:
                predicted_h = runtime.predict_feature(stream_key, global_step_idx)
            except Exception:
                predicted_h = None

            if predicted_h is not None:
                predicted_h = predicted_h.to(device=x.device, dtype=x.dtype)
                if not torch.isfinite(predicted_h).all():
                    predicted_h = None

            if predicted_h is not None:
                if getattr(inner, "predict_codebook_ids", False):
                    return inner.id_predictor(predicted_h)
                return inner.out(predicted_h)

        transformer_patches = transformer_options.get("patches", {})
        num_video_frames = kwargs.get("num_video_frames", getattr(inner, "default_num_video_frames", 1))
        image_only_indicator = kwargs.get("image_only_indicator", None)
        time_context = kwargs.get("time_context", None)

        assert (y is not None) == (inner.num_classes is not None), (
            "must specify y if and only if the model is class-conditional"
        )

        hs = []
        t_emb = timestep_embedding(timesteps, inner.model_channels, repeat_only=False).to(x.dtype)
        emb = inner.time_embed(t_emb)

        if "emb_patch" in transformer_patches:
            for patch in transformer_patches["emb_patch"]:
                emb = patch(emb, inner.model_channels, transformer_options)

        if inner.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + inner.label_emb(y)

        h = x
        for block_id, module in enumerate(inner.input_blocks):
            transformer_options["block"] = ("input", block_id)
            h = forward_timestep_embed(
                module,
                h,
                emb,
                context,
                transformer_options,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )
            h = apply_control(h, control, "input")

            if "input_block_patch" in transformer_patches:
                for patch in transformer_patches["input_block_patch"]:
                    h = patch(h, transformer_options)

            hs.append(h)

            if "input_block_patch_after_skip" in transformer_patches:
                for patch in transformer_patches["input_block_patch_after_skip"]:
                    h = patch(h, transformer_options)

        transformer_options["block"] = ("middle", 0)
        if inner.middle_block is not None:
            h = forward_timestep_embed(
                inner.middle_block,
                h,
                emb,
                context,
                transformer_options,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )
        h = apply_control(h, control, "middle")

        for block_id, module in enumerate(inner.output_blocks):
            transformer_options["block"] = ("output", block_id)
            hsp = hs.pop()
            hsp = apply_control(hsp, control, "output")

            if "output_block_patch" in transformer_patches:
                for patch in transformer_patches["output_block_patch"]:
                    h, hsp = patch(h, hsp, transformer_options)

            h = th.cat([h, hsp], dim=1)
            del hsp

            output_shape = hs[-1].shape if len(hs) > 0 else None
            h = forward_timestep_embed(
                module,
                h,
                emb,
                context,
                transformer_options,
                output_shape,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )

        h = h.type(x.dtype)
        runtime.observe_actual_feature(stream_key, global_step_idx, h)

        if getattr(inner, "predict_codebook_ids", False):
            return inner.id_predictor(h)
        return inner.out(h)

    inner._spectrum_sdxl_original_forward = original_forward
    inner._forward = wrapped_forward
    inner._spectrum_sdxl_wrapped = True


class SDXLSpectrumPatcher:
    @staticmethod
    def patch(model: Any, cfg: SpectrumSDXLConfig) -> Any:
        cfg = cfg.validated()
        patched = _clone_model(model)

        tr_opts = _ensure_transformer_options(patched)
        runtime = tr_opts.get(_RUNTIME_KEY)
        if isinstance(runtime, SpectrumSDXLRuntime):
            runtime.update_cfg(cfg)
        else:
            runtime = SpectrumSDXLRuntime(cfg)

        tr_opts[_CFG_KEY] = cfg
        tr_opts[_RUNTIME_KEY] = runtime
        tr_opts[_ENABLED_KEY] = cfg.enabled
        tr_opts[_BACKEND_KEY] = "sdxl_native"
        tr_opts["spectrum_sdxl_cfg_dict"] = asdict(cfg)

        inner, inner_name = _locate_unet_inner_model(patched)
        runtime.last_info["hook_target"] = inner_name

        if inner is not None and _looks_like_comfy_unet(inner):
            _wrap_sdxl_unet_forward(inner)
            runtime.last_info["patched"] = True
        else:
            runtime.last_info["patched"] = False

        return patched
