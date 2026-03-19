"""Native ComfyUI SDXL wrapper logic for Spectrum integration."""

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
_OUTER_STEP_CONTROLLER_KEY = "spectrum_sdxl_outer_step_controller"
_MODEL_FUNCTION_WRAPPER_KEY = "spectrum_sdxl_model_function_wrapper"

_RUN_ID_KEY = "spectrum_run_id"
_SOLVER_STEP_ID_KEY = "spectrum_solver_step_id"
_TIME_COORD_KEY = "spectrum_time_coord"
_TOTAL_STEPS_KEY = "spectrum_total_steps"


def _clone_model(model: Any) -> Any:
    """Clone a ComfyUI model when possible, otherwise return it unchanged."""
    return model.clone() if hasattr(model, "clone") else model


def _ensure_model_options(model: Any) -> Dict[str, Any]:
    """Ensure the model exposes a mutable ``model_options`` mapping."""
    if not hasattr(model, "model_options") or model.model_options is None:
        model.model_options = {}
    return model.model_options


def _ensure_transformer_options(model: Any) -> Dict[str, Any]:
    """Ensure ``model_options['transformer_options']`` exists and is mutable."""
    opts = _ensure_model_options(model)
    if "transformer_options" not in opts or opts["transformer_options"] is None:
        opts["transformer_options"] = {}
    return opts["transformer_options"]


def _locate_unet_inner_model(model: Any) -> Tuple[Optional[Any], Optional[str]]:
    """Locate the inner native ComfyUI diffusion model to patch."""
    outer = getattr(model, "model", None)
    if outer is not None and hasattr(outer, "diffusion_model"):
        return outer.diffusion_model, "model.diffusion_model"
    if hasattr(model, "diffusion_model"):
        return model.diffusion_model, "diffusion_model"
    return None, None


def _looks_like_comfy_unet(inner: Any) -> bool:
    """Return whether the located inner model matches the expected U-Net shape."""
    required = (
        "input_blocks",
        "output_blocks",
        "time_embed",
        "out",
        "model_channels",
    )
    return all(hasattr(inner, name) for name in required) and not hasattr(inner, "double_blocks")


def _resolve_runtime(transformer_options: Dict[str, Any]) -> Optional[SpectrumSDXLRuntime]:
    """Extract the active Spectrum runtime from transformer options."""
    runtime = transformer_options.get(_RUNTIME_KEY)
    if isinstance(runtime, SpectrumSDXLRuntime):
        return runtime
    return None


class _SpectrumOuterStepController:
    """Attach stable outer solver-step context before ComfyUI evaluates conditions."""

    def __init__(self, runtime: SpectrumSDXLRuntime, delegate=None):
        self.runtime = runtime
        self.delegate = delegate
        self._current_run_token = None
        self._run_serial = 0
        self._next_solver_step_id = 0
        self._active_step_marker = None
        self._active_solver_step_id = 0

    def _extract_time_coord(self, transformer_options: Dict[str, Any], solver_step_id: int) -> float:
        """Map the current solver step onto the active normalized schedule coordinate."""
        sample_sigmas = transformer_options.get("sample_sigmas", None)
        if sample_sigmas is None:
            return float(solver_step_id)
        try:
            values = tuple(float(v) for v in sample_sigmas.detach().flatten().tolist()[:-1])
        except Exception:
            return float(solver_step_id)
        if not values:
            return float(solver_step_id)
        idx = min(max(int(solver_step_id), 0), len(values) - 1)
        start = values[0]
        end = values[-1]
        denom = end - start
        if abs(denom) < 1e-12:
            return 0.0
        return float(((values[idx] - start) / denom) * 2.0 - 1.0)

    def _sample_sigmas_token(self, transformer_options: Dict[str, Any]):
        sample_sigmas = transformer_options.get("sample_sigmas", None)
        if sample_sigmas is None:
            return None
        try:
            return (
                int(sample_sigmas.data_ptr()),
                tuple(int(v) for v in sample_sigmas.shape),
                str(sample_sigmas.device),
                str(sample_sigmas.dtype),
            )
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return ("id", id(sample_sigmas))

    def _sigma_value(self, sigma) -> Optional[float]:
        """Resolve one scalar sigma value for restart detection."""
        try:
            return round(float(sigma.detach().flatten()[0].item()), 8)
        except Exception:
            return None

    def _begin_run(self, run_token) -> None:
        """Reset controller state for one logical sampling run."""
        self._current_run_token = run_token
        self._run_serial += 1
        self._next_solver_step_id = 0
        self._active_step_marker = None
        self._active_solver_step_id = 0

    def _extract_total_steps(self, transformer_options: Dict[str, Any]) -> int:
        sample_sigmas = transformer_options.get("sample_sigmas", None)
        if sample_sigmas is not None:
            try:
                return max(int(sample_sigmas.numel()) - 1, 1)
            except Exception:
                pass
        return max(int(self.runtime.last_info.get("num_steps", 0)), 1)

    def _ensure_step_context(
        self,
        *,
        transformer_options: Dict[str, Any],
        sigma,
        run_token_fallback,
        reuse_active_step: bool,
    ) -> None:
        """Stamp stable Spectrum outer-step metadata onto transformer options."""
        run_token = self._sample_sigmas_token(transformer_options)
        if run_token is None:
            run_token = run_token_fallback
        if run_token != self._current_run_token:
            self._begin_run(run_token)

        current_sigma = self._sigma_value(sigma)
        first_sigma = None
        sample_sigmas = transformer_options.get("sample_sigmas", None)
        if sample_sigmas is not None:
            try:
                first_sigma = self._sigma_value(sample_sigmas[0])
            except Exception:
                first_sigma = None

        if (
            self._next_solver_step_id > 1
            and first_sigma is not None
            and current_sigma is not None
            and current_sigma == first_sigma
            and self._active_step_marker is not None
            and self._active_step_marker[0] == run_token
            and self._active_step_marker[1] != current_sigma
        ):
            self._begin_run(run_token)

        if reuse_active_step:
            step_marker = (run_token, current_sigma)
            if self._active_step_marker != step_marker:
                solver_step_id = self._next_solver_step_id
                self._active_step_marker = step_marker
                self._active_solver_step_id = solver_step_id
                self._next_solver_step_id += 1
            else:
                solver_step_id = self._active_solver_step_id
        else:
            solver_step_id = self._next_solver_step_id
            self._active_step_marker = (run_token, current_sigma)
            self._active_solver_step_id = solver_step_id
            self._next_solver_step_id += 1

        transformer_options[_RUN_ID_KEY] = self._run_serial
        transformer_options[_SOLVER_STEP_ID_KEY] = solver_step_id
        transformer_options[_TIME_COORD_KEY] = self._extract_time_coord(transformer_options, solver_step_id)
        transformer_options[_TOTAL_STEPS_KEY] = self._extract_total_steps(transformer_options)

    def _ensure_outer_step_context(self, args: Dict[str, Any]) -> None:
        model_options = args["model_options"]
        transformer_options = model_options.setdefault("transformer_options", {})
        self._ensure_step_context(
            transformer_options=transformer_options,
            sigma=args.get("sigma", None),
            run_token_fallback=("model_options", id(model_options)),
            reuse_active_step=False,
        )

    def __call__(self, args):
        model_options = args["model_options"]
        self._ensure_outer_step_context(args)
        if self.runtime.cfg.debug:
            transformer_options = model_options.get("transformer_options", {})
            print(
                "[Spectrum SDXL] controller "
                f"run={transformer_options.get(_RUN_ID_KEY)} "
                f"step={transformer_options.get(_SOLVER_STEP_ID_KEY)} "
                f"time={transformer_options.get(_TIME_COORD_KEY)} "
                f"total={transformer_options.get(_TOTAL_STEPS_KEY)} "
                f"model_options_id={id(model_options)} "
                f"transformer_options_id={id(transformer_options)}"
            )
        if self.delegate is not None:
            return self.delegate(args)

        import comfy.samplers

        return comfy.samplers.calc_cond_batch(
            args["model"],
            args["conds"],
            args["input"],
            args["sigma"],
            model_options,
        )


class _SpectrumModelFunctionWrapper:
    """Backfill outer-step context for guider paths that bypass the sampler hook."""

    def __init__(self, controller: _SpectrumOuterStepController, delegate=None):
        self.controller = controller
        self.delegate = delegate

    def _apply_with_spectrum_context(self, apply_model, input_x, timestep, c):
        transformer_options = c.setdefault("transformer_options", {})

        if transformer_options.get(_SOLVER_STEP_ID_KEY) is None:
            self.controller._ensure_step_context(
                transformer_options=transformer_options,
                sigma=timestep,
                run_token_fallback=("transformer_options", id(transformer_options)),
                reuse_active_step=True,
            )

        # Delegated guider paths (for example AutoGuidance) can issue multiple distinct
        # same-step apply_model calls without exposing stable inner-branch identity.
        # Force the runtime onto the real path to avoid forecast stream aliasing.
        if self.delegate is not None:
            transformer_options["spectrum_actual_forward"] = True

        if self.controller.runtime.cfg.debug:
            print(
                "[Spectrum SDXL] model_wrapper "
                f"run={transformer_options.get(_RUN_ID_KEY)} "
                f"step={transformer_options.get(_SOLVER_STEP_ID_KEY)} "
                f"time={transformer_options.get(_TIME_COORD_KEY)} "
                f"total={transformer_options.get(_TOTAL_STEPS_KEY)} "
                f"forced_actual={transformer_options.get('spectrum_actual_forward', None)}"
            )

        return apply_model(input_x, timestep, **c)

    def __call__(self, apply_model, args):
        def spectrum_apply_model(input_x, timestep, **c):
            return self._apply_with_spectrum_context(apply_model, input_x, timestep, c)

        if self.delegate is not None:
            return self.delegate(spectrum_apply_model, args)
        return spectrum_apply_model(args["input"], args["timestep"], **args["c"])


def _install_outer_step_controller(model: Any, runtime: SpectrumSDXLRuntime) -> None:
    """Install the sampler-layer outer-step controller on the patched model."""
    model_options = _ensure_model_options(model)
    controller = model_options.get(_OUTER_STEP_CONTROLLER_KEY)
    if isinstance(controller, _SpectrumOuterStepController):
        controller.runtime = runtime
        model_options["sampler_calc_cond_batch_function"] = controller
        return

    previous = model_options.get("sampler_calc_cond_batch_function")
    controller = _SpectrumOuterStepController(runtime=runtime, delegate=previous)
    model_options[_OUTER_STEP_CONTROLLER_KEY] = controller
    model_options["sampler_calc_cond_batch_function"] = controller


def _install_model_function_wrapper(model: Any, controller: _SpectrumOuterStepController) -> None:
    """Install the compat wrapper for guider paths that call ``apply_model`` directly."""
    model_options = _ensure_model_options(model)
    wrapper = model_options.get(_MODEL_FUNCTION_WRAPPER_KEY)
    current = model_options.get("model_function_wrapper")

    if isinstance(current, _SpectrumModelFunctionWrapper) and wrapper is None:
        wrapper = current

    if isinstance(wrapper, _SpectrumModelFunctionWrapper):
        wrapper.controller = controller
        if current is not wrapper:
            wrapper.delegate = current
        model_options[_MODEL_FUNCTION_WRAPPER_KEY] = wrapper
        model_options["model_function_wrapper"] = wrapper
        return

    wrapper = _SpectrumModelFunctionWrapper(controller=controller, delegate=current)
    model_options[_MODEL_FUNCTION_WRAPPER_KEY] = wrapper
    model_options["model_function_wrapper"] = wrapper


def _wrap_sdxl_unet_forward(inner: Any) -> None:
    """Install the Spectrum-aware wrapper around the native SDXL ``_forward``."""
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
        """Run one SDXL denoiser call with explicit outer-step Spectrum context."""
        runtime = _resolve_runtime(transformer_options)
        if runtime is None or not runtime.cfg.enabled:
            return original_forward(x, timesteps, context, y, control, transformer_options, **kwargs)

        if transformer_options is None:
            transformer_options = {}

        if runtime.cfg.debug:
            print(
                "[Spectrum SDXL] unet "
                f"run={transformer_options.get(_RUN_ID_KEY)} "
                f"step={transformer_options.get(_SOLVER_STEP_ID_KEY)} "
                f"time={transformer_options.get(_TIME_COORD_KEY)} "
                f"total={transformer_options.get(_TOTAL_STEPS_KEY)} "
                f"transformer_options_id={id(transformer_options)} "
                f"stream_key_present="
                f"{runtime.stream_key(transformer_options, tuple(int(v) for v in x.shape)) is not None}"
            )

        from comfy.ldm.modules.diffusionmodules.openaimodel import (
            apply_control,
            forward_timestep_embed,
            timestep_embedding,
        )
        import torch as th

        decision = runtime.begin_step(transformer_options, timesteps, tuple(int(v) for v in x.shape))
        solver_step_id = decision["solver_step_id"]
        actual_forward = decision["actual_forward"]
        stream_key = decision["stream_key"]

        transformer_options["original_shape"] = list(x.shape)
        transformer_options["transformer_index"] = 0

        if not actual_forward:
            try:
                predicted_h = runtime.predict_feature(stream_key, solver_step_id)
            except Exception:
                predicted_h = None

            if predicted_h is not None:
                predicted_h = predicted_h.to(device=x.device, dtype=x.dtype)
                if not torch.isfinite(predicted_h).all():
                    predicted_h = None

            if predicted_h is not None:
                runtime.finalize_step(stream_key, solver_step_id, used_forecast=True)
                if getattr(inner, "predict_codebook_ids", False):
                    return inner.id_predictor(predicted_h)
                return inner.out(predicted_h)

        transformer_patches = transformer_options.get("patches", {})
        num_video_frames = kwargs.get("num_video_frames", transformer_options.get("num_video_frames", 1))
        image_only_indicator = kwargs.get(
            "image_only_indicator",
            transformer_options.get("image_only_indicator", None),
        )
        time_context = kwargs.get("time_context", None)

        hs = []
        t_emb = timestep_embedding(timesteps, inner.model_channels, repeat_only=False).to(x.dtype)
        emb = inner.time_embed(t_emb)

        if "emb_patch" in transformer_patches:
            for patch in transformer_patches["emb_patch"]:
                emb = patch(emb, inner.model_channels, transformer_options)

        if context is not None and inner.num_classes is None:
            pass
        elif inner.num_classes is not None:
            if y is None:
                raise ValueError("ComfyUI Spectrum SDXL patch expected class labels for a class-conditional model.")
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
        runtime.observe_actual_feature(stream_key, solver_step_id, h)

        if getattr(inner, "predict_codebook_ids", False):
            return inner.id_predictor(h)
        return inner.out(h)

    inner._spectrum_sdxl_original_forward = original_forward
    inner._forward = wrapped_forward
    inner._spectrum_sdxl_wrapped = True


class SDXLSpectrumPatcher:
    """Apply the native SDXL Spectrum patch to a ComfyUI model."""

    @staticmethod
    def patch(model: Any, cfg: SpectrumSDXLConfig) -> Any:
        """Clone the model, attach runtime metadata, and patch the inner U-Net."""
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

        _install_outer_step_controller(patched, runtime)
        controller = patched.model_options.get(_OUTER_STEP_CONTROLLER_KEY)
        if isinstance(controller, _SpectrumOuterStepController):
            _install_model_function_wrapper(patched, controller)

        inner, inner_name = _locate_unet_inner_model(patched)
        runtime.last_info["hook_target"] = inner_name

        if inner is not None and _looks_like_comfy_unet(inner):
            _wrap_sdxl_unet_forward(inner)
            runtime.last_info["patched"] = True
        else:
            runtime.last_info["patched"] = False

        return patched
