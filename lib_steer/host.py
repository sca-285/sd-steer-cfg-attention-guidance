"""What differs between Forge, reForge and Forge Classic (Neo), in one place.

All three keep ComfyUI's patcher interface (model_options with
sampler_cfg_function / sampler_post_cfg_function / sampler_pre_cfg_function,
transformer_options["patches_replace"] for attention); they differ in
where the batch runner lives and in the pre-CFG hook's signature.

                    Forge / Neo                       reForge
  batch runner      backend.sampling.sampling_        ldm_patched.modules.samplers.
                    function.calc_cond_uncond_batch   calc_cond_batch
  pre-CFG hook      before the model call:            after the model call:
                    fn(model, cond, uncond, x, t,     fn(args) -> conds_out
                       model_options) -> same tuple
  skipping uncond   pre-CFG hook returns uncond=None  sampler_calc_cond_batch_function
"""

from __future__ import annotations

import functools


def _importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


@functools.lru_cache(maxsize=None)
def name() -> str:
    if _importable("ldm_patched.modules.samplers"):
        return "reforge"
    if _importable("modules_forge.packages.k_diffusion.sampling"):
        return "neo"
    return "forge"


def is_reforge() -> bool:
    return name() == "reforge"


def run_cond(model, cond, x, sigma, model_options):
    """The cond prediction (x0) for `cond` at x, sigma, with these model_options."""
    if is_reforge():
        from ldm_patched.modules.samplers import calc_cond_batch
        return calc_cond_batch(model, [cond], x, sigma, model_options)[0]
    from backend.sampling.sampling_function import calc_cond_uncond_batch
    return calc_cond_uncond_batch(model, cond, None, x, sigma, model_options)[0]


def with_replace(model_options, fn, kind, block, number, index=None):
    """A copy of model_options with an attention replacement for one block.

    Only the dictionaries on the path are copied, as ComfyUI does, so the
    options of the main pass are never touched.
    """
    options = dict(model_options)
    to = dict(options.get("transformer_options", {}))
    replace = dict(to.get("patches_replace", {}))
    kind_map = dict(replace.get(kind, {}))
    key = (block, number) if index is None else (block, number, index)
    kind_map[key] = fn
    replace[kind] = kind_map
    to["patches_replace"] = replace
    options["transformer_options"] = to
    return options


def with_transformer_option(model_options, key, value):
    """A copy of model_options with transformer_options[key] = value (dictionaries on the path copied)."""
    options = dict(model_options)
    to = dict(options.get("transformer_options", {}))
    to[key] = value
    options["transformer_options"] = to
    return options


def sigma_max(model, default=14.6146):
    """The largest sigma of the model's schedule (SD's is 14.6)."""
    for candidate in (getattr(model, "model_sampling", None), getattr(model, "predictor", None)):
        value = getattr(candidate, "sigma_max", None) if candidate is not None else None
        if value is not None:
            try:
                return float(value)
            except Exception:
                pass
    return default


def diffusion_model(unet_patcher):
    model = getattr(unet_patcher, "model", None)
    return getattr(model, "diffusion_model", None)


def is_unet(unet_patcher) -> bool:
    """SD 1.x / 2.x / SDXL-style UNet (has a middle block with attention)."""
    dm = diffusion_model(unet_patcher)
    return dm is not None and hasattr(dm, "middle_block") and hasattr(dm, "output_blocks")


def is_flow(model) -> bool:
    """Flow-matching model (Flux, SD3, Wan...), from the model passed to the CFG hooks."""
    for candidate in (getattr(model, "predictor", None), getattr(model, "model_sampling", None)):
        if candidate is None:
            continue
        if getattr(candidate, "prediction_type", None) == "const":
            return True
        if any(cls.__name__ == "CONST" for cls in type(candidate).__mro__):
            return True
    return False


def install_pre_cfg(unet, on_step):
    """Call on_step(cond_list, uncond_list, sigma) before CFG; return True from it to skip the uncond pass.

    Forge / Neo: the hook runs before the model call, so skipping saves the
    whole uncond pass. reForge: the model call is wrapped instead.
    """
    if is_reforge():
        options = unet.model_options
        previous = options.get("sampler_calc_cond_batch_function")

        def calc(args):
            conds = list(args["conds"])
            uncond = conds[1] if len(conds) > 1 else None
            if on_step(conds[0], uncond, args.get("sigma")) and uncond is not None:
                conds[1] = None
            if previous is not None:
                return previous({**args, "conds": conds})
            from ldm_patched.modules.samplers import calc_cond_batch
            return calc_cond_batch(args["model"], conds, args["input"], args["sigma"], args["model_options"])

        unet.model_options["sampler_calc_cond_batch_function"] = calc
        return

    def pre_cfg(model, cond, uncond, x, timestep, model_options):
        if on_step(cond, uncond, timestep):
            uncond = None
        return model, cond, uncond, x, timestep, model_options

    unet.set_model_sampler_pre_cfg_function(pre_cfg)
