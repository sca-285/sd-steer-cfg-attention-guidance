"""Putting the enabled modules on the WebUI's UNet patcher for one sampling
pass, in a fixed order, so any combination can run together.

    before the model call   pre-CFG hook: the AND-prompt weights, the prompt
                            lists for CHG, and skipping the negative pass
                            once AG has turned CFG off
    inside the model        NAG (cross-attention), SAG's attention recorder,
                            TPG's token shuffle (only in its own pass)
    CFG                     one sampler_cfg_function: AG's check, then CHG's
                            shifted predictions, then FDG, APG, FDG with APG
                            in each band, or plain CFG (lib_steer.cfg)
    after CFG               one post-CFG function each for SAG, PAG, SEG,
                            TPG and SWG, each adding a correction from its
                            own extra model passes
"""

from __future__ import annotations

import math

import torch

from lib_steer import cfg as C
from lib_steer import chg as CH
from lib_steer import host, nag, tpg
from lib_steer.attn import SAGRecorder, pag_attention, parse_blocks, sag_blur_map, seg_attention

CFG_MODULES = ("fdg", "apg", "chg", "ag")
LATENT_FACTOR = 8     # image pixels per latent pixel (SD, SDXL, Flux)


class State:
    """Where the current sampling pass is, updated by the WebUI's CFG denoiser callback."""

    def __init__(self):
        self.active = False
        self.step = 0
        self.total = 1
        self.progress = 0.0
        self.weight = 1.0
        self.base_cfg = 1.0
        self.model_calls = 0
        self.extra_passes = {"pag": 0, "seg": 0, "sag": 0, "tpg": 0, "swg": 0, "chg": 0}
        self.negative_schedule = None
        self.warned_negative = False
        self.conds = (None, None)      # the prompt lists of the model call in progress (CHG)
        self.skipped = False           # this call has no negative pass (AG turned CFG off)
        self.ag_off_step = None        # step at which AG turned CFG off
        self.chg = None                # CHG solver of this pass
        self.notes = set()             # one-off messages already printed

    def note(self, text):
        if text not in self.notes:
            self.notes.add(text)
            print(f"[Steer] {text}")

    def on_cfg_denoiser(self, params):
        if not self.active:
            return
        self.step = int(getattr(params, "sampling_step", 0) or 0)
        self.total = int(getattr(params, "total_sampling_steps", 1) or 1)
        self.progress = min(max(self.step / max(self.total - 1, 1), 0.0), 1.0)
        if nag.CONTROLLER.enabled:
            negative = nag.negative_embedding(getattr(params, "text_uncond", None))
            if negative is None and self.negative_schedule is not None:
                # CFG 1: the WebUI did not encode the negative prompt; use the encoding made at install
                try:
                    from modules import prompt_parser
                    negative = nag.negative_embedding(prompt_parser.reconstruct_cond_batch(self.negative_schedule, self.step))
                except Exception:
                    negative = None
            nag.CONTROLLER.negative = negative
            if negative is None and not self.warned_negative:
                self.warned_negative = True
                print("[Steer] NAG: no negative prompt conditioning for this step; NAG is off for it")


STATE = State()


def _sigma_value(sigma) -> float:
    try:
        return float(torch.as_tensor(sigma).reshape(-1)[0])
    except Exception:
        return 0.0


def make_cfg_function(s):
    use_fdg, use_apg = s["fdg"]["enabled"], s["apg"]["enabled"]
    use_chg, use_ag = s["chg"]["enabled"], s["ag"]["enabled"]
    a = s["apg"]
    apg = C.APG(a["eta"], a["norm_threshold"], a["momentum"]) if use_apg and not use_fdg else None
    band_apg = C.BandAPG(a["eta"], a["norm_threshold"], a["momentum"]) if use_apg and use_fdg else None
    levels = int(s["fdg"]["levels"])
    low_share = float(s["fdg"]["low_share"])
    ag = s["ag"]
    chg_window = s["chg"]

    def combine(cond, uncond, w, sigma, weight):
        if use_fdg:
            # the low band's scale never drops below 1 (1 = no guidance there)
            scales = C.fdg_scales(w, max(1.0, w * low_share), levels)
            return C.fdg(cond, uncond, scales, band_apg, sigma, weight)
        if apg is not None:
            return apg(cond, uncond, w, sigma, weight)
        return uncond + (cond - uncond) * w * weight

    def cfg_function(args):
        STATE.model_calls += 1
        x = args["input"]
        cond = args["cond_denoised"]
        uncond = args.get("uncond_denoised")
        w = float(args["cond_scale"])
        if uncond is None or STATE.skipped or math.isclose(w, 1.0):
            return x - cond
        sigma = _sigma_value(args["sigma"])
        weight = STATE.weight

        if use_ag and STATE.ag_off_step is None and STATE.progress >= float(ag["start"]):
            if ag["compare"] == "Noise":
                sim = C.similarity(x - cond, x - uncond)
            else:
                sim = C.similarity(cond, uncond)
            if float(sim.min()) >= float(ag["threshold"]):
                # this step is already computed with CFG; the next ones skip the negative pass
                STATE.ag_off_step = STATE.step + 1
                print(f"[Steer] AG: similarity {float(sim.min()):.4f}, CFG off from step {STATE.ag_off_step + 1}")

        if use_chg and STATE.chg is not None and _in_window(chg_window):
            cond, uncond = chg_predictions(args, cond, uncond, w)

        return x - combine(cond, uncond, w, sigma, weight)

    return cfg_function


def chg_predictions(args, cond, uncond, w):
    """CHG: the positive and negative predictions at their shifted points."""
    cond_list, uncond_list = STATE.conds
    if cond_list is None or uncond_list is None:
        STATE.note("CHG: the prompts of this step were not available; plain CFG used")
        return cond, uncond
    model, options = args["model"], args["model_options"]
    x, sigma = args["input"], args["sigma"]
    sig = torch.as_tensor(sigma, device=x.device, dtype=torch.float32).reshape(-1)
    if sig.numel() != x.shape[0]:
        sig = sig[:1].expand(x.shape[0])
    abt_current = 1.0 / (1.0 + sig.double() ** 2)
    abt_smallest = 1.0 / (1.0 + host.sigma_max(model) ** 2)

    def at(dx):
        c = host.run_cond(model, cond_list, x + (w - 1.0) * dx, sigma, options)
        u = host.run_cond(model, uncond_list, x + w * dx, sigma, options)
        STATE.extra_passes["chg"] += 2
        return c, u

    def evaluate(dx):
        c, u = at(dx)
        return dx + c - u

    g0 = (cond - uncond).to(torch.float32)
    dx = STATE.chg.solve(lambda d: evaluate(d.to(x.dtype)).to(torch.float32), g0, w,
                         abt_current.to(torch.float32), float(abt_smallest))
    if not dx.abs().any():
        return cond, uncond
    return at(dx.to(x.dtype))


def make_pre_step():
    """Per model call, before the model runs: the AND-prompt weights, the
    prompt lists, and whether the negative pass is skipped (AG)."""

    def on_step(cond, uncond=None, sigma=None):
        try:
            STATE.weight = float(sum(c.get("strength", 1.0) for c in cond)) or 1.0
        except Exception:
            STATE.weight = 1.0
        STATE.conds = (cond, uncond)
        STATE.skipped = uncond is not None and STATE.ag_off_step is not None and STATE.step >= STATE.ag_off_step
        return STATE.skipped

    return on_step


def _in_window(p):
    return float(p["start"]) <= STATE.progress <= float(p["end"])


def make_perturbed_post(kind, s):
    """PAG or SEG: add scale * (cond - cond with perturbed self-attention)."""
    p = s[kind]
    scale = float(p["scale"])
    blocks = parse_blocks(p["blocks"])
    attn = pag_attention if kind == "pag" else seg_attention(float(p["blur_sigma"]))

    def post(args):
        denoised = args["denoised"]
        if scale == 0.0 or not _in_window(p):
            return denoised
        options = args["model_options"]
        for block, number, index in blocks:
            options = host.with_replace(options, attn, "attn1", block, number, index)
        perturbed = host.run_cond(args["model"], args["cond"], args["input"], args["sigma"], options)
        STATE.extra_passes[kind] += 1
        return denoised + (args["cond_denoised"] - perturbed) * scale

    return post


def make_sag_post(s, recorder):
    p = s["sag"]

    def post(args):
        denoised = args["denoised"]
        uncond_list = args.get("uncond")
        uncond_pred = args.get("uncond_denoised")
        if recorder.scores is None or uncond_list is None or uncond_pred is None or STATE.skipped:
            return denoised
        if min(denoised.shape[2:]) <= 4 or denoised.ndim != 4:
            return denoised
        x = args["input"]
        degraded = sag_blur_map(uncond_pred, recorder.scores, float(p["blur_sigma"]), float(p["threshold"]))
        noised = degraded + x - uncond_pred
        sag = host.run_cond(args["model"], uncond_list, noised, args["sigma"], args["model_options"])
        recorder.scores = None
        STATE.extra_passes["sag"] += 1
        return denoised + (degraded - sag) * float(p["scale"])

    return post


def make_tpg_post(s):
    """TPG: add scale * (cond - cond with shuffled tokens in the chosen blocks)."""
    p = s["tpg"]
    scale = float(p["scale"])

    def post(args):
        denoised = args["denoised"]
        if scale == 0.0 or not _in_window(p):
            return denoised
        options = host.with_transformer_option(args["model_options"], tpg.FLAG, True)
        perturbed = host.run_cond(args["model"], args["cond"], args["input"], args["sigma"], options)
        STATE.extra_passes["tpg"] += 1
        return denoised + (args["cond_denoised"] - perturbed) * scale

    return post


def make_swg_post(s):
    """SWG: add scale * (cond - cond predicted on overlapping smaller windows)."""
    p = s["swg"]
    scale = float(p["scale"])
    window_h = max(1, int(p["window_height"]) // LATENT_FACTOR)
    window_w = max(1, int(p["window_width"]) // LATENT_FACTOR)
    overlap = max(0, int(p["overlap"]) // LATENT_FACTOR)
    state = {"off": False}

    def post(args):
        denoised = args["denoised"]
        x = args["input"]
        if state["off"] or scale == 0.0 or not _in_window(p) or x.ndim != 4:
            return denoised
        h, w = x.shape[-2:]
        if window_h >= h and window_w >= w:
            STATE.note(f"SWG: the window ({window_w * LATENT_FACTOR}x{window_h * LATENT_FACTOR}) covers the whole "
                       f"image ({w * LATENT_FACTOR}x{h * LATENT_FACTOR}); make it smaller than the image. Skipped")
            state["off"] = True
            return denoised
        model, cond, sigma, options = args["model"], args["cond"], args["sigma"], args["model_options"]

        def predict(x_window):
            STATE.extra_passes["swg"] += 1
            return host.run_cond(model, cond, x_window, sigma, options)

        try:
            weak = C.sliding_window_prediction(x, window_h, window_w, min(overlap, window_h - 1, window_w - 1), predict)
        except Exception as e:
            # e.g. a ControlNet or inpainting input that cannot be cut into windows
            STATE.note(f"SWG: the model could not run on windows ({type(e).__name__}: {e}); skipped")
            state["off"] = True
            return denoised
        return denoised + (args["cond_denoised"] - weak) * scale

    return post


def install(p, s, log=print):
    """Patch a clone of p's UNet for this pass. Returns the list of modules applied."""
    unet = p.sd_model.forge_objects.unet.clone()
    applied, notes = [], []
    unet_model = host.is_unet(unet)
    hr = bool(getattr(p, "is_hr_pass", False))
    STATE.active = True
    STATE.step, STATE.progress, STATE.weight = 0, 0.0, 1.0
    STATE.base_cfg = float(getattr(p, "hr_cfg", p.cfg_scale) if hr else p.cfg_scale)
    STATE.conds, STATE.skipped, STATE.ag_off_step, STATE.chg = (None, None), False, None, None
    STATE.notes = set()
    STATE.extra_passes = {k: 0 for k in STATE.extra_passes}

    cfg_modules = [m for m in CFG_MODULES if s[m]["enabled"]]
    for m in ("chg", "ag"):
        if m in cfg_modules and STATE.base_cfg <= 1.0:
            notes.append(f"{m.upper()} needs CFG above 1; skipped")
            cfg_modules.remove(m)
    if "chg" in cfg_modules and host.is_flow(getattr(unet, "model", None)):
        notes.append("CHG works with SD 1.x, SD 2.x and SDXL (eps / v prediction), not flow models; skipped")
        cfg_modules.remove("chg")
    if cfg_modules:
        s = {**s, "chg": {**s["chg"], "enabled": "chg" in cfg_modules}, "ag": {**s["ag"], "enabled": "ag" in cfg_modules}}
        if "chg" in cfg_modules:
            c = s["chg"]
            STATE.chg = CH.CHG(c["reg_strength"], c["reg_range"], c["max_iter"], c["bases"], c["reuse"], c["log_tol"],
                               c["step_size"], c["anneal_speed"], c["anneal_strength"], c["aa_memory"])
        previous = unet.model_options.get("sampler_cfg_function")
        if previous is not None:
            name = getattr(previous, "__qualname__", None) or repr(previous)
            notes.append(f"replaced another CFG function ({name}); turn that one off to avoid confusion")
        unet.set_model_sampler_cfg_function(make_cfg_function(s))
        applied += cfg_modules
        host.install_pre_cfg(unet, make_pre_step())

    attention_modules = [m for m in ("sag", "pag", "seg", "tpg", "nag") if s[m]["enabled"]]
    if attention_modules and not unet_model:
        notes.append(f"{', '.join(m.upper() for m in attention_modules)} need a UNet model (SD 1.x, SD 2.x, SDXL); skipped")
        attention_modules = []

    if "sag" in attention_modules:
        if STATE.base_cfg <= 1.0:
            notes.append("SAG needs CFG above 1; skipped")
        else:
            recorder = SAGRecorder()
            unet.set_model_attn1_replace(recorder, "middle", 0, 0)
            unet.set_model_sampler_post_cfg_function(make_sag_post(s, recorder), disable_cfg1_optimization=True)
            applied.append("sag")
    for kind in ("pag", "seg"):
        if kind in attention_modules:
            unet.set_model_sampler_post_cfg_function(make_perturbed_post(kind, s))
            applied.append(kind)
    if "tpg" in attention_modules:
        dm = host.diffusion_model(unet)
        try:
            blocks = tpg.select_blocks(dm, s["tpg"]["blocks"])
        except ValueError as e:
            blocks = []
            notes.append(f"TPG: {e}")
        if blocks:
            tpg.wrap(dm)
            tpg.CONTROLLER.configure(blocks, seed=getattr(p, "seed", 0) or 0)
            unet.set_model_sampler_post_cfg_function(make_tpg_post(s))
            applied.append("tpg")
        else:
            notes.append("TPG found no matching transformer blocks; skipped")
    if s["swg"]["enabled"]:
        unet.set_model_sampler_post_cfg_function(make_swg_post(s))
        applied.append("swg")

    nag.CONTROLLER.configure("nag" in attention_modules, s["nag"]["scale"], s["nag"]["tau"],
                             s["nag"]["alpha"], s["nag"]["sigma_end"])
    STATE.negative_schedule = None
    STATE.warned_negative = False
    if "nag" in attention_modules and STATE.base_cfg <= 1.0:
        # some WebUIs do not encode the negative prompt at CFG 1; have it ready
        STATE.negative_schedule = _encode_negative(p, hr)
    if "nag" in attention_modules:
        layers = nag.wrap(host.diffusion_model(unet))
        if layers:
            unet.set_model_attn2_patch(nag.CONTROLLER.stash)
            applied.append("nag")
        else:
            nag.CONTROLLER.configure(False)
            notes.append("NAG found no cross-attention layers; skipped")

    p.sd_model.forge_objects.unet = unet
    for note in notes:
        log(f"[Steer] {note}")
    return applied


def _encode_negative(p, hr):
    """The negative prompt's conditioning schedule, for WebUIs that skip it at CFG 1."""
    try:
        from modules import prompt_parser
        prompts = getattr(p, "hr_negative_prompts", None) if hr else None
        prompts = prompts or getattr(p, "negative_prompts", None) or [getattr(p, "negative_prompt", "")]
        width = getattr(p, "hr_upscale_to_x", 0) if hr else getattr(p, "width", None)
        height = getattr(p, "hr_upscale_to_y", 0) if hr else getattr(p, "height", None)
        conditioning = prompt_parser.SdConditioning(list(prompts), is_negative_prompt=True,
                                                    width=width or None, height=height or None)
        steps = (getattr(p, "hr_second_pass_steps", 0) or p.steps) if hr else p.steps
        return prompt_parser.get_learned_conditioning(p.sd_model, conditioning, steps)
    except Exception as e:
        print(f"[Steer] NAG: could not encode the negative prompt ({e})")
        return None


def finish():
    """After a generation: nothing stays active."""
    STATE.active = False
    STATE.chg = None
    STATE.conds = (None, None)
    nag.CONTROLLER.configure(False)
    tpg.CONTROLLER.configure(())
