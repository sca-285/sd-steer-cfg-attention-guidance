"""Every module through the WebUI's own sampling code and UNet.

    cd <webui root>; <that webui's python> <this extension>/tests/test_in_host.py

Boots the WebUI (tests/_boot.py) without a checkpoint, builds a small UNet
with the WebUI's own UNet class (random weights), wraps it in the WebUI's
own model patcher, and runs one CFG step through the WebUI's own
sampling_function. So the attention patches go through the WebUI's
transformer blocks, the extra passes through its batch runner, and the CFG
hooks through its CFG code. Each module's effect is compared against a
reference computed a different way (modules patched directly, no hooks).
"""
import copy
import pathlib
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _boot import KIND  # noqa: E402  (boots the WebUI)

import torch  # noqa: E402

from lib_steer import attn as A, cfg as C, chg as CG, guidance, host, nag, settings as S, tpg  # noqa: E402

torch.manual_seed(0)
fails = []


def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   {extra}" if extra != "" else ""))
    if not cond:
        fails.append(name)


def close(a, b, tol=1e-4):
    return (a - b).abs().max().item() <= tol * max(1.0, b.abs().max().item())


def same_effect(out, ref, base, rel=0.02):
    """out - base matches ref - base to `rel` of the effect's size (and the effect is real)."""
    effect = (ref - base).abs().max().item()
    err = (out - ref).abs().max().item()
    return effect > 1e-5 and err <= rel * effect + 1e-7, f"effect {effect:.2e}, error {err:.2e}"


print(f"\n=== {KIND} (host detected as {host.name()})")
CTX = 32
CFG = dict(in_channels=4, model_channels=32, out_channels=4, num_res_blocks=1, channel_mult=(1, 2),
           num_head_channels=16, use_spatial_transformer=True, transformer_depth=[1, 1],
           transformer_depth_output=[1, 1, 1, 1], transformer_depth_middle=1, context_dim=CTX,
           use_linear_in_transformer=True)

if KIND == "reforge":
    from ldm_patched.ldm.modules.diffusionmodules.openaimodel import UNetModel
    from ldm_patched.modules.model_patcher import ModelPatcher
    from ldm_patched.modules import samplers as rs
    from modules_forge.forge_sampler import cond_from_a1111_to_patched_ldm
    import ldm_patched.modules.ops as ops
    dm = UNetModel(image_size=16, operations=ops.disable_weight_init, **CFG)
else:
    from backend.nn.unet import IntegratedUNet2DConditionModel
    from backend.sampling.condition import compile_conditions
    from backend.sampling import sampling_function as sf
    dm = IntegratedUNet2DConditionModel(**CFG)

# random weights that do something: the zero-initialised output layers of
# the WebUI UNet would make every block a no-op
for name_, prm in dm.named_parameters():
    # sharp attention (large q, k) so perturbing it visibly changes the output
    std = 1.0 if ("to_q" in name_ or "to_k" in name_) else 0.4 if ("attn" in name_ or "proj" in name_) else 0.08
    torch.nn.init.normal_(prm, std=std)
dm.eval()


class FakeModel(torch.nn.Module):
    """Stands in for the WebUI's model wrapper: apply_model -> denoised prediction."""

    def __init__(self, diffusion_model):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.rows = []                  # cond_or_uncond of every call

    def apply_model(self, x, t, c_crossattn=None, transformer_options={}, **kwargs):
        self.rows.append(list(transformer_options.get("cond_or_uncond", [])))
        out = self.diffusion_model(x, timesteps=t, context=c_crossattn, transformer_options=transformer_options)
        return x - 0.5 * out

    def memory_required(self, shape, **kwargs):
        return 0

    # reForge's batch runner asks for these
    def extra_conds_shapes(self, **kwargs):
        return {}

    def model_dtype(self):
        return torch.float32


model = FakeModel(dm)
if KIND == "reforge":
    base = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
else:
    from backend.patcher.unet import UnetPatcher
    base = UnetPatcher(model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"),
                       **({"current_device": torch.device("cpu")} if "current_device" in UnetPatcher.__init__.__code__.co_varnames
                          or KIND == "neo" else {}))
base_options = copy.deepcopy(base.model_options)

B = 2
x = torch.randn(B, 4, 16, 12) * 2.0
sigma = torch.tensor([2.0] * B)
pos = torch.randn(B, 7, CTX)
neg = torch.randn(B, 9, CTX)


def conds(t):
    return cond_from_a1111_to_patched_ldm(t) if KIND == "reforge" else compile_conditions(t)


def step(patcher, cfg, sig=sigma):
    model.rows.clear()
    if KIND == "reforge":
        return rs.sampling_function(patcher.model, x, sig, conds(neg), conds(pos), cfg, patcher.model_options, seed=0)
    return sf.sampling_function_inner(patcher.model, x, sig, conds(neg), conds(pos), cfg, patcher.model_options, 0)


def preds(options):
    """cond and uncond predictions with these model_options (no CFG hooks)."""
    c = host.run_cond(model, conds(pos), x, sigma, options)
    u = host.run_cond(model, conds(neg), x, sigma, options)
    return c, u


def run(settings, cfg=5.0, progress_step=5, total=11, p_extra=None):
    guidance.finish()
    p = types.SimpleNamespace(sd_model=types.SimpleNamespace(forge_objects=types.SimpleNamespace(unet=base)),
                              cfg_scale=cfg, is_hr_pass=False, init_images=None, extra_generation_params={})
    if p_extra:
        p.__dict__.update(p_extra)
    applied = guidance.install(p, settings, log=lambda m: print("      note:", m))
    params = types.SimpleNamespace(sampling_step=progress_step, total_sampling_steps=total, text_uncond=neg)
    guidance.STATE.on_cfg_denoiser(params)
    out = step(p.sd_model.forge_objects.unet, cfg)
    rows = [r[:] for r in model.rows]
    return out, applied, rows, p


def on(*modules, **params):
    s = S.defaults()
    s["general"]["enabled"] = True
    for m in modules:
        s[m]["enabled"] = True
    for key, value in params.items():
        module, name = key.split("__")
        s[module][name] = value
    return s


with torch.no_grad():
    c0, u0 = preds(base.model_options)
    plain = u0 + 5.0 * (c0 - u0)
    ok("baseline: WebUI CFG == u + w (c - u)", close(step(base, 5.0), plain))
    rel = ((c0 - u0).abs().mean() / (x - c0).abs().mean()).item()
    ok("sanity: the tiny UNet responds to the prompt (|c - u| / |model output| > 5%)", rel > 0.05, f"{rel:.3f}")

    # ------------------------------------------------------------ guidance formula
    out, applied, _, _ = run(on("apg"))
    ref = C.APG(0.0, 15.0, -0.5)(c0, u0, 5.0, 2.0)
    ok("APG through the WebUI == APG on the WebUI's predictions", close(out, ref), applied)

    out, applied, _, _ = run(on("fdg"))
    ref = C.fdg(c0, u0, [5.0, 2.5])
    ok("FDG through the WebUI == FDG on the WebUI's predictions (scales 5 and 2.5)", close(out, ref), applied)
    ok("FDG differs from plain CFG", not close(out, plain))
    out, _, _, _ = run(on("fdg", fdg__low_share=1.0))
    ok("FDG with low share 1 == plain CFG", close(out, plain))
    out, _, _, _ = run(on("fdg", fdg__levels=4, fdg__low_share=0.2))
    ok("FDG 4 levels, low scale floored at 1", close(out, C.fdg(c0, u0, C.fdg_scales(5.0, 1.0, 4))))

    out, applied, _, _ = run(on("fdg", "apg"))
    ref = C.fdg(c0, u0, [5.0, 2.5], C.BandAPG(0.0, 15.0, -0.5), 2.0)
    ok("FDG + APG == FDG with APG in each band", close(out, ref), applied)

    out, _, rows, _ = run(on("fdg", "apg"), cfg=1.0)
    ok("FDG / APG at CFG 1: plain cond, no negative pass", close(out, c0) and all(1 not in r for r in rows), rows)

    # ------------------------------------------------------------ attention: references by direct module patching
    mid_attn1 = [m for n, m in dm.named_modules() if n.startswith("middle_block") and n.endswith("attn1")]
    ok("the tiny UNet has a middle-block transformer", len(mid_attn1) == 1)

    def with_attn1(fn):
        """Run the cond pass with every middle attn1 replaced by fn(q, k, v, heads)."""
        saved = [m.forward for m in mid_attn1]
        for m in mid_attn1:
            def fwd(self, x_, context=None, value=None, mask=None, **kw):
                q = self.to_q(x_)
                k = self.to_k(x_ if context is None else context)
                v = self.to_v(x_ if value is None else value)
                return self.to_out(fn(q, k, v, self.heads))
            m.forward = types.MethodType(fwd, m)
        try:
            return host.run_cond(model, conds(pos), x, sigma, base.model_options)
        finally:
            for m, f in zip(mid_attn1, saved):
                m.forward = f

    identity = with_attn1(lambda q, k, v, h: v)
    effect = ((c0 - identity).abs().mean() / (c0 - u0).abs().mean()).item()
    ok("sanity: the middle block's self-attention matters (identity changes the cond by > 2% of CFG's effect)",
       effect > 0.02, f"{effect:.3f}")
    out, applied, rows, _ = run(on("pag", pag__scale=3.0))
    good, info = same_effect(out, plain + 3.0 * (c0 - identity), plain)
    ok("PAG == CFG + 3 (cond - cond with identity self-attention)", good, info)
    ok("PAG: one extra cond-only pass", rows.count([0]) >= 1, rows)

    out, applied, rows, _ = run(on("pag", pag__scale=3.0), cfg=1.0)
    good, info = same_effect(out, c0 + 3.0 * (c0 - identity), c0)
    ok("PAG at CFG 1 == cond + 3 (cond - perturbed), no negative pass", good and all(1 not in r for r in rows), info)

    mean_q = with_attn1(lambda q, k, v, h: A.attention(q.mean(1, keepdim=True).expand_as(q), k, v, h))
    seg_calls = {"n": 0}
    real_seg = guidance.seg_attention

    def spy_seg(sigma_):
        fn = real_seg(sigma_)

        def wrapped(*a, **k):
            seg_calls["n"] += 1
            return fn(*a, **k)
        return wrapped
    guidance.seg_attention = spy_seg
    out, applied, _, _ = run(on("seg", seg__scale=2.0))
    guidance.seg_attention = real_seg
    ok("SEG's attention runs once per extra pass, in the one middle-block transformer", seg_calls["n"] == 1, seg_calls["n"])
    good, info = same_effect(out, plain + 2.0 * (c0 - mean_q), plain)
    ok("SEG (infinite blur) == CFG + 2 (cond - cond with mean query)", good, info)
    out1, _, _, _ = run(on("seg", seg__scale=2.0, seg__blur_sigma=1.0))
    out0, _, _, _ = run(on("seg", seg__scale=2.0, seg__blur_sigma=0.0))
    d_inf = (out - plain).abs().mean().item()
    d_1 = (out1 - plain).abs().mean().item()
    d_0 = (out0 - plain).abs().mean().item()
    ok("SEG: blur 0 is no perturbation, blur 1 is milder than infinite blur", d_0 < 1e-6 and 0 < d_1 < d_inf,
       f"effect: sigma 0 {d_0:.2e}, sigma 1 {d_1:.2e}, infinite {d_inf:.2e}")

    out, applied, rows, _ = run(on("pag", pag__blocks="output 1"))
    ok("PAG on another block ('output 1') runs and differs", not close(out, plain) and torch.isfinite(out).all().item(), applied)
    try:
        A.parse_blocks("sideways 3")
        ok("bad block names are refused", False)
    except ValueError:
        ok("bad block names are refused", True)

    out, applied, rows, _ = run(on("sag", sag__scale=0.5))
    ok("SAG: runs, finite, changes the result", torch.isfinite(out).all().item() and not close(out, plain), applied)
    ok("SAG: extra pass is an uncond pass", guidance.STATE.extra_passes["sag"] >= 1 and [1] in rows, rows)
    _, applied, _, _ = run(on("sag"), cfg=1.0)
    ok("SAG is skipped at CFG 1", "sag" not in applied)

    # ------------------------------------------------------------ NAG
    attn2 = nag.cross_attention_layers(dm)
    blocks = [m for m in dm.modules() if type(m).__name__ == "BasicTransformerBlock"]
    ok("NAG finds every cross-attention layer", len(attn2) == len(blocks) == 7, len(attn2))

    def nag_reference(scale, tau, alpha, cfg):
        """Cond pass with each attn2 computing NAG itself (no wrapper, no stash)."""
        saved = [m.forward for m in attn2]
        for m in attn2:
            def fwd(self, x_, context=None, value=None, mask=None, **kw):
                q = self.to_q(x_)
                zp = A.attention(q, self.to_k(context), self.to_v(context), self.heads)
                zn = A.attention(q, self.to_k(neg), self.to_v(neg), self.heads)
                return self.to_out(nag.nag(zp, zn, scale, tau, alpha))
            m.forward = types.MethodType(fwd, m)
        try:
            return host.run_cond(model, conds(pos), x, sigma, base.model_options)
        finally:
            for m, f in zip(attn2, saved):
                m.forward = f

    out, applied, rows, _ = run(on("nag"), cfg=1.0)
    ref = nag_reference(5.0, 2.5, 0.25, 1.0)
    good, info = same_effect(out, ref, c0)
    ok("NAG at CFG 1 == cond pass with NAG in every cross-attention", good, info)
    ok("NAG at CFG 1: no negative pass (the negative acts inside attention)", all(1 not in r for r in rows), rows)
    out, _, _, _ = run(on("nag"), cfg=5.0)
    good, info = same_effect(out, u0 + 5.0 * (ref - u0), plain)
    ok("NAG with CFG: only the cond prediction changes", good, info)
    out, _, _, _ = run(on("nag", nag__sigma_end=3.0), cfg=1.0)
    ok("NAG stops below sigma_end (sigma 2 < 3 -> plain cond)", close(out, c0))


    # ------------------------------------------------------------ AG
    sim = float(C.similarity(c0, u0).min())
    guidance.finish()
    p_ag = types.SimpleNamespace(sd_model=types.SimpleNamespace(forge_objects=types.SimpleNamespace(unet=base)),
                                 cfg_scale=5.0, is_hr_pass=False, extra_generation_params={})
    guidance.install(p_ag, on("ag", ag__threshold=max(0.9, sim - 1e-3)), log=lambda m: None)
    patched = p_ag.sd_model.forge_objects.unet
    guidance.STATE.on_cfg_denoiser(types.SimpleNamespace(sampling_step=3, total_sampling_steps=10, text_uncond=neg))
    first = step(patched, 5.0)
    rows_first = [r[:] for r in model.rows]
    guidance.STATE.on_cfg_denoiser(types.SimpleNamespace(sampling_step=4, total_sampling_steps=10, text_uncond=neg))
    second = step(patched, 5.0)
    rows_second = [r[:] for r in model.rows]
    ok("AG: the step that crosses the threshold is still plain CFG", close(first, plain) and any(1 in r for r in rows_first),
       f"similarity {sim:.4f}")
    ok("AG: the next step skips the negative pass and is the cond prediction", close(second, c0) and all(1 not in r for r in rows_second),
       rows_second)
    out, applied, rows, _ = run(on("ag", ag__threshold=1.0))
    ok("AG with threshold 1 never turns CFG off", close(out, plain) and guidance.STATE.ag_off_step is None, applied)
    _, applied, _, _ = run(on("ag"), cfg=1.0)
    ok("AG is skipped at CFG 1", "ag" not in applied)

    # ------------------------------------------------------------ TPG
    mid_block = [m for n, m in tpg.transformer_blocks(dm) if n.startswith("middle_block")]
    ok("TPG: 'middle' selects the one middle transformer", tpg.select_blocks(dm, "middle") == mid_block)
    auto = tpg.select_blocks(dm, "auto")
    ok("TPG: 'auto' selects input transformers", len(auto) >= 1 and all(tpg.locate(n)[0] == "input"
       for n, m in tpg.transformer_blocks(dm) if m in auto), len(auto))
    tokens_mid = {}

    def shuffled_middle():
        """Cond pass with the middle transformer's input tokens shuffled (permutation from seed 0)."""
        blk = mid_block[0]
        saved = blk.forward
        g = torch.Generator().manual_seed(0)

        def fwd(x_, context=None, transformer_options={}, **kw):
            perm = torch.randperm(x_.shape[1], generator=g)
            return saved(x_[:, perm], context=context, transformer_options=transformer_options, **kw)
        blk.forward = fwd
        try:
            return host.run_cond(model, conds(pos), x, sigma, base.model_options)
        finally:
            blk.forward = saved

    shuffled = shuffled_middle()
    out, applied, rows, _ = run(on("tpg", tpg__blocks="middle", tpg__scale=3.0))
    good, info = same_effect(out, plain + 3.0 * (c0 - shuffled), plain)
    ok("TPG == CFG + 3 (cond - cond with shuffled tokens in the middle transformer)", good, info)
    ok("TPG: one extra cond-only pass, shuffle only in that pass", tpg.CONTROLLER.calls == 1 and rows.count([0]) >= 1,
       f"shuffles {tpg.CONTROLLER.calls}")
    out, applied, _, _ = run(on("tpg"))
    ok("TPG 'auto' runs and differs", "tpg" in applied and not close(out, plain) and torch.isfinite(out).all().item())
    out, _, rows, _ = run(on("tpg", tpg__blocks="middle"), cfg=1.0)
    good, info = same_effect(out, c0 + 3.0 * (c0 - shuffled), c0)
    ok("TPG at CFG 1, no negative pass", good and all(1 not in r for r in rows), info)

    # ------------------------------------------------------------ SWG
    def swg_reference(window, overlap, scale, base_out, base_cond):
        wl, ol = window // 8, overlap // 8
        weak = C.sliding_window_prediction(x, wl, wl, ol, lambda xw: host.run_cond(model, conds(pos), xw, sigma, base.model_options))
        return base_out + scale * (base_cond - weak)

    out, applied, rows, _ = run(on("swg", swg__window_width=64, swg__window_height=64, swg__overlap=16, swg__scale=2.0))
    good, info = same_effect(out, swg_reference(64, 16, 2.0, plain, c0), plain)
    ok("SWG == CFG + 2 (cond - average of cond on 64 px windows)", good, info)
    windows = len(C.window_starts(x.shape[-2], 8, 6)) * len(C.window_starts(x.shape[-1], 8, 6))
    ok("SWG: one cond pass per window", guidance.STATE.extra_passes["swg"] == windows,
       f"{guidance.STATE.extra_passes['swg']} passes, {windows} windows")
    out, applied, _, _ = run(on("swg", swg__window_width=768, swg__window_height=768))
    ok("SWG with a window larger than the image does nothing", close(out, plain))

    # ------------------------------------------------------------ CHG
    solver = CG.CHG(max_iter=20)

    def chg_reference():
        abt = torch.full((B,), 1.0 / (1.0 + 2.0 ** 2))

        def at(dx):
            return (host.run_cond(model, conds(pos), x + 4.0 * dx, sigma, base.model_options),
                    host.run_cond(model, conds(neg), x + 5.0 * dx, sigma, base.model_options))
        dx = solver.solve(lambda d: d + at(d)[0] - at(d)[1], c0 - u0, 5.0, abt, 1.0 / (1.0 + 14.6146 ** 2))
        c, u = at(dx)
        return u + 5.0 * (c - u), dx

    ref, dx = chg_reference()
    out, applied, rows, _ = run(on("chg", chg__max_iter=20))
    ok("CHG through the WebUI == CHG on the WebUI's predictions", close(out, ref, 1e-3), f"|dx| {dx.abs().mean():.3e}, "
       f"converged {solver.last[1]}, passes {guidance.STATE.extra_passes['chg']}")
    ok("CHG changes the result when it converges", (not any(solver.last[1])) or not close(out, plain))
    out, _, _, _ = run(on("chg", chg__start=0.8))
    ok("CHG outside its steps == plain CFG", close(out, plain))
    _, applied, _, _ = run(on("chg"), cfg=1.0)
    ok("CHG is skipped at CFG 1", "chg" not in applied)
    out, applied, _, _ = run(on("chg", "apg", chg__max_iter=20))
    ok("CHG + APG: runs, finite", torch.isfinite(out).all().item() and "chg" in applied and "apg" in applied)

    # ------------------------------------------------------------ everything at once
    everything = on(*S.MODULES.keys(), swg__window_width=64, swg__window_height=64, swg__overlap=16,
                    ag__threshold=1.0, chg__max_iter=10)
    out, applied, rows, _ = run(everything)
    ok(f"all {len(S.MODULES)} modules together: finite", torch.isfinite(out).all().item(), applied)
    ok(f"all {len(S.MODULES)} modules together: all applied", sorted(applied) == sorted(S.MODULES.keys()))

    # ------------------------------------------------------------ nothing left behind
    guidance.finish()
    ok("the WebUI's own patcher is untouched", base.model_options == base_options)
    ok("after a run, plain CFG is back", close(step(base, 5.0), plain))

    # ------------------------------------------------------------ PNG info
    s = on("pag", "nag", "fdg", pag__scale=4.5, nag__alpha=0.4, fdg__low_share=0.35)
    back = S.from_infotext(S.to_infotext(s))
    ok("PNG info round trip", back == s, S.to_infotext(s))
    s = on("ag", "tpg", "chg", "swg", ag__compare="Noise", tpg__blocks="input 7.2-9, input 8", chg__bases=3,
           chg__log_tol=-3.5, swg__window_width=640)
    back = S.from_infotext(S.to_infotext(s))
    ok("PNG info round trip (AG, TPG, CHG, SWG)", back == s, S.to_infotext(s))

print()
print("ALL PASS" if not fails else f"{len(fails)} FAILURES: " + "; ".join(fails))
sys.exit(1 if fails else 0)
