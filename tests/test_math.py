"""The guidance maths against the reference code of each method.

    python tests/test_math.py          (any Python with torch; no WebUI needed)

Each reference below is copied from the method's own source (see NOTICE.md)
and kept as is; the extension's version must match it to float precision.
"""
import math
import pathlib
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from lib_steer import attn as A  # noqa: E402
from lib_steer import cfg as C  # noqa: E402
from lib_steer import chg as CG  # noqa: E402
from lib_steer import nag as N  # noqa: E402
from lib_steer import tpg as T  # noqa: E402

torch.manual_seed(0)
fails = []


def check(name, a, b, tol=1e-5):
    err = (a.double() - b.double()).abs().max().item()
    ok = err <= tol
    print(("PASS " if ok else "FAIL ") + f"{name}   max err {err:.2e}")
    if not ok:
        fails.append(name)


B, CH, H, W = 2, 4, 16, 12
x = torch.randn(B, CH, H, W) * 3
cond = torch.randn(B, CH, H, W)
uncond = torch.randn(B, CH, H, W)


# --- APG: Sadat et al. 2024, Algorithm 1 ----------------------------------------
class MomentumBuffer:
    def __init__(self, momentum):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def project(v0, v1):
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def adaptive_projected_guidance(pred_cond, pred_uncond, guidance_scale, momentum_buffer=None, eta=1.0, norm_threshold=0.0):
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    diff_parallel, diff_orthogonal = project(diff, pred_cond)
    normalized_update = diff_orthogonal + eta * diff_parallel
    pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
    return pred_guided


for eta, thr, mom in [(0.0, 15.0, -0.5), (1.0, 0.0, 0.0), (0.3, 5.0, -0.75)]:
    ours = C.APG(eta, thr, mom)
    ref_buf = MomentumBuffer(mom) if mom != 0 else None
    for step, sigma in enumerate([10.0, 5.0, 1.0]):
        c, u = torch.randn(B, CH, H, W), torch.randn(B, CH, H, W)
        check(f"APG eta={eta} r={thr} beta={mom} step {step}",
              ours(c, u, 7.5, sigma), adaptive_projected_guidance(c, u, 7.5, ref_buf, eta, thr))
# momentum resets when sigma goes back up (next image, hires pass)
apg = C.APG(0.0, 0.0, -0.5)
apg(cond, uncond, 5.0, 1.0)
check("APG momentum reset on a new run", apg(cond, uncond, 5.0, 10.0),
      adaptive_projected_guidance(cond, uncond, 5.0, MomentumBuffer(-0.5), 0.0, 0.0))


# --- NAG: ChenDarYen/ComfyUI-NAG utils.nag ---------------------------------------
def nag_ref(z_positive, z_negative, scale, tau, alpha):
    m = min(z_positive.shape[0], z_negative.shape[0])
    z_positive = z_positive[-m:]
    z_negative = z_negative[-m:]
    z_guidance = z_positive * scale - z_negative * (scale - 1)
    eps = 1e-6
    norm_positive = (torch.norm(z_positive, p=1, dim=-1, keepdim=True).clamp_min(eps).expand_as(z_positive))
    norm_guidance = (torch.norm(z_guidance, p=1, dim=-1, keepdim=True).clamp_min(eps).expand_as(z_guidance))
    s = norm_guidance / norm_positive
    z_guidance = z_guidance * torch.minimum(s, s.new_full((1,), tau)) / s
    z_guidance = z_guidance * alpha + z_positive * (1 - alpha)
    return z_guidance


zp, zn = torch.randn(2, 64, 320), torch.randn(2, 64, 320)
for sc, tau, al in [(5.0, 2.5, 0.25), (3.0, 1.5, 0.5), (11.0, 5.0, 1.0)]:
    check(f"NAG scale={sc} tau={tau} alpha={al}", N.nag(zp, zn, sc, tau, al), nag_ref(zp, zn, sc, tau, al))


# --- SEG / SAG blur: pamparamm/sd-perturbed-attention guidance_utils.gaussian_blur_2d (MIT) ---
def gaussian_blur_2d_ref(img, kernel_size, sigma):
    height = img.shape[-1]
    kernel_size = min(kernel_size, height - (height % 2 - 1))
    ksize_half = (kernel_size - 1) * 0.5
    x_ = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x_ / sigma).pow(2))
    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)
    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])
    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]
    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])
    return img


img = torch.randn(2, 8, 32, 32)
for sg in (1.0, 2.0, 4.0):
    ks = math.ceil(6 * sg) + 1 - math.ceil(6 * sg) % 2
    check(f"Gaussian blur sigma={sg} (separable == 2-D reference)", A.gaussian_blur_2d(img, ks, sg), gaussian_blur_2d_ref(img, ks, sg))

# SEG with infinite blur: every query is the spatial mean
q = torch.randn(1, 12 * 16, 64)
opts = {"n_heads": 4, "original_shape": [1, 4, 24, 32]}
k = v = torch.randn(1, 12 * 16, 64)
mean_q = q.mean(dim=1, keepdim=True).expand_as(q)
check("SEG infinite blur == attention with the mean query", A.seg_attention(10000.0)(q, k, v, opts), A.attention(mean_q, k, v, 4))

# attention helper == textbook attention
qq, kk, vv = torch.randn(3, 2, 10, 32).unbind(0)
ref = torch.softmax(qq.reshape(2, 10, 4, 8).transpose(1, 2) @ kk.reshape(2, 10, 4, 8).transpose(1, 2).transpose(-1, -2) / math.sqrt(8), -1) \
    @ vv.reshape(2, 10, 4, 8).transpose(1, 2)
check("attention helper", A.attention(qq, kk, vv, 4), ref.transpose(1, 2).reshape(2, 10, 32))
check("attention helper with probabilities", A.attention_with_probs(qq, kk, vv, 4)[0], ref.transpose(1, 2).reshape(2, 10, 32))



# --- FDG: Sadat et al. 2025, Algorithm 2 (with kornia, as the paper) ------------
import kornia  # noqa: E402
from kornia.geometry.transform import build_laplacian_pyramid  # noqa: E402


def project_ref(v0, v1):
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def build_image_from_pyramid(pyramid):
    img = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        img = kornia.geometry.pyrup(img) + pyramid[i]
    return img


def laplacian_guidance(pred_cond, pred_uncond, guidance_scale=[1.0, 1.0], parallel_weights=None):
    levels = len(guidance_scale)
    if parallel_weights == None:  # noqa: E711
        parallel_weights = [1.0] * levels
    pred_cond_pyramid = build_laplacian_pyramid(pred_cond, levels)
    pred_uncond_pyramid = build_laplacian_pyramid(pred_uncond, levels)
    pred_guided_pyramid = []
    parameters = zip(pred_cond_pyramid, pred_uncond_pyramid, guidance_scale, parallel_weights)
    for idx, (p_cond, p_uncond, scale, par_weight) in enumerate(parameters):
        diff = p_cond - p_uncond
        diff_parallel, diff_orthogonal = project_ref(diff, p_cond)
        diff = par_weight * diff_parallel + diff_orthogonal
        p_guided = p_cond + (scale - 1) * diff
        pred_guided_pyramid.append(p_guided)
    pred_guided = build_image_from_pyramid(pred_guided_pyramid)
    return pred_guided.to(pred_cond.dtype)


c32, u32 = torch.randn(2, 4, 32, 32), torch.randn(2, 4, 32, 32)
for scales in ([10.0, 5.0], [7.0, 3.0], [12.0, 1.5], [9.0, 7.0, 5.0, 3.0]):
    check(f"FDG scales {scales} == paper (kornia)", C.fdg(c32, u32, scales), laplacian_guidance(c32, u32, scales), 1e-4)
for eta in (0.0, 0.5):
    band = C.BandAPG(eta, 0.0, 0.0)
    check(f"FDG + APG projection eta={eta} == paper's parallel_weights", C.fdg(c32, u32, [10.0, 5.0], band, 1.0),
          laplacian_guidance(c32, u32, [10.0, 5.0], [eta, eta]), 1e-4)
for shape in ((1, 4, 13, 21), (2, 16, 104, 152), (1, 4, 3, 5)):
    t = torch.randn(shape)
    check(f"pyramid reconstructs exactly at {shape[2]}x{shape[3]}", C.reconstruct(C.laplacian_pyramid(t, 3)), t, 1e-5)
    c_, u_ = torch.randn(shape), torch.randn(shape)
    check(f"FDG with equal scales == plain CFG at {shape[2]}x{shape[3]}", C.fdg(c_, u_, [6.0, 6.0]), u_ + 6.0 * (c_ - u_), 1e-4)
ok_ = C.fdg_scales(10, 5, 2) == [10, 5] and C.fdg_scales(10, 4, 4) == [10, 8, 6, 4]
print(("PASS " if ok_ else "FAIL ") + "FDG scales per band")
if not ok_:
    fails.append("FDG scales per band")


# --- AG: cosine similarity as ComfyUI-Adaptive-Guidance computes it (batch of 1) ----
cos_ref = torch.nn.CosineSimilarity(dim=1)
c1, u1 = torch.randn(1, 4, 16, 16), torch.randn(1, 4, 16, 16)
check("AG similarity == asagi4 cos(cond.reshape(1, -1), uncond.reshape(1, -1))",
      C.similarity(c1, u1), cos_ref(c1.reshape(1, -1), u1.reshape(1, -1)).double(), 1e-6)
check("AG similarity per sample", C.similarity(cond, uncond),
      torch.stack([cos_ref(cond[i].reshape(1, -1), uncond[i].reshape(1, -1))[0] for i in range(B)]).double(), 1e-6)


# --- TPG: pamparamm/sd-perturbed-attention tpg_nodes.shuffle_tokens -----------
def shuffle_tokens(x):
    permutation = torch.randperm(x.shape[1], device=x.device)
    return x[:, permutation]


tokens = torch.randn(2, 96, 64)
g = torch.Generator().manual_seed(7)
torch.manual_seed(7)
ref = shuffle_tokens(tokens)
T.CONTROLLER.generator = g
ours = tokens[:, T.CONTROLLER.permutation(tokens.shape[1], tokens.device)]
check("TPG token shuffle == reference (same permutation source)", ours, ref, 0.0)
ok_ = sorted(ours[0, :, 0].tolist()) == sorted(tokens[0, :, 0].tolist()) and torch.equal(ours[0].argsort(0)[:, :1] * 0, ours[0].argsort(0)[:, :1] * 0)
print(("PASS " if ok_ else "FAIL ") + "TPG shuffle keeps every token")
if not ok_:
    fails.append("TPG shuffle keeps every token")
for spec, want in (("input 7.2-9, input 8", [("input", 7, 2, 9), ("input", 8, None, None)]),
                   ("middle, out 1.0", [("middle", 0, None, None), ("output", 1, 0, 0)])):
    got = A.parse_block_ranges(spec)
    print(("PASS " if got == want else "FAIL ") + f"block ranges {spec!r}")
    if got != want:
        fails.append(f"block ranges {spec!r}")


# --- SWG: every latent pixel is covered, overlaps are averaged ------------------
def swg_naive(x, wh, ww, overlap, predict):
    h, w = x.shape[-2:]
    acc = torch.zeros_like(x)
    cnt = torch.zeros_like(x[:, :1])
    tops = sorted({min(t, h - wh) for t in range(0, h, wh - overlap)})
    lefts = sorted({min(l_, w - ww) for l_ in range(0, w, ww - overlap)})
    for t in tops:
        for l_ in lefts:
            acc[..., t:t + wh, l_:l_ + ww] += predict(x[..., t:t + wh, l_:l_ + ww])
            cnt[..., t:t + wh, l_:l_ + ww] += 1
    return acc / cnt


xs = torch.randn(1, 4, 128, 152)
pred = lambda t: torch.tanh(t) * t.mean()  # noqa: E731  (depends on the whole window)
for wh, ww, ov in ((96, 96, 32), (64, 100, 16), (128, 152, 0)):
    check(f"SWG windows {ww}x{wh} overlap {ov} == independent tiling", C.sliding_window_prediction(xs, wh, ww, ov, pred),
          swg_naive(xs, wh, ww, ov, pred), 1e-6)
starts = C.window_starts(128, 96, 64)
ok_ = starts == [0, 32] and C.window_starts(100, 128, 10) == [0]
print(("PASS " if ok_ else "FAIL ") + f"SWG window offsets {starts}")
if not ok_:
    fails.append("SWG window offsets")


# --- CHG: scraed/CharacteristicGuidanceWebUI CharaIte.py -----------------------
# solve_least_squares, split_basis, proj_least_squares and the body of
# chara_ite_inner_loop, copied as they are; only the model evaluation and the
# `self.` settings are replaced by arguments.
def solve_least_squares(A, B):
    C = torch.matmul(A.transpose(-2, -1), A)
    U, S, Vh = torch.linalg.svd(C.float(), full_matrices=False)
    D_inv = torch.diag_embed(1.0 / torch.maximum(S, torch.ones_like(S) * 1e-4))
    C_inv = Vh.transpose(-1,-2).matmul(D_inv).matmul(U.transpose(-1,-2))
    X = torch.matmul(torch.matmul(C_inv, A.transpose(-2, -1)), B)
    return X


def split_basis(g, n):
    g_flat = g.view(g.shape[0], g.shape[1], -1)
    quantiles = torch.quantile(g_flat, torch.linspace(0, 1, n + 1, device=g.device), dim=-1).permute(1, 2, 0)
    output = torch.zeros(*g.shape, n, device=g.device)
    for i in range(n):
        lower = quantiles[..., i][..., None, None]
        upper = quantiles[..., i + 1][..., None, None]
        if i < n - 1:
            mask = (g >= lower) & (g < upper)
        else:
            mask = (g >= lower) & (g <= upper)
        output[..., i] = g * mask
    output = output.view(*g.shape, n)
    return output


def proj_least_squares(A, B, reg):
    C = torch.matmul(A.transpose(-2, -1), A) + reg * torch.eye(A.shape[-1], device=A.device)
    eigenvalues, eigenvectors = torch.linalg.eigh(C)
    D_inv = torch.diag_embed(1.0 / torch.maximum(eigenvalues, torch.ones_like(eigenvalues) * 1e-4))
    C_inv = torch.matmul(torch.matmul(eigenvectors, D_inv), eigenvectors.transpose(-2, -1))
    B_proj = torch.matmul(A, torch.matmul(torch.matmul(C_inv, A.transpose(-2, -1)), B))
    return B_proj


class CHGRef:
    def __init__(self, reg_ini, reg_range, ite, noise_base, chara_decay, res_thres, lr_chara, reg_size, reg_w, aa_dim):
        self.reg_ini, self.reg_range, self.ite, self.noise_base = reg_ini, reg_range, ite, noise_base
        self.chara_decay, self.res_thres, self.lr_chara = chara_decay, res_thres, lr_chara
        self.reg_size, self.reg_w, self.aa_dim = reg_size, reg_w, aa_dim
        self.dxs_buffer = None
        self.abt_buffer = None

    def loop(self, eps_pair, x_in, cond_scale, abt_current, abt_smallest):
        """eps_pair(x_cond, x_uncond) -> (x - D) for cond and uncond: the reference's eps_evaluation (Forge)."""
        dxs = torch.zeros_like(x_in)
        res_thres = self.res_thres
        num_x_in_cond = 1
        h = cond_scale * num_x_in_cond
        dxs_Anderson = []
        g_Anderson = []

        def AndersonAccR(dxs, g, reg_level, reg_target, pre_condition=None, m=3):
            batch = dxs.shape[0]
            x_shape = dxs.shape[1:]
            reg_residual_form = reg_level
            g_flat = g.reshape(batch, -1)
            dxs_flat = dxs.reshape(batch, -1)
            res_g = self.reg_size * (reg_residual_form[:, None] - reg_target[:, None])
            res_dxs = reg_residual_form[:, None]
            g_Anderson.append(torch.cat((g_flat, res_g), dim=-1))
            dxs_Anderson.append(torch.cat((dxs_flat, res_dxs), dim=-1))
            if len(g_Anderson) < 2:
                return dxs, g, res_dxs[:, 0], res_g[:, 0]
            else:
                g_Anderson[-2] = g_Anderson[-1] - g_Anderson[-2]
                dxs_Anderson[-2] = dxs_Anderson[-1] - dxs_Anderson[-2]
                if len(g_Anderson) > m:
                    del dxs_Anderson[0]
                    del g_Anderson[0]
                gA = torch.cat([g[..., None] for g in g_Anderson[:-1]], dim=-1)
                gB = g_Anderson[-1][..., None]
                gA_norm = torch.maximum(torch.sum(gA ** 2, dim=-2, keepdim=True) ** 0.5, torch.ones_like(gA) * 1e-4)
                gamma = torch.linalg.lstsq(gA / gA_norm, gB).solution
                if torch.sum( torch.isnan(gamma) ) > 0:
                    gamma = solve_least_squares(gA/gA_norm, gB)
                xA = torch.cat([x[..., None] for x in dxs_Anderson[:-1]], dim=-1)
                xB = dxs_Anderson[-1][..., None]
                xO = xB - (xA / gA_norm).matmul(gamma)
                gO = gB - (gA / gA_norm).matmul(gamma)
                dxsO = xO[:, :-1].reshape(batch, *x_shape)
                dgO = gO[:, :-1].reshape(batch, *x_shape)
                resxO = xO[:, -1, 0]
                resgO = gO[:, -1, 0]
                return dxsO, dgO, resxO, resgO

        def downsample_reg_g(dx, g_1, reg):
            if g_1 is None:
                return dx
            elif self.noise_base >= 1:
                A = g_1.reshape(g_1.shape[0] * g_1.shape[1], g_1.shape[2] * g_1.shape[3], g_1.shape[4])
                B = dx.reshape(dx.shape[0] * dx.shape[1], -1, 1)
                regl = reg[:, None].expand(-1, dx.shape[1]).reshape(dx.shape[0] * dx.shape[1], 1, 1)
                dx_proj = proj_least_squares(A, B, regl)
                return dx_proj.reshape(*dx.shape)
            else:
                A = g_1.reshape(g_1.shape[0], g_1.shape[1]* g_1.shape[2] * g_1.shape[3], g_1.shape[4])
                B = dx.reshape(dx.shape[0], -1, 1)
                regl = reg[:, None].reshape(dx.shape[0], 1, 1)
                dx_proj = proj_least_squares(A, B, regl)
                return dx_proj.reshape(*dx.shape)
        g_1 = None

        reg_level = torch.zeros(dxs.shape[0], device=dxs.device) + max(5,self.reg_ini)
        reg_target_level = self.reg_ini * (abt_smallest / abt_current[:, 0, 0, 0]) ** (1 / self.reg_range)
        best_res_el = torch.mean(dxs, dim=(-1, -2, -3), keepdim=True) + 100
        best_dxs = torch.zeros_like(dxs)
        n_iterations = self.ite

        if self.dxs_buffer is not None:
            abt_prev = self.abt_buffer
            dxs = self.dxs_buffer
            dxs = dxs * ((abt_prev - abt_current * abt_prev) / (abt_current - abt_current * abt_prev))
            dxs = self.chara_decay * dxs

        iteration_counts = 0
        for iteration in range(n_iterations):
            def compute_correction_direction(dxs):
                eps_u, eps_c = eps_pair(x_in + (h - 1) * dxs, x_in + h * dxs)
                ggg = (eps_u - eps_c)
                return ggg

            ggg = compute_correction_direction(dxs)
            g = dxs - downsample_reg_g(ggg, g_1, reg_level)
            if g_1 is None:
                g_basis = -compute_correction_direction(dxs*0)
                g_1 = split_basis(g_basis, max( self.noise_base,1 ) )
                if self.noise_base >=1:
                    g_1_norm = torch.sum(g_1 ** 2, dim=(-2, -3), keepdim=True) ** 0.5
                    g_1 = g_1 / torch.maximum(g_1_norm, torch.ones_like(
                        g_1_norm) * 1e-4)
                else:
                    g_1_norm = torch.sum(g_1 ** 2, dim=(-2, -3, -4), keepdim=True) ** 0.5
                    g_1 = g_1 / torch.maximum(g_1_norm, torch.ones_like(
                        g_1_norm) * 1e-4)
            reg_Acc = (reg_level * self.reg_w) ** 0.5
            reg_target = (reg_target_level * self.reg_w) ** 0.5
            g_flat_res = g.reshape(dxs.shape[0], -1)
            reg_g = self.reg_size * (reg_Acc[:, None] - reg_target[:, None])
            g_flat_res_reg = torch.cat((g_flat_res, reg_g), dim=-1)
            res_el = ((torch.mean((g_flat_res_reg) ** 2, dim=(-1), keepdim=False)) ** 0.5)[:, None, None, None]
            if iteration == 0:
                best_res_el = res_el
                best_dxs = dxs
                not_converged = torch.ones_like(res_el).bool()
            res_mask = torch.logical_and(res_el < best_res_el, not_converged).int()
            best_res_el = res_mask * res_el + (1 - res_mask) * best_res_el
            best_dxs = res_mask * dxs + (1 - res_mask) * best_dxs
            res_max = torch.max(best_res_el)
            not_converged = torch.logical_and(res_el >= res_thres, not_converged)
            if res_max < res_thres:
                break
            if self.noise_base >=1:
                aa_dim = self.aa_dim
            else:
                aa_dim = 1
            dxs_Acc, g_Acc, reg_dxs_Acc, reg_g_Acc = AndersonAccR(dxs, g, reg_Acc, reg_target, pre_condition=None,
                                                                m=aa_dim + 1)
            dxs = dxs_Acc - self.lr_chara * g_Acc
            reg_Acc = reg_dxs_Acc - self.lr_chara * reg_g_Acc
            reg_level = reg_Acc ** 2 / self.reg_w
            iteration_counts = iteration_counts * (1 - not_converged.long()) + iteration * not_converged.long()
        final_dxs = best_dxs * (1 - not_converged.long())
        self.dxs_buffer = final_dxs
        self.abt_buffer = abt_current
        return final_dxs


torch.manual_seed(3)
xin = torch.randn(2, 4, 12, 10)
kc, ku = torch.randn(4, 4, 3, 3) * 0.15, torch.randn(4, 4, 3, 3) * 0.15


def D(x_, k_):
    """A small non-linear 'denoiser'."""
    return 0.6 * x_ + 0.4 * torch.tanh(F.conv2d(x_, k_, padding=1))


def eps_pair(x_c, x_u):        # the reference's (x - D) for uncond and cond, as in its Forge branch
    return x_u - D(x_u, ku), x_c - D(x_c, kc)


for label, kw in (("defaults", dict()), ("3 bases", dict(bases=3, aa_memory=3)),
                  ("strong regularisation", dict(reg_strength=4.0, reg_range=2.0)),
                  ("slider above 5", dict(reg_strength=6.5))):
    params = dict(reg_strength=1.0, reg_range=1.0, max_iter=30, bases=0, reuse=1.0, log_tol=-4.0, step_size=1.0,
                  anneal_speed=0.4, anneal_strength=0.5, aa_memory=2)
    params.update(kw)
    ours = CG.CHG(**params)
    ref = CHGRef(CG.regularisation_scale(params["reg_strength"]), CG.regularisation_scale(params["reg_range"]),
                 params["max_iter"], params["bases"], params["reuse"], 10 ** params["log_tol"], params["step_size"],
                 params["anneal_speed"], params["anneal_strength"], params["aa_memory"])
    s_ = 7.0
    for step_, sig in enumerate((6.0, 3.0)):          # two steps: the second starts from the first's correction
        abt = torch.full((2, 1, 1, 1), 1.0 / (1.0 + sig ** 2))
        abt_min = 1.0 / (1.0 + 14.6146 ** 2)
        want = ref.loop(eps_pair, xin, s_, abt, abt_min)

        def evaluate(dx):
            return dx + D(xin + (s_ - 1) * dx, kc) - D(xin + s_ * dx, ku)
        got = ours.solve(evaluate, D(xin, kc) - D(xin, ku), s_, abt.reshape(-1), abt_min)
        check(f"CHG correction == reference ({label}, step {step_ + 1})", got, want, 1e-4)
print(f"      CHG iterations / converged (last case): {ours.last}")
ok_ = abs(CG.regularisation_scale(6.5) - (math.exp(0.8898 * 1.5) / 0.8898 + 5 - 1 / 0.8898)) < 1e-9
print(("PASS " if ok_ else "FAIL ") + "CHG slider mapping above 5")
if not ok_:
    fails.append("CHG slider mapping")

print()
print("ALL PASS" if not fails else f"{len(fails)} FAILURES: " + "; ".join(fails))
sys.exit(1 if fails else 0)
