"""CHG, Characteristic Guidance (Zheng & Lan, ICML 2024, arXiv:2312.07586;
https://github.com/scraed/CharacteristicGuidanceWebUI, GPL-3.0).

CFG evaluates the positive and the negative prediction at the same x. CHG
evaluates them at shifted points,

    positive at x + (s - 1) dx,    negative at x + s dx        (s = CFG scale)

and combines them as CFG does. The shift dx solves the fixed point

    dx = P( dx + D+(x + (s - 1) dx) - D-(x + s dx) )

where D+ / D- are the denoised predictions and P is a regularised
least-squares projection onto a few basis images. The regularisation starts
strong and is annealed towards a target that falls over the sampling steps;
the iteration uses Anderson acceleration. Samples that do not converge fall
back to plain CFG for that step.

This follows the reference implementation's iteration (CharaIte.py,
`chara_ite_inner_loop`) step for step, written in the denoised-prediction
space, where D(x) = x - sigma * eps(x) holds for eps- and v-prediction
models alike.

Pure torch, no WebUI imports: the model is reached through `evaluate`.
"""

from __future__ import annotations

import math

import torch


def split_basis(g: torch.Tensor, n: int) -> torch.Tensor:
    """(b, c, h, w) -> (b, c, h, w, n): g split by its per-channel quantiles."""
    flat = g.reshape(g.shape[0], g.shape[1], -1)
    quantiles = torch.quantile(flat.float(), torch.linspace(0, 1, n + 1, device=g.device), dim=-1).permute(1, 2, 0)
    out = torch.zeros(*g.shape, n, device=g.device, dtype=g.dtype)
    for i in range(n):
        lower = quantiles[..., i][..., None, None].to(g.dtype)
        upper = quantiles[..., i + 1][..., None, None].to(g.dtype)
        mask = (g >= lower) & ((g < upper) if i < n - 1 else (g <= upper))
        out[..., i] = g * mask
    return out


def proj_least_squares(a: torch.Tensor, b: torch.Tensor, reg: torch.Tensor) -> torch.Tensor:
    """A (A^T A + reg I)^+ A^T B, batched; small eigenvalues floored at 1e-4."""
    c = a.transpose(-2, -1) @ a + reg * torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
    values, vectors = torch.linalg.eigh(c)
    inv = torch.diag_embed(1.0 / torch.maximum(values, torch.ones_like(values) * 1e-4))
    c_inv = vectors @ inv @ vectors.transpose(-2, -1)
    return a @ (c_inv @ (a.transpose(-2, -1) @ b))


def _solve_least_squares(a, b):
    c = a.transpose(-2, -1) @ a
    u, s, vh = torch.linalg.svd(c.float(), full_matrices=False)
    d_inv = torch.diag_embed(1.0 / torch.maximum(s, torch.ones_like(s) * 1e-4))
    c_inv = vh.transpose(-1, -2) @ d_inv @ u.transpose(-1, -2)
    return c_inv.to(a.dtype) @ (a.transpose(-2, -1) @ b)


def regularisation_scale(value: float) -> float:
    """The reference's slider mapping: linear up to 5, exponential above."""
    if value <= 5:
        return float(value)
    k = 0.8898
    return math.exp(k * (value - 5)) / k + 5 - 1 / k


class CHG:
    """The iteration's settings and the correction kept from the previous step."""

    def __init__(self, reg_strength=1.0, reg_range=1.0, max_iter=50, bases=0, reuse=1.0, log_tol=-4.0,
                 step_size=1.0, anneal_speed=0.4, anneal_strength=0.5, aa_memory=2):
        self.reg_ini = regularisation_scale(reg_strength)
        self.reg_range = regularisation_scale(reg_range)
        self.max_iter = max(1, int(max_iter))
        self.bases = max(0, int(bases))
        self.reuse = float(reuse)
        self.tol = 10 ** float(log_tol)
        self.lr = float(step_size)
        self.reg_size = float(anneal_speed)     # the reference's reg_size ("annealing speed")
        self.reg_w = float(anneal_strength)     # the reference's reg_w ("annealing strength")
        self.aa_dim = max(1, int(aa_memory))
        self.buffer = None                      # (dx, alpha_bar) of the previous step
        self.last = None                        # (iterations, converged) of the last solve, per sample

    def reset(self):
        self.buffer = None

    def solve(self, evaluate, g0, scale, abt_current, abt_smallest):
        """The shift dx for this step.

        evaluate(dx) -> dx + D+(x + (s-1) dx) - D-(x + s dx)   (the reference's `ggg`)
        g0          -> that quantity at dx = 0, i.e. D+(x) - D-(x), already known
        abt_current -> (b,) alpha-bar of this step; abt_smallest: of the noisiest step
        """
        b = g0.shape[0]
        device, dtype = g0.device, g0.dtype
        abt_current = abt_current.to(device=device, dtype=dtype).reshape(b, 1, 1, 1)

        dxs = torch.zeros_like(g0)
        if self.buffer is not None and self.buffer[0].shape == g0.shape:
            prev, abt_prev = self.buffer
            dxs = prev * ((abt_prev - abt_current * abt_prev) / (abt_current - abt_current * abt_prev))
            dxs = self.reuse * dxs

        dxs_hist, g_hist = [], []

        def anderson(dxs, g, reg_level, reg_target, m):
            batch = dxs.shape[0]
            shape = dxs.shape[1:]
            res_g = self.reg_size * (reg_level[:, None] - reg_target[:, None])
            res_dxs = reg_level[:, None]
            g_hist.append(torch.cat((g.reshape(batch, -1), res_g), dim=-1))
            dxs_hist.append(torch.cat((dxs.reshape(batch, -1), res_dxs), dim=-1))
            if len(g_hist) < 2:
                return dxs, g, res_dxs[:, 0], res_g[:, 0]
            g_hist[-2] = g_hist[-1] - g_hist[-2]
            dxs_hist[-2] = dxs_hist[-1] - dxs_hist[-2]
            if len(g_hist) > m:
                del dxs_hist[0]
                del g_hist[0]
            g_a = torch.cat([h[..., None] for h in g_hist[:-1]], dim=-1)
            g_b = g_hist[-1][..., None]
            g_a_norm = torch.maximum(torch.sum(g_a ** 2, dim=-2, keepdim=True) ** 0.5, torch.ones_like(g_a) * 1e-4)
            gamma = torch.linalg.lstsq((g_a / g_a_norm).float(), g_b.float()).solution.to(dtype)
            if torch.isnan(gamma).any():
                gamma = _solve_least_squares(g_a / g_a_norm, g_b)
            x_a = torch.cat([h[..., None] for h in dxs_hist[:-1]], dim=-1)
            x_b = dxs_hist[-1][..., None]
            x_o = x_b - (x_a / g_a_norm) @ gamma
            g_o = g_b - (g_a / g_a_norm) @ gamma
            return (x_o[:, :-1].reshape(batch, *shape), g_o[:, :-1].reshape(batch, *shape),
                    x_o[:, -1, 0], g_o[:, -1, 0])

        def project(dx, basis, reg):
            if basis is None:
                return dx
            if self.bases >= 1:
                a = basis.reshape(basis.shape[0] * basis.shape[1], basis.shape[2] * basis.shape[3], basis.shape[4])
                bb = dx.reshape(dx.shape[0] * dx.shape[1], -1, 1)
                r = reg[:, None].expand(-1, dx.shape[1]).reshape(dx.shape[0] * dx.shape[1], 1, 1)
            else:
                a = basis.reshape(basis.shape[0], basis.shape[1] * basis.shape[2] * basis.shape[3], basis.shape[4])
                bb = dx.reshape(dx.shape[0], -1, 1)
                r = reg[:, None].reshape(dx.shape[0], 1, 1)
            return proj_least_squares(a, bb, r).reshape(*dx.shape)

        basis = None
        reg_level = torch.zeros(b, device=device, dtype=dtype) + max(5, self.reg_ini)
        reg_target_level = self.reg_ini * (abt_smallest / abt_current[:, 0, 0, 0]) ** (1 / self.reg_range)
        best_res_el = best_dxs = not_converged = None
        iteration_counts = torch.zeros(b, 1, 1, 1, device=device, dtype=torch.long)
        m = (self.aa_dim if self.bases >= 1 else 1) + 1

        for iteration in range(self.max_iter):
            ggg = g0 if (iteration == 0 and not dxs.abs().any()) else evaluate(dxs)
            g = dxs - project(ggg, basis, reg_level)
            if basis is None:
                basis = split_basis(-g0, max(self.bases, 1))
                dims = (-2, -3) if self.bases >= 1 else (-2, -3, -4)
                norm = torch.sum(basis ** 2, dim=dims, keepdim=True) ** 0.5
                basis = basis / torch.maximum(norm, torch.ones_like(norm) * 1e-4)
            reg_acc = (reg_level * self.reg_w) ** 0.5
            reg_target = (reg_target_level * self.reg_w) ** 0.5
            g_flat = g.reshape(b, -1)
            reg_g = self.reg_size * (reg_acc[:, None] - reg_target[:, None])
            res_el = (torch.mean(torch.cat((g_flat, reg_g), dim=-1) ** 2, dim=-1) ** 0.5)[:, None, None, None]
            if iteration == 0:
                best_res_el = res_el
                best_dxs = dxs
                not_converged = torch.ones_like(res_el).bool()
            better = torch.logical_and(res_el < best_res_el, not_converged).to(dtype)
            best_res_el = better * res_el + (1 - better) * best_res_el
            best_dxs = better * dxs + (1 - better) * best_dxs
            not_converged = torch.logical_and(res_el >= self.tol, not_converged)
            if torch.max(best_res_el) < self.tol:
                break
            dxs_acc, g_acc, reg_dxs_acc, reg_g_acc = anderson(dxs, g, reg_acc, reg_target, m)
            dxs = dxs_acc - self.lr * g_acc
            reg_acc = reg_dxs_acc - self.lr * reg_g_acc
            reg_level = reg_acc ** 2 / self.reg_w
            iteration_counts = iteration_counts * (1 - not_converged.long()) + iteration * not_converged.long()

        final = best_dxs * (1 - not_converged.to(dtype))
        self.buffer = (final, abt_current)
        self.last = (iteration_counts.reshape(-1).tolist(), (~not_converged).reshape(-1).tolist())
        return final
