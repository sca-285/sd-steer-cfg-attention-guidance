"""How the positive and negative predictions are combined: plain CFG, APG,
FDG, or FDG with APG's projection.

Pure torch, no WebUI imports. Inputs and outputs are denoised (x0)
predictions of shape (batch, channels, h, w) - the space both papers work in.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-8


def _dims(t: torch.Tensor):
    return tuple(range(1, t.ndim))


def project(v0: torch.Tensor, v1: torch.Tensor):
    """Parts of v0 parallel and orthogonal to v1, per sample, in float64 (both papers' `project`)."""
    dtype = v0.dtype
    a, b = v0.double(), v1.double()
    b = b / b.flatten(1).norm(dim=1).clamp_min(EPS).reshape(-1, *([1] * (b.ndim - 1)))
    parallel = (a * b).sum(dim=_dims(a), keepdim=True) * b
    return parallel.to(dtype), (a - parallel).to(dtype)


# --------------------------------------------------------------------- APG
# Sadat et al. 2024, "Eliminating Oversaturation and Artifacts of High Guidance
# Scales in Diffusion Models" (arXiv:2410.02416), Algorithm 1.

class APG:
    """Adaptive Projected Guidance with its momentum buffer (one per sampling pass)."""

    def __init__(self, eta=0.0, norm_threshold=15.0, momentum=-0.5):
        self.eta = float(eta)
        self.norm_threshold = float(norm_threshold)
        self.momentum = float(momentum)
        self.running = None
        self.prev_sigma = None

    def update(self, diff: torch.Tensor, sigma: float) -> torch.Tensor:
        """Momentum and norm threshold applied to cond - uncond."""
        if self.prev_sigma is not None and sigma > self.prev_sigma + 1e-6:
            self.running = None           # a new pass starts at a higher sigma
        self.prev_sigma = sigma
        if self.momentum != 0.0:
            if self.running is None or self.running.shape != diff.shape:
                self.running = diff
            else:
                self.running = diff + self.momentum * self.running
            diff = self.running
        if self.norm_threshold > 0:
            norm = diff.float().norm(p=2, dim=_dims(diff), keepdim=True)
            diff = diff * torch.minimum(torch.ones_like(norm), self.norm_threshold / norm.clamp_min(EPS)).to(diff.dtype)
        return diff

    def __call__(self, cond, uncond, scale, sigma, weight=1.0):
        diff = self.update((cond - uncond) * weight, sigma)
        parallel, orthogonal = project(diff, cond)
        return cond + (scale - 1.0) * (orthogonal + self.eta * parallel)


# --------------------------------------------------------------------- FDG
# Sadat et al. 2025, "Guidance in the Frequency Domain Enables High-Fidelity
# Sampling at Low CFG Scales" (arXiv:2506.19713), Algorithm 2.
#
# The paper builds a Laplacian pyramid with kornia and guides each band with
# its own scale. The pyramid here uses the same 5x5 binomial filter and
# bilinear resampling as kornia's pyrdown / pyrup, but upsamples each level to
# the exact size of the one above instead of padding the latent to a power of
# two, so any latent size decomposes and reconstructs exactly (kornia's
# reference returns the padded size for, say, a 832x1216 image).

_BINOMIAL = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


def _blur(x: torch.Tensor) -> torch.Tensor:
    c = x.shape[1]
    k = _BINOMIAL.to(device=x.device, dtype=x.dtype)
    mode = "reflect" if min(x.shape[-2:]) > 2 else "replicate"
    x = F.pad(x, [2, 2, 0, 0], mode=mode)
    x = F.conv2d(x, k.view(1, 1, 1, 5).expand(c, 1, 1, 5), groups=c)
    x = F.pad(x, [0, 0, 2, 2], mode=mode)
    return F.conv2d(x, k.view(1, 1, 5, 1).expand(c, 1, 5, 1), groups=c)


def pyr_down(x: torch.Tensor) -> torch.Tensor:
    h, w = x.shape[-2:]
    return F.interpolate(_blur(x), size=(max(1, h // 2), max(1, w // 2)), mode="bilinear", align_corners=False)


def pyr_up(x: torch.Tensor, size) -> torch.Tensor:
    return _blur(F.interpolate(x, size=tuple(size), mode="bilinear", align_corners=False))


def laplacian_pyramid(x: torch.Tensor, levels: int):
    """[band 0 (finest detail), ..., band levels-2, low-pass residual]."""
    bands, current = [], x
    for _ in range(levels - 1):
        down = pyr_down(current)
        bands.append(current - pyr_up(down, current.shape[-2:]))
        current = down
    bands.append(current)
    return bands


def reconstruct(bands):
    image = bands[-1]
    for band in reversed(bands[:-1]):
        image = pyr_up(image, band.shape[-2:]) + band
    return image


def fdg_scales(high: float, low: float, levels: int):
    """Scale per band from the finest (high) to the residual (low), linear in between."""
    if levels <= 1:
        return [high]
    return [high + (low - high) * i / (levels - 1) for i in range(levels)]


def fdg(cond, uncond, scales, apg: APG | None = None, sigma: float = 0.0, weight=1.0):
    """Frequency-decoupled guidance.

    Each band is guided as  cond + (w - 1) * diff. With `apg`, diff is the
    APG update of that band (momentum, norm threshold and eta projection per
    band) - the combination the FDG paper describes; without it the full
    difference is used (the paper's parallel weight 1).
    """
    dtype = cond.dtype
    c_bands = laplacian_pyramid(cond.float(), len(scales))
    u_bands = laplacian_pyramid(uncond.float(), len(scales))
    guided = []
    for i, (c, u, w) in enumerate(zip(c_bands, u_bands, scales)):
        diff = (c - u) * weight
        if apg is not None:
            diff = apg.band(i).update(diff, sigma)
            parallel, orthogonal = project(diff, c)
            diff = orthogonal + apg.eta * parallel
        guided.append(c + (w - 1.0) * diff)
    return reconstruct(guided).to(dtype)


class BandAPG:
    """APG settings with one momentum buffer per FDG band."""

    def __init__(self, eta=0.0, norm_threshold=15.0, momentum=-0.5):
        self.eta, self.norm_threshold, self.momentum = float(eta), float(norm_threshold), float(momentum)
        self._bands = {}

    def band(self, i) -> APG:
        if i not in self._bands:
            self._bands[i] = APG(self.eta, self.norm_threshold, self.momentum)
        return self._bands[i]


# --------------------------------------------------------------------- AG
# Castillo et al. 2023, "Adaptive Guidance: Training-free Acceleration of
# Conditional Diffusion Models" (arXiv:2312.12487). Once the positive and
# negative predictions agree (cosine similarity above a threshold), CFG is
# turned off for the rest of the pass, which skips the negative pass. The
# paper compares the noise predictions; ComfyUI-Adaptive-Guidance (asagi4)
# compares the denoised ones, whose similarity rises more gradually. Both
# are offered ("AG compare").

def similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity per sample, (batch,), in float64."""
    a, b = a.double().flatten(1), b.double().flatten(1)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(EPS)


# --------------------------------------------------------------------- SWG
# Kaiser et al. 2024, "The Unreasonable Effectiveness of Guidance for
# Diffusion Models" (arXiv:2411.10257): the weak prediction is the same model
# run on overlapping windows smaller than the image, averaged where they
# overlap. The windows step by (window - overlap) and the last one is moved
# back to the image edge, so every window has the chosen size.

def window_starts(length: int, window: int, stride: int):
    """Window offsets that cover [0, length); the last window ends at the edge."""
    if window >= length:
        return [0]
    stride = max(1, stride)
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def sliding_window_prediction(x: torch.Tensor, window_h: int, window_w: int, overlap: int, predict):
    """Average of predict(x[..., window]) over overlapping windows.

    predict(x_window) returns the denoised prediction for that window. The
    windows are window_h x window_w (clamped to the image) and overlap by
    `overlap` (all in latent pixels).
    """
    h, w = x.shape[-2:]
    wh, ww = min(window_h, h), min(window_w, w)
    total = torch.zeros_like(x)
    count = torch.zeros_like(x[:, :1])
    for top in window_starts(h, wh, wh - overlap):
        for left in window_starts(w, ww, ww - overlap):
            rows, cols = slice(top, top + wh), slice(left, left + ww)
            total[..., rows, cols] += predict(x[..., rows, cols])
            count[..., rows, cols] += 1
    return total / count
