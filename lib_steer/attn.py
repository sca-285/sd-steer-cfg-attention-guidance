"""The attention-level modules: PAG, SEG, SAG (self-attention, extra model
pass) and the attention helpers they share.

Attention is computed here with torch's scaled_dot_product_attention, the
same maths as every WebUI's attention back end, so these functions do not
depend on which one the WebUI picked.
"""

from __future__ import annotations

import math
import re

import torch
import torch.nn.functional as F


def attention(q, k, v, heads, mask=None):
    """(batch, tokens, heads * dim) in and out, like the WebUIs' attention_function."""
    b, _, inner = q.shape
    d = inner // heads
    q, k, v = (t.reshape(b, -1, heads, d).transpose(1, 2) for t in (q, k, v))
    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return out.transpose(1, 2).reshape(b, -1, heads * d)


def attention_with_probs(q, k, v, heads):
    """Attention and its probability map (batch * heads, q tokens, k tokens), in float32."""
    b, _, inner = q.shape
    d = inner // heads
    q, k, v = (t.reshape(b, -1, heads, d).permute(0, 2, 1, 3).reshape(b * heads, -1, d) for t in (q, k, v))
    sim = torch.einsum("b i d, b j d -> b i j", q.float(), k.float()) * d ** -0.5
    sim = sim.softmax(dim=-1)
    out = torch.einsum("b i j, b j d -> b i d", sim.to(v.dtype), v)
    out = out.reshape(b, heads, -1, d).permute(0, 2, 1, 3).reshape(b, -1, heads * d)
    return out, sim


def gaussian_blur_2d(img, kernel_size, sigma):
    """Separable Gaussian blur of (batch, channels, h, w) with reflect padding."""
    # as the SEG reference: never wider than the map (odd size)
    side = min(img.shape[-2:])
    kernel_size = min(kernel_size, side - (side % 2 - 1))
    if kernel_size <= 1:
        return img
    half = (kernel_size - 1) * 0.5
    x = torch.linspace(-half, half, steps=kernel_size, device=img.device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = (kernel / kernel.sum()).to(img.dtype)
    c = img.shape[-3]
    pad = kernel_size // 2
    # reflect padding needs pad < size: fall back to replicate on tiny maps
    mode = "reflect" if pad < min(img.shape[-2:]) else "replicate"
    img = F.pad(img, [pad, pad, 0, 0], mode=mode)
    img = F.conv2d(img, kernel.view(1, 1, 1, -1).expand(c, 1, 1, kernel_size), groups=c)
    img = F.pad(img, [0, 0, pad, pad], mode=mode)
    img = F.conv2d(img, kernel.view(1, 1, -1, 1).expand(c, 1, kernel_size, 1), groups=c)
    return img


# --------------------------------------------------------------------- blocks

_BLOCK = re.compile(r"^(input|in|down|middle|mid|output|out|up)\s*(\d+)?(?:\.(\d+))?$")
_ALIAS = {"in": "input", "down": "input", "mid": "middle", "out": "output", "up": "output"}


def parse_blocks(text: str):
    """'middle, output 1, input 7.0' -> [('middle', 0, None), ('output', 1, None), ('input', 7, 0)].

    The number is the UNet block index (input_blocks / output_blocks), the
    optional .n picks one transformer inside it; without it every
    transformer in the block is patched.
    """
    blocks = []
    for part in (text or "middle").split(","):
        part = part.strip().lower()
        if not part:
            continue
        m = _BLOCK.match(part)
        if not m:
            raise ValueError(f"not a UNet block: {part!r} (use e.g. 'middle', 'output 1', 'input 7.0')")
        name = _ALIAS.get(m.group(1), m.group(1))
        number = int(m.group(2)) if m.group(2) is not None else 0
        if name != "middle" and m.group(2) is None:
            raise ValueError(f"{part!r}: give the block number, e.g. '{name} 1'")
        index = int(m.group(3)) if m.group(3) is not None else None
        blocks.append((name, number, index))
    return blocks or [("middle", 0, None)]


_RANGE = re.compile(r"^(input|in|down|middle|mid|output|out|up)\s*(\d+)?(?:\.(\d+)(?:\s*-\s*(\d+))?)?$")


def parse_block_ranges(text: str):
    """Like parse_blocks, with transformer ranges: 'input 7.2-9, input 8, middle'
    -> [('input', 7, 2, 9), ('input', 8, None, None), ('middle', 0, None, None)]."""
    blocks = []
    for part in str(text or "").split(","):
        part = part.strip().lower()
        if not part:
            continue
        m = _RANGE.match(part)
        if not m:
            raise ValueError(f"not a UNet block: {part!r} (use e.g. 'middle', 'output 1', 'input 7.2-9')")
        name = _ALIAS.get(m.group(1), m.group(1))
        if name != "middle" and m.group(2) is None:
            raise ValueError(f"{part!r}: give the block number, e.g. '{name} 1'")
        number = int(m.group(2)) if m.group(2) is not None else 0
        lo = int(m.group(3)) if m.group(3) is not None else None
        hi = int(m.group(4)) if m.group(4) is not None else lo
        blocks.append((name, number, lo, hi))
    if not blocks:
        raise ValueError("no UNet blocks given")
    return blocks


# --------------------------------------------------------------------- PAG
# Ahn et al. 2024, "Self-Rectifying Diffusion Sampling with Perturbed-Attention
# Guidance". The self-attention map is replaced by the identity.

def pag_attention(q, k, v, extra_options, mask=None):
    return v


# --------------------------------------------------------------------- SEG
# Hong 2024, "Smoothed Energy Guidance: Guiding Diffusion Models with Reduced
# Energy Curvature of Attention". The self-attention query is blurred.

def _token_grid(extra_options, tokens):
    """(h, w) of the token grid at this block, from the latent's aspect ratio."""
    shape = extra_options.get("original_shape") or [1, 1, 1, 1]
    h0, w0 = int(shape[-2]), int(shape[-1])
    ratio = w0 / max(h0, 1)
    if ratio >= 1.0:
        h = max(1, round((tokens / ratio) ** 0.5))
        return h, tokens // h
    w = max(1, round((tokens * ratio) ** 0.5))
    return tokens // w, w


def seg_attention(blur_sigma: float):
    """SEG's self-attention: blur the query spatially, then attend.

    blur_sigma >= 9999 (or < 0) means infinite blur: the query is replaced
    by its spatial mean, as in the paper's default.
    """
    infinite = blur_sigma < 0 or blur_sigma >= 9999

    def fn(q, k, v, extra_options, mask=None):
        heads = extra_options["n_heads"]
        b, tokens, inner = q.shape
        h, w = _token_grid(extra_options, tokens)
        if h * w != tokens:          # odd grid (e.g. region prompts): fall back to the mean
            q = q.mean(dim=1, keepdim=True).expand_as(q)
        else:
            grid = q.permute(0, 2, 1).reshape(b, inner, h, w)
            if infinite:
                grid = grid.mean(dim=(-2, -1), keepdim=True).expand_as(grid)
            else:
                size = math.ceil(6 * blur_sigma) + 1 - math.ceil(6 * blur_sigma) % 2
                grid = gaussian_blur_2d(grid, size, blur_sigma)
            q = grid.reshape(b, inner, tokens).permute(0, 2, 1)
        return attention(q, k, v, heads, mask)

    return fn


# --------------------------------------------------------------------- SAG
# Hong et al. 2023, "Improving Sample Quality of Diffusion Models Using
# Self-Attention Guidance", as ComfyUI implements it: the uncond pass's
# middle-block attention map marks where to blur the prediction.

class SAGRecorder:
    """attn1 replacement for the middle block that keeps the uncond attention map."""

    def __init__(self):
        self.scores = None

    def __call__(self, q, k, v, extra_options, mask=None):
        heads = extra_options["n_heads"]
        cond_or_uncond = extra_options.get("cond_or_uncond") or []
        if 1 in cond_or_uncond:
            b = q.shape[0] // len(cond_or_uncond)
            out, sim = attention_with_probs(q, k, v, heads)
            i = cond_or_uncond.index(1)
            n = heads * b
            self.scores = sim[n * i:n * (i + 1)]
            return out
        return attention(q, k, v, heads, mask)


def sag_blur_map(x0, attn, sigma=2.0, threshold=1.0):
    """Blur x0 where the attention map says the model is looking at."""
    _, hw1, hw2 = attn.shape
    b, _, lh, lw = x0.shape
    attn = attn.reshape(b, -1, hw1, hw2)
    mask = attn.mean(1, keepdim=False).sum(1, keepdim=False) > threshold
    ratio = math.sqrt(lh * lw / hw1)
    h = max(1, round(lh / ratio))
    w = max(1, hw1 // h)
    if h * w != hw1:
        h, w = _grid_from_count(hw1, lh, lw)
    mask = mask.reshape(b, h, w).unsqueeze(1).to(x0.dtype)
    mask = F.interpolate(mask, (lh, lw))
    blurred = gaussian_blur_2d(x0, kernel_size=9, sigma=sigma)
    return blurred * mask + x0 * (1 - mask)


def _grid_from_count(count, lh, lw):
    best = (1, count)
    for h in range(1, count + 1):
        if count % h == 0:
            w = count // h
            if abs(w / h - lw / lh) < abs(best[1] / best[0] - lw / lh):
                best = (h, w)
    return best
