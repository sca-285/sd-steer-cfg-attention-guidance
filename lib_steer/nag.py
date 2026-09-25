"""NAG, Normalized Attention Guidance (Chen et al. 2025, arXiv:2505.21179;
https://github.com/ChenDarYen/Normalized-Attention-Guidance, MIT).

In every cross-attention layer, the image tokens of the positive batch
attend to the positive and to the negative prompt; the two results are
extrapolated away from the negative, the result's L1 norm is capped at
tau times the positive's, and it is blended back with alpha. This happens
before the output projection, as in the reference, and only for the
positive (cond) rows. The negative prompt works even at CFG 1, which is
what few-step and distilled models need.

The WebUI's CrossAttention modules are wrapped once; the wrapper passes
straight through unless NAG is on for the current step.
"""

from __future__ import annotations

import types

import torch

from lib_steer.attn import attention


def nag(z_positive, z_negative, scale, tau, alpha):
    z_guidance = z_positive * scale - z_negative * (scale - 1)
    eps = 1e-6
    norm_positive = torch.norm(z_positive, p=1, dim=-1, keepdim=True).clamp_min(eps)
    norm_guidance = torch.norm(z_guidance, p=1, dim=-1, keepdim=True).clamp_min(eps)
    s = norm_guidance / norm_positive
    z_guidance = z_guidance * torch.minimum(s, s.new_full((1,), tau)) / s
    return z_guidance * alpha + z_positive * (1 - alpha)


class Controller:
    """What the wrapped layers need to know about the current model call."""

    def __init__(self):
        self.enabled = False
        self.scale = 5.0
        self.tau = 2.5
        self.alpha = 0.25
        self.sigma_end = 0.0
        self.negative = None          # (batch, tokens, dim): the negative prompt's text embedding
        self.cond_or_uncond = None    # of the model call in progress
        self.sigma = None
        self.calls = 0                # layers that applied NAG (for tests / diagnostics)

    def configure(self, enabled, scale=5.0, tau=2.5, alpha=0.25, sigma_end=0.0):
        self.enabled = bool(enabled)
        self.scale, self.tau, self.alpha, self.sigma_end = float(scale), float(tau), float(alpha), float(sigma_end)
        self.negative = None
        self.cond_or_uncond = None

    def stash(self, n, context, value, extra_options):
        """attn2_patch: remember which rows are cond, and the sigma, for this call."""
        self.cond_or_uncond = extra_options.get("cond_or_uncond")
        sigmas = extra_options.get("sigmas")
        try:
            self.sigma = float(sigmas.reshape(-1)[0]) if sigmas is not None else None
        except Exception:
            self.sigma = None
        return n, context, value

    def active(self):
        if not self.enabled or self.negative is None or self.scale <= 1.0:
            return False
        if not self.cond_or_uncond or 0 not in self.cond_or_uncond:
            return False
        return self.sigma is None or self.sigma >= self.sigma_end


CONTROLLER = Controller()


def _nag_forward(self, x, context=None, value=None, mask=None, **kwargs):
    ctrl = CONTROLLER
    if context is None or not ctrl.active():
        return self._steer_original_forward(x, context=context, value=value, mask=mask, **kwargs)
    chunks = ctrl.cond_or_uncond
    if x.shape[0] % len(chunks):
        return self._steer_original_forward(x, context=context, value=value, mask=mask, **kwargs)
    b = x.shape[0] // len(chunks)
    heads = self.heads
    q = self.to_q(x)
    k = self.to_k(context)
    v = self.to_v(value if value is not None else context)
    out = attention(q, k, v, heads, mask)

    negative = ctrl.negative.to(device=x.device, dtype=context.dtype)
    if negative.shape[0] < b:
        negative = negative.repeat((b + negative.shape[0] - 1) // negative.shape[0], 1, 1)
    negative = negative[:b]
    if negative.shape[-1] != context.shape[-1]:
        return self.to_out(out)     # a different text encoder than this layer expects: leave it
    k_neg = self.to_k(negative)
    v_neg = self.to_v(negative)
    for i, kind in enumerate(chunks):
        if kind != 0:
            continue
        rows = slice(i * b, (i + 1) * b)
        z_neg = attention(q[rows], k_neg, v_neg, heads)
        out[rows] = nag(out[rows], z_neg, ctrl.scale, ctrl.tau, ctrl.alpha)
    ctrl.calls += 1
    return self.to_out(out)


def cross_attention_layers(diffusion_model):
    """The UNet's cross-attention modules (named ...attn2)."""
    layers = []
    for name, module in diffusion_model.named_modules():
        if name.endswith("attn2") and all(hasattr(module, a) for a in ("to_q", "to_k", "to_v", "to_out", "heads")):
            layers.append(module)
    return layers


def wrap(diffusion_model):
    """Wrap every cross-attention layer once. Returns how many there are."""
    layers = cross_attention_layers(diffusion_model)
    for module in layers:
        if getattr(module, "_steer_original_forward", None) is None:
            module._steer_original_forward = module.forward
            module.forward = types.MethodType(_nag_forward, module)
    return len(layers)


def unwrap(diffusion_model):
    for module in cross_attention_layers(diffusion_model):
        original = getattr(module, "_steer_original_forward", None)
        if original is not None:
            module.forward = original
            module._steer_original_forward = None


def negative_embedding(text_uncond):
    """The cross-attention part of the WebUI's uncond conditioning (SD: tensor, SDXL: dict)."""
    if text_uncond is None:
        return None
    if isinstance(text_uncond, dict):
        text_uncond = text_uncond.get("crossattn")
    return text_uncond if isinstance(text_uncond, torch.Tensor) and text_uncond.ndim == 3 else None
