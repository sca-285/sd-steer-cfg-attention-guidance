"""TPG, Token Perturbation Guidance (Rajabi et al., NeurIPS 2025,
arXiv:2506.10036; https://github.com/TaatiTeam/Token-Perturbation-Guidance,
MIT), as also implemented in pamparamm/sd-perturbed-attention (`tpg_nodes.py`,
MIT).

The weak prediction is the positive prediction with the tokens entering
some transformer blocks shuffled (one random permutation of the token
order, the same for the whole batch). The guidance adds
scale * (cond - cond with shuffled tokens), like PAG and SEG, but the
attention itself is left alone.

The WebUI's transformer blocks are wrapped once; the wrapper passes
straight through unless the model call carries the TPG flag in its
transformer options and the block is one of the chosen ones.
"""

from __future__ import annotations

import re
import types

import torch

FLAG = "steer_tpg"


class Controller:
    def __init__(self):
        self.blocks = set()          # id() of the chosen transformer blocks
        self.generator = None        # CPU generator: the WebUI's own random state is left alone
        self.calls = 0

    def configure(self, blocks=(), seed=0):
        self.blocks = {id(b) for b in blocks}
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed) & 0xFFFFFFFF)
        self.calls = 0

    def permutation(self, n, device):
        if self.generator is None:
            self.generator = torch.Generator(device="cpu")
            self.generator.manual_seed(0)
        return torch.randperm(n, generator=self.generator).to(device)


CONTROLLER = Controller()


def _tpg_forward(self, x, context=None, transformer_options={}, **kwargs):
    ctrl = CONTROLLER
    if transformer_options.get(FLAG) and id(self) in ctrl.blocks:
        x = x[:, ctrl.permutation(x.shape[1], x.device)]
        ctrl.calls += 1
    return self._steer_tpg_original_forward(x, context=context, transformer_options=transformer_options, **kwargs)


def transformer_blocks(diffusion_model):
    """[(name, module)] of the UNet's transformer blocks."""
    return [(name, module) for name, module in diffusion_model.named_modules()
            if type(module).__name__ == "BasicTransformerBlock"]


def wrap(diffusion_model):
    blocks = transformer_blocks(diffusion_model)
    for _, module in blocks:
        if getattr(module, "_steer_tpg_original_forward", None) is None:
            module._steer_tpg_original_forward = module.forward
            module.forward = types.MethodType(_tpg_forward, module)
    return len(blocks)


def unwrap(diffusion_model):
    for _, module in transformer_blocks(diffusion_model):
        original = getattr(module, "_steer_tpg_original_forward", None)
        if original is not None:
            module.forward = original
            module._steer_tpg_original_forward = None


_NAME = re.compile(r"^(input_blocks|middle_block|output_blocks)\.(\d+)\..*transformer_blocks\.(\d+)$")
_PART = {"input_blocks": "input", "middle_block": "middle", "output_blocks": "output"}


def locate(name):
    """'input_blocks.7.1.transformer_blocks.3' -> ('input', 7, 3); middle blocks are number 0."""
    m = _NAME.match(name)
    if not m:
        return None
    part = _PART[m.group(1)]
    return part, (0 if part == "middle" else int(m.group(2))), int(m.group(3))


def default_blocks(diffusion_model):
    """'auto': every transformer of the lowest-resolution input stage that has
    attention, skipping its first two transformers when it has more than two.
    On SDXL this is the authors' default (layers d6-d23, sd-perturbed-attention's
    "d2.2-9,d3": input 7 from its third transformer, and input 8); on
    SD 1.x / 2.x it is input 7 and 8."""
    found = [(locate(n), m) for n, m in transformer_blocks(diffusion_model)]
    found = [(loc, m) for loc, m in found if loc and loc[0] == "input"]
    if not found:
        return [m for _, m in transformer_blocks(diffusion_model)]
    numbers = sorted({loc[1] for loc, _ in found})
    # input blocks with attention come in pairs per resolution stage
    stage = numbers[-2:] if len(numbers) >= 2 else numbers
    chosen = []
    for number in stage:
        members = [(loc[2], m) for loc, m in found if loc[1] == number]
        if number == stage[0] and len(members) > 2:
            members = [(i, m) for i, m in members if i >= 2]
        chosen += [m for _, m in members]
    return chosen


def select_blocks(diffusion_model, spec):
    """The transformer blocks named by `spec` ('auto', or e.g. 'input 7.2-9, input 8, middle')."""
    from lib_steer.attn import parse_block_ranges

    if not spec or str(spec).strip().lower() == "auto":
        return default_blocks(diffusion_model)
    wanted = parse_block_ranges(spec)
    chosen = []
    for name, module in transformer_blocks(diffusion_model):
        loc = locate(name)
        if loc is None:
            continue
        for part, number, lo, hi in wanted:
            if loc[0] == part and (part == "middle" or loc[1] == number) and (lo is None or lo <= loc[2] <= hi):
                chosen.append(module)
                break
    return chosen
