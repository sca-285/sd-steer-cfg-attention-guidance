"""Every control, declared once: the UI, the PNG info and the X/Y/Z plot axes
are all built from this table, and the arguments come back in its order."""

from __future__ import annotations

import json
from urllib.parse import quote, unquote
from dataclasses import dataclass, field



@dataclass
class F:
    key: str                 # "module.name"
    label: str
    kind: str                # slider | check | drop | text | radio
    default: object
    lo: float = 0.0
    hi: float = 1.0
    step: float = 0.01
    choices: list = field(default_factory=list)
    info: str = ""
    advanced: bool = False


MODULES = {
    # key: (tab title, one short hint)
    "fdg": ("FDG", "Detail at your CFG; shape and colour at CFG x low share."),
    "apg": ("APG", "High CFG without oversaturation. Raise CFG (SDXL ~10-15)."),
    "pag": ("PAG", "Better structure. +1 model pass per step. Works at CFG 1."),
    "seg": ("SEG", "Like PAG, gentler on colour. Blur 10000 = infinite. +1 pass/step."),
    "sag": ("SAG", "Sharper detail. Needs CFG > 1. +1 model pass per step."),
    "nag": ("NAG", "Negative prompt that works at CFG 1. Sigma end 0 = all steps."),
    "ag": ("AG", "Stops CFG once positive and negative agree: faster. Threshold 0.99-1."),
    "tpg": ("TPG", "Like PAG, shuffles tokens instead of attention. +1 pass/step."),
    "chg": ("CHG", "Cleaner high CFG (10-30). Slow: many passes/step; end 0.4 is faster."),
    "swg": ("SWG", "Guidance from smaller windows of the image. ~+4 passes/step on SDXL."),
}

SPEC = [
    F("general.enabled", "Enable", "check", False),

    F("fdg.enabled", "Enable FDG", "check", False),
    F("fdg.low_share", "FDG low share", "slider", 0.5, 0.0, 1.0, 0.05),
    F("fdg.levels", "FDG levels", "slider", 2, 2, 5, 1),

    F("apg.enabled", "Enable APG", "check", False),
    F("apg.eta", "APG eta", "slider", 0.0, 0.0, 1.0, 0.05),
    F("apg.norm_threshold", "APG norm threshold", "slider", 15.0, 0.0, 50.0, 0.5),
    F("apg.momentum", "APG momentum", "slider", -0.5, -1.0, 1.0, 0.05),

    F("pag.enabled", "Enable PAG", "check", False),
    F("pag.scale", "PAG scale", "slider", 3.0, 0.0, 20.0, 0.1),
    F("pag.start", "PAG start", "slider", 0.0, 0.0, 1.0, 0.01),
    F("pag.end", "PAG end", "slider", 1.0, 0.0, 1.0, 0.01),
    F("pag.blocks", "PAG UNet blocks", "text", "middle"),

    F("seg.enabled", "Enable SEG", "check", False),
    F("seg.scale", "SEG scale", "slider", 3.0, 0.0, 20.0, 0.1),
    F("seg.blur_sigma", "SEG blur sigma", "slider", 10000.0, 0.0, 10000.0, 0.5),
    F("seg.start", "SEG start", "slider", 0.0, 0.0, 1.0, 0.01),
    F("seg.end", "SEG end", "slider", 1.0, 0.0, 1.0, 0.01),
    F("seg.blocks", "SEG UNet blocks", "text", "middle"),

    F("sag.enabled", "Enable SAG", "check", False),
    F("sag.scale", "SAG scale", "slider", 0.5, -2.0, 5.0, 0.01),
    F("sag.blur_sigma", "SAG blur sigma", "slider", 2.0, 0.0, 10.0, 0.01),
    F("sag.threshold", "SAG mask threshold", "slider", 1.0, 0.0, 4.0, 0.01),

    F("nag.enabled", "Enable NAG", "check", False),
    F("nag.scale", "NAG scale", "slider", 5.0, 1.0, 20.0, 0.1),
    F("nag.tau", "NAG tau", "slider", 2.5, 1.0, 10.0, 0.1),
    F("nag.alpha", "NAG alpha", "slider", 0.25, 0.0, 1.0, 0.01),
    F("nag.sigma_end", "NAG sigma end", "slider", 0.0, 0.0, 20.0, 0.01),

    F("ag.enabled", "Enable AG", "check", False),
    F("ag.threshold", "AG threshold", "slider", 0.99, 0.9, 1.0, 0.0005),
    F("ag.start", "AG check from", "slider", 0.0, 0.0, 1.0, 0.01),
    F("ag.compare", "AG compare", "radio", "Denoised", choices=["Denoised", "Noise"]),

    F("tpg.enabled", "Enable TPG", "check", False),
    F("tpg.scale", "TPG scale", "slider", 3.0, 0.0, 20.0, 0.1),
    F("tpg.start", "TPG start", "slider", 0.0, 0.0, 1.0, 0.01),
    F("tpg.end", "TPG end", "slider", 1.0, 0.0, 1.0, 0.01),
    F("tpg.blocks", "TPG UNet blocks", "text", "auto"),

    F("chg.enabled", "Enable CHG", "check", False),
    F("chg.reg_strength", "CHG regularization", "slider", 1.0, 0.0, 10.0, 0.01),
    F("chg.reg_range", "CHG regularization range", "slider", 1.0, 0.01, 10.0, 0.01),
    F("chg.max_iter", "CHG max iterations", "slider", 50, 1, 50, 1),
    F("chg.bases", "CHG bases", "slider", 0, 0, 10, 1),
    F("chg.start", "CHG start", "slider", 0.0, 0.0, 1.0, 0.01),
    F("chg.end", "CHG end", "slider", 1.0, 0.0, 1.0, 0.01),
    F("chg.reuse", "CHG reuse previous", "slider", 1.0, 0.0, 1.0, 0.01),
    F("chg.log_tol", "CHG log10 tolerance", "slider", -4.0, -6.0, -2.0, 0.1),
    F("chg.step_size", "CHG step size", "slider", 1.0, 0.0, 1.0, 0.01),
    F("chg.anneal_speed", "CHG annealing speed", "slider", 0.4, 0.0, 1.0, 0.1),
    F("chg.anneal_strength", "CHG annealing strength", "slider", 0.5, 0.0, 5.0, 0.01),
    F("chg.aa_memory", "CHG AA memory", "slider", 2, 1, 10, 1),

    F("swg.enabled", "Enable SWG", "check", False),
    F("swg.scale", "SWG scale", "slider", 5.0, 0.0, 30.0, 0.1),
    F("swg.window_width", "SWG window width", "slider", 768, 64, 2048, 8),
    F("swg.window_height", "SWG window height", "slider", 768, 64, 2048, 8),
    F("swg.overlap", "SWG overlap", "slider", 256, 0, 1024, 8),
    F("swg.start", "SWG start", "slider", 0.0, 0.0, 1.0, 0.01),
    F("swg.end", "SWG end", "slider", 1.0, 0.0, 1.0, 0.01),
]

BY_KEY = {f.key: f for f in SPEC}


def from_args(args) -> dict:
    """Nested settings {module: {name: value}} from the UI's values (SPEC order)."""
    s: dict = {}
    for f, value in zip(SPEC, args):
        module, name = f.key.split(".")
        if f.kind == "slider" and value is not None:
            value = int(value) if isinstance(f.default, int) else float(value)
        s.setdefault(module, {})[name] = f.default if value is None else value
    for f in SPEC[len(args):]:
        module, name = f.key.split(".")
        s.setdefault(module, {})[name] = f.default
    return s


def defaults() -> dict:
    return from_args([f.default for f in SPEC])


def enabled_modules(s) -> list:
    return [m for m in MODULES if s.get(m, {}).get("enabled")]


def _fmt(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(round(value, 6))
    if isinstance(value, int):
        return str(value)
    return quote(str(value), safe="")


def _parse(text):
    if text in ("true", "false"):
        return text == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return unquote(text)


def to_infotext(s) -> str:
    """The enabled modules and their non-default values, e.g.
    'apg; pag scale=4.5; ag compare=Noise'. No commas or colons, so PNG info
    shows it as is."""
    parts = []
    for m in enabled_modules(s):
        values = []
        for f in SPEC:
            module, name = f.key.split(".")
            if module == m and name != "enabled" and s[m][name] != f.default:
                values.append(f"{name}={_fmt(s[m][name])}")
        parts.append(" ".join([m] + values))
    return "; ".join(parts)


def from_infotext(text):
    """Settings from PNG info: every module listed is enabled, missing values are defaults."""
    text = str(text or "").strip()
    if text.startswith('"') and text.endswith('"'):
        text = json.loads(text)
    if text.startswith("{"):            # JSON form
        data = json.loads(text)
    else:
        data = {}
        for part in filter(None, (x.strip() for x in text.split(";"))):
            tokens = part.split()
            data[tokens[0]] = dict(t.split("=", 1) for t in tokens[1:] if "=" in t)
            data[tokens[0]] = {k: _parse(v) for k, v in data[tokens[0]].items()}
    s = defaults()
    s["general"]["enabled"] = bool([k for k in data if k in MODULES])
    for m, values in data.items():
        if m not in MODULES or not isinstance(values, dict):
            continue
        s[m]["enabled"] = True
        for name, value in values.items():
            if name in s[m]:
                f = BY_KEY[f"{m}.{name}"]
                s[m][name] = type(f.default)(value) if isinstance(f.default, (int, float)) and not isinstance(f.default, bool) else value
    return s
