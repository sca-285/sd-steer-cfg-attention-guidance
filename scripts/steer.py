"""Steer (CFG Attention Guidance): ten guidance methods, one tab each,
usable alone or together, on Forge, reForge and Forge Classic (Neo)."""

import gradio as gr

from modules import script_callbacks, scripts

from lib_steer import guidance, settings as S

try:
    from modules.ui_components import InputAccordion
except Exception:  # very old WebUIs
    InputAccordion = None

INFOTEXT_KEY = "Steer"
LABEL = "Steer (CFG Attention Guidance)"
ORDER = ("fdg", "apg", "chg", "ag", "pag", "seg", "sag", "tpg", "nag", "swg")

script_callbacks.on_cfg_denoiser(guidance.STATE.on_cfg_denoiser)


def _component(f, eid):
    common = dict(label=f.label, elem_id=eid)
    if f.kind == "check":
        return gr.Checkbox(value=f.default, **common)
    if f.kind == "slider":
        return gr.Slider(minimum=f.lo, maximum=f.hi, step=f.step, value=f.default, **common)
    if f.kind == "radio":
        return gr.Radio(choices=f.choices, value=f.default, **common)
    return gr.Textbox(value=f.default, lines=1, max_lines=1, **common)


def _status(*flags):
    active = [S.MODULES[m][0] for m, on in zip(ORDER, flags) if on]
    return "Active: " + (", ".join(active) if active else "none")


class Steer(scripts.Script):
    sorting_priority = 14

    def title(self):
        return "Steer"          # also the API name (alwayson_scripts "Steer")

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"
        comps = {}

        def eid(key):
            return f"{tab}_steer_{key.replace('.', '_')}"

        container = (InputAccordion(False, label=LABEL, elem_id=eid("accordion")) if InputAccordion
                     else gr.Accordion(LABEL, open=False))
        with container as acc:
            if InputAccordion:
                comps["general.enabled"] = acc
            else:
                comps["general.enabled"] = gr.Checkbox(value=False, label="Enable", elem_id=eid("general.enabled"))
            status = gr.Markdown(_status(), elem_id=eid("status"))
            with gr.Tabs(elem_id=eid("tabs")):
                for module in ORDER:
                    title, hint = S.MODULES[module]
                    with gr.Tab(title, elem_id=eid(f"tab_{module}")):
                        comps[f"{module}.enabled"] = _component(S.BY_KEY[f"{module}.enabled"], eid(f"{module}.enabled"))
                        gr.Markdown(hint, elem_classes=["steer-hint"])
                        fields = [f for f in S.SPEC if f.key.startswith(module + ".") and f.key != f"{module}.enabled"]
                        for i in range(0, len(fields), 2):          # two controls per row
                            with gr.Row():
                                for f in fields[i:i + 2]:
                                    comps[f.key] = _component(f, eid(f.key))

        switches = [comps[f"{m}.enabled"] for m in ORDER]
        for switch in switches:
            switch.change(_status, inputs=switches, outputs=[status], show_progress=False)

        def paste(key):
            def read(d):
                if INFOTEXT_KEY not in d:
                    return False if key == "general.enabled" else None
                try:
                    s = S.from_infotext(d[INFOTEXT_KEY])
                except Exception:
                    return None
                module, name = key.split(".")
                return s[module][name]
            return read

        self.infotext_fields = [(comps[f.key], paste(f.key)) for f in S.SPEC]
        self.paste_field_names = [INFOTEXT_KEY]
        return [comps[f.key] for f in S.SPEC]

    def process_before_every_sampling(self, p, *args, **kwargs):
        s = S.from_args(args)
        _apply_xyz(p, s)
        hr = bool(getattr(p, "is_hr_pass", False))
        if not s["general"]["enabled"] or not S.enabled_modules(s):
            guidance.finish()
            return
        applied = guidance.install(p, s)
        if applied:
            p.extra_generation_params[INFOTEXT_KEY] = S.to_infotext(s)
            names = ", ".join(S.MODULES[m][0] for m in applied)
            print(f"[Steer] {'hires pass' if hr else 'pass'}: {names}")

    def postprocess(self, p, processed, *args):
        guidance.finish()


# --------------------------------------------------------------------- X/Y/Z plot

XYZ = [
    ("FDG low-frequency share", "fdg.low_share", float),
    ("FDG levels", "fdg.levels", int),
    ("APG eta", "apg.eta", float),
    ("APG norm threshold", "apg.norm_threshold", float),
    ("APG momentum", "apg.momentum", float),
    ("PAG scale", "pag.scale", float),
    ("SEG scale", "seg.scale", float),
    ("SEG blur sigma", "seg.blur_sigma", float),
    ("SAG scale", "sag.scale", float),
    ("NAG scale", "nag.scale", float),
    ("NAG tau", "nag.tau", float),
    ("NAG alpha", "nag.alpha", float),
    ("AG threshold", "ag.threshold", float),
    ("TPG scale", "tpg.scale", float),
    ("CHG regularization", "chg.reg_strength", float),
    ("CHG regularization range", "chg.reg_range", float),
    ("CHG end", "chg.end", float),
    ("SWG scale", "swg.scale", float),
    ("SWG window", "swg.window", int),
]
MODULE_NAMES = {S.MODULES[m][0]: m for m in S.MODULES}


def _apply_xyz(p, s):
    overrides = getattr(p, "_steer_xyz", None)
    if not overrides:
        return
    # the Modules axis sets which modules run; the other axes then turn on
    # the module whose value they set
    for key, value in sorted(overrides.items(), key=lambda kv: kv[0] != "modules"):
        if key == "modules":
            wanted = {MODULE_NAMES.get(v.strip(), v.strip().lower()) for v in str(value).split("+")}
            for m in S.MODULES:
                s[m]["enabled"] = m in wanted
            s["general"]["enabled"] = bool(wanted - {"none", ""})
            continue
        module, name = key.split(".")
        if key == "swg.window":           # one axis for a square window
            s["swg"]["window_width"] = s["swg"]["window_height"] = value
        else:
            s[module][name] = value
        s[module]["enabled"] = True
        s["general"]["enabled"] = True


def _setter(key):
    def apply(p, x, xs):
        overrides = dict(getattr(p, "_steer_xyz", {}) or {})
        overrides[key] = x
        p._steer_xyz = overrides
    return apply


def _register_xyz():
    xyz = None
    for data in scripts.scripts_data:
        if data.script_class.__module__ in ("xyz_grid.py", "scripts.xyz_grid") or data.path.endswith("xyz_grid.py"):
            xyz = data.module
            break
    if xyz is None:
        return
    if any(getattr(o, "label", "").startswith("[Steer]") for o in xyz.axis_options):
        return
    options = [xyz.AxisOption(f"[Steer] {label}", kind, _setter(key)) for label, key, kind in XYZ]
    options.append(xyz.AxisOption("[Steer] Modules", str, _setter("modules"),
                                  choices=lambda: ["none"] + [S.MODULES[m][0] for m in S.MODULES]))
    xyz.axis_options.extend(options)


try:
    script_callbacks.on_before_ui(_register_xyz)
except Exception:
    pass
