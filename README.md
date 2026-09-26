# SD Steer (CFG Attention Guidance)

Ten guidance methods for **Forge**, **reForge** and **Forge Classic (Neo)**, one tab each. Use one, or tick several and they run together.

| Tab | Method | What it does | Extra cost per step |
|---|---|---|---|
| **FDG** | Frequency-Decoupled Guidance | Detail guided at your CFG, shape and colour at a lower scale | none |
| **APG** | Adaptive Projected Guidance | High CFG without oversaturation | none |
| **CHG** | Characteristic Guidance | Cleaner images at high CFG (10–30) | several model passes |
| **AG** | Adaptive Guidance | Turns CFG off once it stops mattering: faster | saves passes |
| **PAG** | Perturbed-Attention Guidance | Better structure | 1 model pass |
| **SEG** | Smoothed Energy Guidance | Like PAG, gentler on colour | 1 model pass |
| **SAG** | Self-Attention Guidance | Sharper detail | 1 model pass |
| **TPG** | Token Perturbation Guidance | Like PAG, but shuffles tokens instead of changing attention | 1 model pass |
| **NAG** | Normalized Attention Guidance | A negative prompt that works at CFG 1 | small |
| **SWG** | Sliding Window Guidance | Guidance from the model run on smaller windows of the image | 1 pass per window (4 on SDXL defaults) |

## Install

Extensions → Install from URL, or copy the `sd-webui-steer` folder into `extensions/`. Restart the WebUI. The **Steer (CFG Attention Guidance)** accordion appears in txt2img and img2img.

```bash
git clone https://github.com/sca-285/sd-steer-cfg-attention-guidance.git
```

## Use

1. Tick **Steer (CFG Attention Guidance)**.
2. Open a tab and tick **Enable**. The line at the top lists the active modules.
3. The first pass and the hires fix pass are both guided.

## Starting values

| Module | Start with |
|---|---|
| FDG | low share 0.5 (the paper's SDXL setting) |
| APG | the defaults (eta 0, norm threshold 15, momentum −0.5); CFG 10–15 on SDXL |
| CHG | the defaults; regularization 5 on SDXL; **CHG end** 0.4 for speed |
| AG | threshold 0.99; raise it towards 1 if the image loses prompt adherence |
| PAG / SEG / TPG | scale 3; SEG blur 10000 (infinite) or 4–64 for a milder effect |
| NAG | the defaults; sigma end 4 on SDXL |
| SWG | scale 5; a window about ¾ of the image (768 at 1024), overlap 256 |

## Notes per module

- **FDG + APG:** APG runs inside each FDG band.
- **CHG:**
  - It is iterative. Each iteration evaluates the positive and negative prompt at shifted points, up to **CHG max iterations** times per step.
  - A step that does not converge falls back to plain CFG for that step.
  - Higher **CHG regularization** and **range** converge more easily and stay closer to CFG. Lower values correct more.
  - **CHG bases** above 0 correct per channel instead of per image: more correction, harder convergence.
- **AG:**
  - The step on which the similarity crosses the threshold still uses CFG. From the next step on, the negative pass is skipped. The console says from which step.
  - **AG compare:** *Denoised* compares the denoised predictions, as ComfyUI-Adaptive-Guidance does. *Noise* compares the noise predictions, as the paper does. Those stay very similar from the first steps, so *Noise* needs a threshold close to 1.
- **PAG, SEG, TPG:**
  - **Start** / **End** limit them to part of the steps.
  - PAG and SEG take **UNet blocks** such as `middle, output 1, input 7.0`.
  - TPG takes `auto` (default) or blocks with ranges, such as `input 7.2-9, input 8`. On SDXL, `auto` is the TPG authors' default (their layers d6–d23: input 7 from its third transformer, and input 8). On SD 1.x / 2.x it is input 7 and 8.
- **SWG:**
  - Window and overlap are in image pixels.
  - The window must be smaller than the image, or SWG does nothing (the console says so).
  - An input that cannot be cut into windows, such as a ControlNet hint, turns SWG off for the pass.

## Model support

| | SD 1.x / 2.x / SDXL | Flux, SD3 and other flow / DiT models |
|---|---|---|
| FDG, APG, AG, SWG | ✓ | ✓ |
| CHG | ✓ | skipped |
| PAG, SEG, SAG, TPG, NAG | ✓ | skipped (they need a UNet) |

- CHG, AG and SAG need CFG above 1.
- PAG, SEG, TPG and NAG also work at CFG 1.

**Built-in versions of the same methods:** turn them off while using the same method here. That means Forge's PAG and SAG, and reForge's SAG and APG ("APG is now your CFG").

## PNG info, API and X/Y/Z plot

- Settings are saved in PNG info, for example `Steer: fdg; pag scale=4.5; ag compare=Noise`, and restored when pasted back.
- API: `"alwayson_scripts": {"Steer": {"args": [...]}}`, in the order of `SPEC` in `lib_steer/settings.py`.
- The X/Y/Z plot gets `[Steer] …` axes for the main values. The **Modules** axis switches modules on per cell, for example `PAG+FDG`.

## Tests

- `python tests/test_math.py` checks each method against its reference code (see [NOTICE.md](NOTICE.md)). FDG's check needs kornia.
- `tests/test_in_host.py` is run from a WebUI folder, with that WebUI's Python. It runs every module through the WebUI's own sampling code and UNet classes.

## Acknowledgements

Steer stands on the work of the method authors and of the extensions that brought these methods to the WebUIs first.

**Methods**

- **FDG:** Sadat, Vontobel, Salehi, Weber. *Guidance in the Frequency Domain Enables High-Fidelity Sampling at Low CFG Scales* (2025), arXiv:2506.19713.
- **APG:** Sadat, Hilliges, Weber. *Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models* (ICLR 2025), arXiv:2410.02416.
- **CHG:** Zheng, Lan. *Characteristic Guidance: Non-linear Correction for Diffusion Model at Large Guidance Scale* (ICML 2024), arXiv:2312.07586.
- **AG:** Castillo et al. *Adaptive Guidance: Training-free Acceleration of Conditional Diffusion Models* (AAAI 2025), arXiv:2312.12487.
- **PAG:** Ahn et al. *Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance* (ECCV 2024).
- **SEG:** Hong. *Smoothed Energy Guidance: Guiding Diffusion Models with Reduced Energy Curvature of Attention* (NeurIPS 2024).
- **SAG:** Hong, Lee, Jang, Kim. *Improving Sample Quality of Diffusion Models Using Self-Attention Guidance* (ICCV 2023).
- **TPG:** Rajabi, Mehraban, Sadat, Taati. *Token Perturbation Guidance for Diffusion Models* (NeurIPS 2025), arXiv:2506.10036.
- **NAG:** Chen et al. *Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models* (2025), arXiv:2505.21179.
- **SWG:** Kaiser, Adaloglou, Kollmann. *The Unreasonable Effectiveness of Guidance for Diffusion Models* (2024), arXiv:2411.10257.

**Extensions and code that came before**

- [scraed/CharacteristicGuidanceWebUI](https://github.com/scraed/CharacteristicGuidanceWebUI): the authors' WebUI extension for CHG. Steer's CHG iteration follows it step by step.
- [asagi4/ComfyUI-Adaptive-Guidance](https://github.com/asagi4/ComfyUI-Adaptive-Guidance): Adaptive Guidance for ComfyUI, and its denoised-prediction comparison.
- [TaatiTeam/Token-Perturbation-Guidance](https://github.com/TaatiTeam/Token-Perturbation-Guidance): the authors' TPG code and its default layers.
- [pamparamm/sd-perturbed-attention](https://github.com/pamparamm/sd-perturbed-attention): PAG, SEG, SWG and TPG for ComfyUI and reForge. Its SEG, TPG and SWG are references for Steer's.
- [v0xie/sd-webui-incantations](https://github.com/v0xie/sd-webui-incantations): PAG, SEG and more for AUTOMATIC1111.
- [ChenDarYen/Normalized-Attention-Guidance](https://github.com/ChenDarYen/Normalized-Attention-Guidance) and [ChenDarYen/ComfyUI-NAG](https://github.com/ChenDarYen/ComfyUI-NAG): the authors' NAG code.
- [SusungHong/SEG-SDXL](https://github.com/SusungHong/SEG-SDXL): the authors' SEG code.
- [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI): the PAG and SAG nodes, and the model patcher interface that Forge, reForge and Neo keep.
- The WebUIs:
  - [lllyasviel](https://github.com/lllyasviel)'s [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge), with its built-in PAG and SAG;
  - [Panchovix](https://github.com/Panchovix)'s [reForge](https://github.com/Panchovix/stable-diffusion-webui-reForge), with its built-in APG and SAG;
  - [Haoming02](https://github.com/Haoming02)'s [Forge Classic (Neo)](https://github.com/Haoming02/sd-webui-forge-classic).
- [kornia](https://github.com/kornia/kornia): the pyramid filters that FDG's paper uses.

Thanks also to **Claude**, Anthropic's AI assistant, for help building the extension.

## Licence

GPL-3.0. [NOTICE.md](NOTICE.md) lists which file follows which source, and under which licence.
