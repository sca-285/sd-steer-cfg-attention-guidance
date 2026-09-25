# Notice

Steer (CFG Attention Guidance) is licensed under the GNU General Public License v3.0 (see [LICENSE](LICENSE)).

The code is written for this extension. Where it follows a published implementation, the method and its source are listed below. Each source's licence permits this use in a GPL-3.0 work.

| File | Method | Follows | Licence of the source |
|---|---|---|---|
| `lib_steer/cfg.py` `fdg`, `laplacian_pyramid`, `reconstruct` | FDG | Sadat et al. 2025, Algorithm 2 (arXiv:2506.19713); pyramid filters as in [kornia](https://github.com/kornia/kornia) `pyrdown` / `pyrup` | paper / Apache-2.0 |
| `lib_steer/cfg.py` `APG`, `project` | APG | Sadat et al. 2024, Algorithm 1 (arXiv:2410.02416) | paper |
| `lib_steer/cfg.py` `similarity` | AG | Castillo et al. 2023 (arXiv:2312.12487); [asagi4/ComfyUI-Adaptive-Guidance](https://github.com/asagi4/ComfyUI-Adaptive-Guidance) | paper / GPL-3.0 |
| `lib_steer/cfg.py` `window_starts`, `sliding_window_prediction` | SWG | Kaiser et al. 2024 (arXiv:2411.10257); the WebUI form of [pamparamm/sd-perturbed-attention](https://github.com/pamparamm/sd-perturbed-attention) `swg_pred_calc` (post-CFG term, window size in pixels), with the windows laid out so each has the chosen size | paper / MIT |
| `lib_steer/chg.py` | CHG | Zheng & Lan 2024 (arXiv:2312.07586); [scraed/CharacteristicGuidanceWebUI](https://github.com/scraed/CharacteristicGuidanceWebUI) `CharaIte.py` (`split_basis`, `proj_least_squares`, `solve_least_squares`, the iteration of `chara_ite_inner_loop`) | GPL-3.0 |
| `lib_steer/tpg.py` | TPG | Rajabi et al. 2025 (arXiv:2506.10036); [TaatiTeam/Token-Perturbation-Guidance](https://github.com/TaatiTeam/Token-Perturbation-Guidance) `shuffle_tokens` and default layers; [pamparamm/sd-perturbed-attention](https://github.com/pamparamm/sd-perturbed-attention) `tpg_nodes.py` | MIT / MIT |
| `lib_steer/attn.py` `pag_attention` | PAG | Ahn et al. 2024; ComfyUI `nodes_pag.py` | paper / GPL-3.0 |
| `lib_steer/attn.py` `seg_attention`, `gaussian_blur_2d` | SEG | Hong 2024 (the authors' code, [SusungHong/SEG-SDXL](https://github.com/SusungHong/SEG-SDXL), states no licence and is not copied); [pamparamm/sd-perturbed-attention](https://github.com/pamparamm/sd-perturbed-attention) `guidance_utils.py` | paper / MIT |
| `lib_steer/attn.py` `SAGRecorder`, `sag_blur_map` | SAG | Hong et al. 2023; ComfyUI `nodes_sag.py` | paper / GPL-3.0 |
| `lib_steer/attn.py` `parse_block_ranges` | block list syntax | the transformer ranges of sd-perturbed-attention's `parse_unet_blocks` | MIT |
| `lib_steer/nag.py` | NAG | [ChenDarYen/Normalized-Attention-Guidance](https://github.com/ChenDarYen/Normalized-Attention-Guidance), [ChenDarYen/ComfyUI-NAG](https://github.com/ChenDarYen/ComfyUI-NAG) | MIT |
| `tests/test_math.py` | reference code for the checks | the sources above, copied as they are into the test | as above |

The patcher interface used in `lib_steer/host.py` and `lib_steer/guidance.py` (`sampler_cfg_function`, `sampler_pre_cfg_function`, `sampler_post_cfg_function`, `patches_replace`, `transformer_options`) is ComfyUI's (GPL-3.0), as kept by Forge, reForge and Forge Classic (Neo).

If an attribution is missing or wrong, please open an issue and it will be corrected.
