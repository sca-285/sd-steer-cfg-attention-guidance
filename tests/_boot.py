"""Boot the WebUI in the current directory the way its webui.py does, without
a checkpoint, so tests run against the real modules. Import this first."""
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
WEBUI = pathlib.Path.cwd()
sys.path.insert(0, str(WEBUI))


def detect():
    if (WEBUI / "ldm_patched").is_dir():
        return "reforge"
    if (WEBUI / "modules_forge/packages/k_diffusion").is_dir():
        return "neo"
    if (WEBUI / "backend").is_dir():
        return "forge"
    raise SystemExit(f"run from a Forge, reForge or Forge Classic (Neo) folder (cwd is {WEBUI})")


KIND = detect()
sys.argv = ["webui.py", "--skip-prepare-environment", "--skip-install",
            *(["--cpu"] if KIND == "neo" else ["--always-cpu", "--no-download-sd-model"])]


def boot():
    from modules import initialize, timer  # noqa: F401
    if hasattr(initialize, "shush"):
        initialize.shush()
    from modules_forge.initialization import initialize_forge
    initialize_forge()
    initialize.imports()
    initialize.initialize()


boot()
