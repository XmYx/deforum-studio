"""
Client for the FLUX.3 Action world-model worker (``flux3_action_worker.py``).

The worker runs in a separate Python 3.12 environment (``install_worldmodel.sh`` creates it as
``.venv-worldmodel``) because `flux-action` pins torch/transformers/numpy versions that differ from Deforum's.
Configuration (environment or .env):

  WORLD_MODEL_PYTHON     python of that environment (default: <repo>/.venv-worldmodel/bin/python)
  WORLD_MODEL_POLICY     Hub repo or local dir of a FLUX.3 Action policy (default: black-forest-labs/flux-3-action-droid)
  WORLD_MODEL_SUBFOLDER  package variant (default: variants/gd-fp8r = guidance-distilled, 4 steps, FP8)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading

from PIL import Image

from deforum.utils.logging_config import logger

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
DEFAULT_POLICY = "black-forest-labs/flux-3-action-droid"
DEFAULT_SUBFOLDER = "variants/gd-fp8r"


def _env(key, default):
    from decouple import config
    return config(key, default=default)


def default_python():
    exe = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    return _env("WORLD_MODEL_PYTHON", os.path.join(_REPO, ".venv-worldmodel", exe))


class WorldModelClient:
    """Starts the worker on first use and keeps it (and the loaded policy) alive between calls."""

    def __init__(self, policy=None, subfolder=None, python=None, offload_text_encoder=True):
        self.policy = policy or _env("WORLD_MODEL_POLICY", DEFAULT_POLICY)
        self.subfolder = subfolder if subfolder is not None else (_env("WORLD_MODEL_SUBFOLDER", DEFAULT_SUBFOLDER) or None)
        self.python = python or default_python()
        self.offload_text_encoder = offload_text_encoder
        self.proc = None
        self.info = None
        self.tmp = tempfile.mkdtemp(prefix="deforum_wm_")
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ process
    def _start(self):
        if self.proc is not None and self.proc.poll() is None:
            return
        if not os.path.isfile(self.python):
            raise RuntimeError(
                f"World model environment not found ({self.python}). Run install_worldmodel.sh, or set "
                f"WORLD_MODEL_PYTHON to a Python 3.12 environment with flux-action installed.")
        worker = _env("WORLD_MODEL_WORKER", os.path.join(_HERE, "flux3_action_worker.py"))  # override for tests
        logger.info(f"Starting FLUX.3 Action worker: {self.python} {worker}")
        self.proc = subprocess.Popen([self.python, "-u", worker], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=None, text=True, bufsize=1)
        self.info = self._call(cmd="load", policy=self.policy, subfolder=self.subfolder,
                               offload_text_encoder=self.offload_text_encoder)["info"]
        logger.info(f"FLUX.3 Action loaded: {self.policy} [{self.subfolder or 'root'}] {self.info}")

    def _call(self, **req):
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"World model worker exited (code {self.proc.poll()}); see its log above")
        resp = json.loads(line)
        if not resp.get("ok"):
            raise RuntimeError(f"World model worker error: {resp.get('error')}")
        return resp

    # ------------------------------------------------------------------ API
    def predict(self, image: Image.Image, caption: str, seed: int = 0, guidance=None, steps=None):
        """Current frame + instruction -> list of predicted future frames (PIL, same size as ``image``)."""
        with self._lock:
            self._start()
            src = os.path.join(self.tmp, "observed.png")
            image.convert("RGB").save(src, compress_level=1)
            out_dir = os.path.join(self.tmp, "pred")
            shutil.rmtree(out_dir, ignore_errors=True)
            resp = self._call(cmd="predict", image=src, caption=caption or "", seed=int(seed), out_dir=out_dir,
                              guidance=guidance, steps=steps)
            return [Image.open(p).convert("RGB") for p in resp["frames"]]

    @property
    def fps(self):
        return (self.info or {}).get("fps", 15.0)

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                self._call(cmd="quit")
            except Exception:
                self.proc.kill()
        self.proc = None
        shutil.rmtree(self.tmp, ignore_errors=True)
