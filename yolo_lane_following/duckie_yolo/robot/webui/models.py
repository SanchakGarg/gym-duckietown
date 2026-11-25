"""Model management for the web UI: list / load / stop models, and long-running jobs (TensorRT build)."""
from __future__ import annotations

import collections
import importlib.util
import os
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ...agent import LaneFollowingAgent
from ...config import ActuationConfig, load_agent_config
from ...perception.camera_model import CameraModel
from ...perception.segmenters import build_segmenter
from .util import own, safe_filename

MODEL_EXTS = (".pt", ".onnx", ".engine")
COLOR = "__color__"  # pseudo model: HSV colour thresholds, no neural network


def ultralytics_available() -> bool:
    return importlib.util.find_spec("ultralytics") is not None


def _free_gpu_memory() -> None:
    if "torch" in sys.modules:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class ModelManager:
    """Owns the running LaneFollowingAgent. Loading happens in a background thread."""

    def __init__(self, models_dir: str, agent_config: str = ""):
        self.models_dir = models_dir
        self.agent_config = agent_config
        os.makedirs(models_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._gen = 0
        self.agent: Optional[LaneFollowingAgent] = None
        self.state = "idle"  # idle | loading | running | error
        self.name: Optional[str] = None
        self.error: Optional[str] = None
        self.device: Optional[str] = None
        self.conf: Optional[float] = None
        self._actuation_from_robot = True

    # ------------------------------------------------------------------ files
    def list(self) -> List[Dict[str, Any]]:
        yolo_ok = ultralytics_available()
        out = [{"name": COLOR, "label": "Colour thresholds (no model)", "kind": "color", "usable": True,
                "bytes": 0, "mtime": 0}]
        for fn in sorted(os.listdir(self.models_dir)):
            p = os.path.join(self.models_dir, fn)
            if os.path.isfile(p) and fn.lower().endswith(MODEL_EXTS):
                st = os.stat(p)
                out.append({"name": fn, "label": fn, "kind": os.path.splitext(fn)[1][1:].lower(), "usable": yolo_ok,
                            "bytes": st.st_size, "mtime": st.st_mtime})
        return out

    def path(self, name: str) -> str:
        return os.path.join(self.models_dir, safe_filename(name, MODEL_EXTS))

    def save_upload(self, name: str, stream, length: int, chunk: int = 1 << 20) -> str:
        """Write an uploaded model (raw request body) atomically into models/."""
        dst = self.path(name)
        tmp = dst + ".part"
        remaining = length
        with open(tmp, "wb") as f:
            while remaining > 0:
                buf = stream.read(min(chunk, remaining))
                if not buf:
                    raise IOError("upload interrupted")
                f.write(buf)
                remaining -= len(buf)
        os.replace(tmp, dst)
        own(dst)
        return dst

    def delete(self, name: str) -> None:
        if self.name == name and self.state in ("loading", "running"):
            raise ValueError("stop the model before deleting it")
        os.remove(self.path(name))

    # ------------------------------------------------------------------ running
    def start(self, name: str, camera: CameraModel, actuation: ActuationConfig, conf: Optional[float] = None) -> None:
        if name != COLOR:
            self.path(name)  # validates
            if not ultralytics_available():
                raise RuntimeError("Ultralytics is not installed here: start the web UI in the Docker container "
                                   "(duckie_yolo/robot/webui/start_webui.sh) to run YOLO models")
        with self._lock:
            self._gen += 1
            gen = self._gen
            self.agent = None
            self.state, self.name, self.error, self.device, self.conf = "loading", name, None, None, conf
        threading.Thread(target=self._load, args=(gen, name, camera, actuation, conf), daemon=True,
                         name="model-load").start()

    def _load(self, gen: int, name: str, camera: CameraModel, actuation: ActuationConfig, conf: Optional[float]) -> None:
        try:
            cfg = load_agent_config(self.agent_config if self.agent_config and os.path.exists(self.agent_config) else "")
            if name == COLOR:
                kind, cfg.segmenter.weights = "color", ""
            else:
                kind, cfg.segmenter.weights = "yolo", self.path(name)
            if conf is not None:
                cfg.segmenter.conf = float(conf)
            from_robot = cfg.actuation == ActuationConfig()
            if from_robot:
                cfg.actuation = actuation  # from the wheel calibration
            seg = build_segmenter(cfg.segmenter, cfg.estimator.work_scale, kind)
            agent = LaneFollowingAgent(seg, camera, cfg)
            w, h = camera.rectifier.in_size if camera.rectifier is not None else (camera.width, camera.height)
            agent.act(np.zeros((h, w, 3), np.uint8))  # warm-up: CUDA / TensorRT initialisation
            agent.reset()
            device = "cpu (colour thresholds)" if kind == "color" else _model_device(seg)
            with self._lock:
                if gen != self._gen:
                    return  # stopped or replaced while loading
                self.agent, self.state, self.device = agent, "running", device
                self._actuation_from_robot = from_robot
                self.conf = cfg.segmenter.conf
            print(f"[webui] model {name} running on {device}", flush=True)
        except Exception as e:
            with self._lock:
                if gen == self._gen:
                    self.state, self.error, self.agent = "error", f"{type(e).__name__}: {e}", None
            print(f"[webui] model {name} failed: {e}", flush=True)

    def stop(self) -> None:
        with self._lock:
            self._gen += 1
            self.agent = None
            self.state, self.name, self.error, self.device = "idle", None, None, None
        _free_gpu_memory()

    def fail(self, err: str) -> None:
        """Called by the control loop when inference raises."""
        with self._lock:
            self._gen += 1
            self.agent, self.state, self.error = None, "error", err

    def set_camera(self, camera: CameraModel, actuation: Optional[ActuationConfig] = None) -> None:
        """New calibration: rebuild the agent around the same (already loaded) segmenter."""
        with self._lock:
            if self.agent is not None:
                cfg = self.agent.cfg
                if actuation is not None and self._actuation_from_robot:
                    cfg.actuation = actuation
                self.agent = LaneFollowingAgent(self.agent.segmenter, camera, cfg)

    def set_conf(self, conf: float) -> None:
        with self._lock:
            a = self.agent
            if a is not None and hasattr(a.segmenter, "_kw"):
                a.segmenter._kw["conf"] = float(conf)
                a.cfg.segmenter.conf = float(conf)
                self.conf = float(conf)

    def status(self) -> Dict[str, Any]:
        return {"state": self.state, "name": self.name, "error": self.error, "device": self.device, "conf": self.conf}


def _model_device(seg) -> str:
    pred = getattr(getattr(seg, "model", None), "predictor", None)
    dev = getattr(pred, "device", None)
    if dev is None:
        return "unknown"
    s = str(dev)
    if s.startswith("cuda"):
        try:
            import torch

            return f"{s} ({torch.cuda.get_device_name(0)})"
        except Exception:
            pass
    return s


class JobRunner:
    """Runs one long subprocess at a time (e.g. the TensorRT engine build) and keeps its log."""

    def __init__(self, max_lines: int = 300):
        self._lock = threading.Lock()
        self.log: "collections.deque[str]" = collections.deque(maxlen=max_lines)
        self.state = "idle"  # idle | running | done | failed | cancelled
        self.name: Optional[str] = None
        self.returncode: Optional[int] = None
        self.started = 0.0
        self.ended = 0.0
        self._proc: Optional[subprocess.Popen] = None
        self._cancelled = False

    def running(self) -> bool:
        return self.state == "running"

    def start(self, name: str, cmd: Sequence[str], cwd: Optional[str] = None,
              on_done: Optional[Callable[[int], None]] = None) -> None:
        with self._lock:
            if self.state == "running":
                raise RuntimeError(f"job '{self.name}' is still running")
            self.log.clear()
            self.state, self.name, self.returncode, self._cancelled = "running", name, None, False
            self.started, self.ended = time.time(), 0.0
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            self._proc = subprocess.Popen(list(cmd), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          env=env, bufsize=0)
        self.log.append("$ " + " ".join(cmd))
        threading.Thread(target=self._pump, args=(self._proc, on_done), daemon=True, name="job").start()

    def _pump(self, proc: subprocess.Popen, on_done) -> None:
        buf = b""
        while True:
            chunk = proc.stdout.read(512)
            if not chunk:
                break
            buf += chunk
            # progress bars use \r: keep only the latest state of a line
            while True:
                i = min([k for k in (buf.find(b"\n"), buf.find(b"\r")) if k >= 0], default=-1)
                if i < 0:
                    break
                line, sep, buf = buf[:i], buf[i:i + 1], buf[i + 1:]
                text = line.decode(errors="replace").rstrip()
                if text:
                    # a line ended by \r is still "in progress" (marked with a leading \r): the next
                    # output overwrites it, like a terminal would
                    entry = ("\r" if sep == b"\r" else "") + text
                    if self.log and self.log[-1].startswith("\r"):
                        self.log[-1] = entry
                    else:
                        self.log.append(entry)
        if buf.strip():
            self.log.append(buf.decode(errors="replace").rstrip())
        rc = proc.wait()
        with self._lock:
            self.returncode = rc
            self.ended = time.time()
            self.state = "cancelled" if self._cancelled else ("done" if rc == 0 else "failed")
        self.log.append(f"[exit code {rc}]")
        if on_done is not None:
            try:
                on_done(rc)
            except Exception as e:
                self.log.append(f"[post-processing failed: {e}]")

    def cancel(self) -> None:
        with self._lock:
            if self.state == "running" and self._proc is not None:
                self._cancelled = True
                self._proc.terminate()

    def status(self, with_log: bool = False) -> Dict[str, Any]:
        d = {"state": self.state, "name": self.name, "returncode": self.returncode,
             "elapsed": (self.ended or time.time()) - self.started if self.started else 0.0}
        if with_log:
            d["log"] = [l.lstrip("\r") for l in self.log]
        return d
