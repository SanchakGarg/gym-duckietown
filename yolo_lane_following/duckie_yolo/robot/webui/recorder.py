"""Dataset recording sessions for the web UI.

A session is a folder ``<root>/<YYYYmmdd-HHMMSS>_<name>/`` with
    raw/    camera frames as captured (BGR JPEG) - can be re-corrected after a new calibration
    rect/   the same frames rectified to the sim-like camera: this is what the model sees / gets trained on
    meta.json
Frames are written by a background thread so the control loop never waits for the SD card.
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import threading
import time
import zipfile
from typing import Any, BinaryIO, Callable, Dict, List, Optional

import cv2
import numpy as np

from .util import own

_NAME_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")


def slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", (text or "").strip()).strip("-")
    return s[:40] or "session"


class Recorder:
    def __init__(self, root: str, jpeg_quality: int = 95):
        self.root = root
        self.quality = jpeg_quality
        os.makedirs(root, exist_ok=True)
        self._lock = threading.Lock()
        self.active = False
        self.session: Optional[str] = None
        self.hz = 3.0
        self.count = 0
        self._t_last = 0.0
        self._meta: Dict[str, Any] = {}
        self._q: "queue.Queue" = queue.Queue(maxsize=64)
        self.dropped = 0
        threading.Thread(target=self._writer, daemon=True, name="recorder-writer").start()

    # ------------------------------------------------------------------ recording
    def start(self, name: str = "", hz: float = 3.0, extra_meta: Optional[dict] = None) -> str:
        with self._lock:
            if self.active:
                self._stop_locked()
            sess = time.strftime("%Y%m%d-%H%M%S") + "_" + slug(name)
            d = os.path.join(self.root, sess)
            for sub in ("raw", "rect"):
                os.makedirs(os.path.join(d, sub), exist_ok=True)
            own(self.root)
            own(d, recursive=True)
            self.session, self.active = sess, True
            self.hz = float(min(max(hz, 0.2), 10.0))
            self.count, self._t_last, self.dropped = 0, 0.0, 0
            self._meta = dict(extra_meta or {}, name=name, hz=self.hz, started=time.strftime("%Y-%m-%d %H:%M:%S"))
            self._write_meta()
            return sess

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        if not self.active:
            return
        self.active = False
        self._q.join()  # flush pending frames before writing the final count
        self._meta.update(frames=self.count, dropped=self.dropped, ended=time.strftime("%Y-%m-%d %H:%M:%S"))
        self._write_meta()

    def _write_meta(self) -> None:
        p = os.path.join(self.root, self.session, "meta.json")
        with open(p, "w") as f:
            json.dump(self._meta, f, indent=2)
        own(p)

    def due(self, now: float) -> bool:
        return self.active and now - self._t_last >= 1.0 / self.hz

    def maybe_save(self, now: float, raw_bgr: np.ndarray, rect_bgr: Optional[np.ndarray]) -> bool:
        """Queue a frame if recording and the rate allows it. Called from the control loop."""
        if not self.active or now - self._t_last < 1.0 / self.hz:
            return False
        self._t_last = now
        name = f"{self.count:05d}_{int(now * 1000)}.jpg"
        d = os.path.join(self.root, self.session)
        try:
            self._q.put_nowait((d, name, raw_bgr.copy(), None if rect_bgr is None else rect_bgr.copy()))
        except queue.Full:
            self.dropped += 1
            return False
        self.count += 1
        return True

    def _writer(self) -> None:
        params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        while True:
            d, name, raw, rect = self._q.get()
            try:
                for sub, img in (("raw", raw), ("rect", rect)):
                    if img is not None:
                        p = os.path.join(d, sub, name)
                        cv2.imwrite(p, img, params)
                        own(p)
            except Exception as e:  # never kill the writer thread
                print(f"[webui] recorder write failed: {e}", flush=True)
            finally:
                self._q.task_done()

    def status(self) -> Dict[str, Any]:
        return {"active": self.active, "session": self.session, "count": self.count, "hz": self.hz,
                "dropped": self.dropped}

    # ------------------------------------------------------------------ sessions
    def _dir(self, name: str) -> str:
        if not _NAME_OK.match(name or ""):
            raise ValueError(f"bad session name {name!r}")
        d = os.path.join(self.root, name)
        if not os.path.isdir(d):
            raise FileNotFoundError(name)
        return d

    def list(self) -> List[Dict[str, Any]]:
        out = []
        for name in sorted(os.listdir(self.root), reverse=True):
            d = os.path.join(self.root, name)
            if not os.path.isdir(d) or not _NAME_OK.match(name):
                continue
            meta = {}
            try:
                with open(os.path.join(d, "meta.json")) as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                pass
            n_raw = _count_jpgs(os.path.join(d, "raw"))
            n_rect = _count_jpgs(os.path.join(d, "rect"))
            out.append({
                "name": name, "frames": max(n_raw, n_rect), "raw": n_raw, "rect": n_rect,
                "bytes": _dir_size(d), "meta": meta, "recording": self.active and name == self.session,
            })
        return out

    def delete(self, name: str) -> None:
        d = self._dir(name)
        if self.active and name == self.session:
            raise ValueError("stop recording before deleting this session")
        shutil.rmtree(d)

    def write_zip(self, name: str, out: BinaryIO) -> None:
        """Stream the session as a zip (stored, JPEGs don't compress) to a non-seekable stream."""
        d = self._dir(name)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
            for base, _, files in os.walk(d):
                for fn in sorted(files):
                    p = os.path.join(base, fn)
                    z.write(p, os.path.join(name, os.path.relpath(p, d)))

    def recorrect(self, name: str, rectify: Callable[[np.ndarray], np.ndarray], calib_meta: Optional[dict] = None) -> int:
        """Rebuild rect/ from raw/ with the current calibration. Returns the number of frames."""
        d = self._dir(name)
        if self.active and name == self.session:
            raise ValueError("stop recording before re-correcting this session")
        raw_dir, rect_dir = os.path.join(d, "raw"), os.path.join(d, "rect")
        os.makedirs(rect_dir, exist_ok=True)
        n = 0
        for fn in sorted(os.listdir(raw_dir)):
            if not fn.lower().endswith(".jpg"):
                continue
            img = cv2.imread(os.path.join(raw_dir, fn))
            if img is None:
                continue
            p = os.path.join(rect_dir, fn)
            cv2.imwrite(p, rectify(img), [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            own(p)
            n += 1
        mp = os.path.join(d, "meta.json")
        try:
            with open(mp) as f:
                meta = json.load(f)
        except (OSError, ValueError):
            meta = {}
        meta.update(recorrected=time.strftime("%Y-%m-%d %H:%M:%S"), calibration=calib_meta or meta.get("calibration"))
        with open(mp, "w") as f:
            json.dump(meta, f, indent=2)
        own(mp)
        return n


def _count_jpgs(d: str) -> int:
    try:
        return sum(1 for f in os.listdir(d) if f.lower().endswith(".jpg"))
    except OSError:
        return 0


def _dir_size(d: str) -> int:
    total = 0
    for base, _, files in os.walk(d):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(base, f))
            except OSError:
                pass
    return total
