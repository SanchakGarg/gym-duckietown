"""Camera sources for the real robot (no ROS).

* ``csi``  - Jetson CSI camera (IMX219 on DB21/DB26) through ``nvarguscamerasrc``.
             Uses OpenCV's GStreamer backend when available, otherwise reads raw BGR frames
             from a ``gst-launch-1.0 ... ! fdsink`` subprocess (works with pip OpenCV builds).
* ``usb:N`` - any V4L2/UVC camera via OpenCV.
* ``file:PATH`` - a video file or a folder of images (offline testing).
* ``mock`` - synthetic frames.

All sources run a capture thread and expose only the latest frame (no queueing latency).
"""
from __future__ import annotations

import glob
import os
import shlex
import subprocess
import threading
import time
from typing import Optional, Tuple

import numpy as np


def csi_pipeline(
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    sensor_id: int = 0,
    sensor_w: int = 1640,
    sensor_h: int = 1232,
    sensor_mode: int = -1,
    flip_method: int = 0,
    sink: str = "appsink",
) -> str:
    """GStreamer pipeline for an IMX219 on a Jetson. 1640x1232@30 is the full-FOV binned mode."""
    mode = f" sensor-mode={sensor_mode}" if sensor_mode >= 0 else ""
    src = (
        f"nvarguscamerasrc sensor-id={sensor_id}{mode} ! "
        f"video/x-raw(memory:NVMM),width={sensor_w},height={sensor_h},framerate={fps}/1,format=NV12 ! "
        f"nvvidconv flip-method={flip_method} ! video/x-raw,width={width},height={height},format=BGRx ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
    )
    if sink == "appsink":
        return src + "appsink drop=true max-buffers=1 sync=false"
    return src + "fdsink fd=1 sync=false"


class _Threaded:
    def __init__(self):
        self._lock = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._stamp = 0.0
        self._seq = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None

    def _grab(self) -> Optional[np.ndarray]:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def _open(self) -> None:
        pass

    def _close(self) -> None:
        pass

    def start(self) -> "_Threaded":
        self._open()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=type(self).__name__, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        fails = 0
        while self._running:
            try:
                f = self._grab()
            except Exception as e:  # keep the thread alive, report the error
                self.error = repr(e)
                f = None
            if f is None:
                fails += 1
                time.sleep(0.01 if fails < 100 else 0.2)
                continue
            fails = 0
            with self._lock:
                self._frame = f
                self._stamp = time.time()
                self._seq += 1
                self._lock.notify_all()

    def latest(self, after_seq: int = -1, timeout: float = 0.2) -> Tuple[Optional[np.ndarray], float, int]:
        """(BGR frame, unix time, sequence). Waits up to ``timeout`` for a frame newer than after_seq."""
        with self._lock:
            end = time.time() + timeout
            while self._seq <= after_seq or self._frame is None:
                rem = end - time.time()
                if rem <= 0:
                    break
                self._lock.wait(rem)
            return self._frame, self._stamp, self._seq

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._close()


def _cv2_has_gstreamer() -> bool:
    import cv2

    info = cv2.getBuildInformation()
    for line in info.splitlines():
        if "GStreamer" in line:
            return "YES" in line
    return False


class CSICamera(_Threaded):
    def __init__(self, width=640, height=480, fps=30, flip_method=0, sensor_id=0, sensor_mode=-1, backend="auto"):
        super().__init__()
        self.width, self.height, self.fps = width, height, fps
        self.kw = dict(width=width, height=height, fps=fps, flip_method=flip_method, sensor_id=sensor_id, sensor_mode=sensor_mode)
        self.backend = backend
        self._cap = None
        self._proc: Optional[subprocess.Popen] = None

    def _open(self) -> None:
        backend = self.backend
        if backend == "auto":
            backend = "opencv" if _cv2_has_gstreamer() else "pipe"
        self.backend = backend
        if backend == "opencv":
            import cv2

            self._cap = cv2.VideoCapture(csi_pipeline(sink="appsink", **self.kw), cv2.CAP_GSTREAMER)
            if not self._cap.isOpened():
                raise RuntimeError(
                    "could not open the CSI camera through GStreamer. Is another process (e.g. a Duckietown "
                    "container) holding it? Try: sudo systemctl restart nvargus-daemon"
                )
        else:
            cmd = "gst-launch-1.0 -q " + csi_pipeline(sink="fdsink", **self.kw)
            self._proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)

    def _grab(self) -> Optional[np.ndarray]:
        if self._cap is not None:
            ok, f = self._cap.read()
            return f if ok else None
        n = self.width * self.height * 3
        buf = bytearray()
        while len(buf) < n:
            chunk = self._proc.stdout.read(n - len(buf))
            if not chunk:
                raise RuntimeError("gst-launch exited (camera busy or pipeline error)")
            buf.extend(chunk)
        return np.frombuffer(bytes(buf), np.uint8).reshape(self.height, self.width, 3)

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(2)
            except subprocess.TimeoutExpired:
                self._proc.kill()


class USBCamera(_Threaded):
    def __init__(self, index=0, width=640, height=480, fps=30):
        super().__init__()
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self._cap = None

    def _open(self) -> None:
        import cv2

        self._cap = cv2.VideoCapture(self.index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open camera {self.index}")

    def _grab(self):
        import cv2

        ok, f = self._cap.read()
        if not ok:
            return None
        if f.shape[1] != self.width or f.shape[0] != self.height:
            f = cv2.resize(f, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return f

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()


class FileCamera(_Threaded):
    """Replays a video file or an image folder at ``fps`` (loops forever)."""

    def __init__(self, path: str, width=640, height=480, fps=30):
        super().__init__()
        self.path, self.width, self.height, self.fps = path, width, height, fps
        self._files = []
        self._i = 0
        self._cap = None

    def _open(self) -> None:
        import cv2

        if os.path.isdir(self.path):
            self._files = sorted(
                f for ext in ("*.jpg", "*.jpeg", "*.png") for f in glob.glob(os.path.join(self.path, ext))
            )
            if not self._files:
                raise RuntimeError(f"no images in {self.path}")
        else:
            self._cap = cv2.VideoCapture(self.path)

    def _grab(self):
        import cv2

        time.sleep(1.0 / self.fps)
        if self._files:
            f = cv2.imread(self._files[self._i % len(self._files)])
            self._i += 1
        else:
            ok, f = self._cap.read()
            if not ok:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, f = self._cap.read()
                if not ok:
                    return None
        return cv2.resize(f, (self.width, self.height), interpolation=cv2.INTER_AREA)


class MockCamera(_Threaded):
    def __init__(self, width=640, height=480, fps=30):
        super().__init__()
        self.width, self.height, self.fps = width, height, fps

    def _grab(self):
        time.sleep(1.0 / self.fps)
        f = np.full((self.height, self.width, 3), 60, np.uint8)
        f[self.height // 2 :, self.width // 2 - 8 : self.width // 2 + 8] = (0, 220, 255)  # BGR yellow stripe
        f[0:4, : (self._seq % self.width)] = 255
        return f


def open_camera(spec: str, width=640, height=480, fps=30, flip_method=0) -> _Threaded:
    """spec: csi | csi:<sensor_id> | usb:<index> | file:<path> | mock"""
    kind, _, arg = spec.partition(":")
    if kind == "csi":
        cam = CSICamera(width, height, fps, flip_method=flip_method, sensor_id=int(arg or 0))
    elif kind == "usb":
        cam = USBCamera(int(arg or 0), width, height, fps)
    elif kind == "file":
        cam = FileCamera(arg, width, height, fps)
    elif kind == "mock":
        cam = MockCamera(width, height, fps)
    else:
        raise ValueError(f"unknown camera spec {spec!r}")
    return cam.start()
