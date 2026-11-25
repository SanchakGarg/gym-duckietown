"""Tiny MJPEG-over-HTTP debug stream (stdlib only) to watch a headless robot from a browser."""
from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import numpy as np

PAGE = b"""<!doctype html><html><head><title>duckie_yolo</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{margin:0;background:#111;color:#ddd;font-family:sans-serif}img{width:100%;max-width:1100px;display:block;margin:auto}</style>
</head><body><img src="/stream"></body></html>"""


class FrameBuffer:
    """Latest JPEG frame shared between one producer and many HTTP stream readers."""

    def __init__(self, quality: int = 70):
        self.quality = quality
        self._jpeg: Optional[bytes] = None
        self._cond = threading.Condition()

    def publish(self, bgr: np.ndarray) -> None:
        import cv2

        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            self.publish_jpeg(buf.tobytes())

    def publish_jpeg(self, jpg: bytes) -> None:
        with self._cond:
            self._jpeg = jpg
            self._cond.notify_all()

    def latest(self) -> Optional[bytes]:
        with self._cond:
            return self._jpeg

    def wait_new(self, last: Optional[bytes], timeout: float) -> Optional[bytes]:
        with self._cond:
            if self._jpeg is last:
                self._cond.wait(timeout)
            return None if self._jpeg is last else self._jpeg


class MJPEGServer(FrameBuffer):
    def __init__(self, port: int = 8080, host: str = "0.0.0.0", quality: int = 70, max_fps: float = 15.0):
        super().__init__(quality)
        self.min_dt = 1.0 / max_fps
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(PAGE)
                elif self.path.startswith("/snapshot"):
                    jpg = server.latest()
                    if jpg is None:
                        self.send_error(503)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.end_headers()
                    self.wfile.write(jpg)
                elif self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    last = None
                    try:
                        while True:
                            jpg = server.wait_new(last, 2.0)
                            if jpg is None:
                                continue
                            last = jpg
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                            time.sleep(server.min_dt)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                else:
                    self.send_error(404)

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
