"""HTTP side of the web UI (stdlib only): the page, the MJPEG stream and a small JSON API."""
from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from .calibration import Calibration, checkerboard_svg
from .controller import RobotController
from .models import COLOR, JobRunner, ModelManager, ultralytics_available
from .util import own

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png",
                 ".jpg": "image/jpeg"}


class HTTPError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


class WebUI:
    """Ties the controller, model manager, recorder, calibration and jobs to HTTP routes."""

    def __init__(self, controller: RobotController, calibration: Calibration, jobs: JobRunner, workdir: str = "."):
        self.ctl = controller
        self.models: ModelManager = controller.models
        self.recorder = controller.recorder
        self.calib = calibration
        self.jobs = jobs
        self.workdir = workdir
        R = re.compile
        self.routes: List[Tuple[str, Any, Callable]] = [
            ("GET", R(r"^/$"), self.page),
            ("GET", R(r"^/static/([A-Za-z0-9_.-]+)$"), self.static),
            ("GET", R(r"^/stream\.mjpg$"), None),  # handled by the handler (long-lived)
            ("GET", R(r"^/snapshot\.jpg$"), self.snapshot),
            ("GET", R(r"^/checkerboard\.svg$"), self.checkerboard),
            ("GET", R(r"^/calib/([A-Za-z0-9_.-]+)$"), self.calib_image),
            ("GET", R(r"^/api/status$"), self.status),
            ("POST", R(r"^/api/drive$"), self.drive),
            ("POST", R(r"^/api/stop$"), self.stop),
            ("POST", R(r"^/api/heartbeat$"), self.heartbeat),
            ("POST", R(r"^/api/view$"), self.view),
            ("POST", R(r"^/api/autodrive$"), self.autodrive),
            ("GET", R(r"^/api/models$"), self.list_models),
            ("PUT", R(r"^/api/models/upload$"), None),  # streamed body, handled by the handler
            ("DELETE", R(r"^/api/models/([A-Za-z0-9_.-]+)$"), self.delete_model),
            ("POST", R(r"^/api/model/start$"), self.model_start),
            ("POST", R(r"^/api/model/stop$"), self.model_stop),
            ("POST", R(r"^/api/model/conf$"), self.model_conf),
            ("POST", R(r"^/api/engine/build$"), self.engine_build),
            ("POST", R(r"^/api/job/cancel$"), self.job_cancel),
            ("GET", R(r"^/api/job$"), self.job),
            ("POST", R(r"^/api/record/start$"), self.record_start),
            ("POST", R(r"^/api/record/stop$"), self.record_stop),
            ("GET", R(r"^/api/sessions$"), self.sessions),
            ("GET", R(r"^/api/sessions/([A-Za-z0-9_.-]+)\.zip$"), None),  # streamed zip
            ("POST", R(r"^/api/sessions/([A-Za-z0-9_.-]+)/recorrect$"), self.session_recorrect),
            ("DELETE", R(r"^/api/sessions/([A-Za-z0-9_.-]+)$"), self.session_delete),
            ("POST", R(r"^/api/calib/intrinsics/start$"), self.intr_start),
            ("POST", R(r"^/api/calib/intrinsics/stop$"), self.intr_stop),
            ("POST", R(r"^/api/calib/intrinsics/compute$"), self.intr_compute),
            ("POST", R(r"^/api/calib/extrinsics$"), self.extrinsics),
            ("POST", R(r"^/api/calib/wheels/run$"), self.wheels_run),
            ("POST", R(r"^/api/calib/wheels/apply$"), self.wheels_apply),
        ]

    # ------------------------------------------------------------------ pages and files
    def page(self, req, body):
        return self._file(os.path.join(STATIC_DIR, "index.html"))

    def static(self, req, body, name):
        return self._file(os.path.join(STATIC_DIR, name))

    @staticmethod
    def _file(path: str):
        if not os.path.isfile(path):
            raise HTTPError(404, "not found")
        with open(path, "rb") as f:
            data = f.read()
        return 200, CONTENT_TYPES.get(os.path.splitext(path)[1].lower(), "application/octet-stream"), data

    def snapshot(self, req, body):
        jpg = self.ctl.frames.latest()
        if jpg is None:
            raise HTTPError(503, "no frame yet")
        return 200, "image/jpeg", jpg

    def checkerboard(self, req, body):
        q = req.query
        svg = checkerboard_svg(int(q.get("cols", 7)), int(q.get("rows", 5)), float(q.get("square_mm", 31)))
        return 200, "image/svg+xml", svg.encode()

    def calib_image(self, req, body, name):
        try:
            return self._file(self.calib.image_path(name))
        except FileNotFoundError:
            raise HTTPError(404, "not found")

    # ------------------------------------------------------------------ status and driving
    def status(self, req, body):
        s = self.ctl.status()
        s["job"] = self.jobs.status()
        s["calib"] = self.calib.status()
        s["ultralytics"] = ultralytics_available()
        return s

    def drive(self, req, body):
        self.ctl.teleop(float(body.get("v", 0.0)), float(body.get("w", 0.0)))
        return {"ok": True}

    def stop(self, req, body):
        self.ctl.stop(body.get("reason") or "STOP pressed")
        return {"ok": True}

    def heartbeat(self, req, body):
        self.ctl.heartbeat()
        return {"ok": True}

    def view(self, req, body):
        self.ctl.set_view(str(body.get("view", "")))
        return {"ok": True}

    def autodrive(self, req, body):
        self.ctl.set_autodrive(bool(body.get("enabled")), body.get("max_speed"))
        return {"ok": True}

    # ------------------------------------------------------------------ models
    def list_models(self, req, body):
        return {"models": self.models.list(), "ultralytics": ultralytics_available()}

    def delete_model(self, req, body, name):
        self.models.delete(unquote(name))
        return {"ok": True}

    def model_start(self, req, body):
        if self.jobs.running():
            raise HTTPError(409, "wait for the running job to finish")
        name = str(body.get("name", ""))
        conf = body.get("conf")
        self.ctl.set_autodrive(False)
        self.models.start(name, self.ctl.camera, self.ctl.robot.actuation, None if conf is None else float(conf))
        return {"ok": True}

    def model_stop(self, req, body):
        self.ctl.set_autodrive(False)
        self.models.stop()
        return {"ok": True}

    def model_conf(self, req, body):
        self.models.set_conf(float(body["conf"]))
        return {"ok": True}

    def engine_build(self, req, body):
        name = str(body.get("name", ""))
        if not name.lower().endswith((".pt", ".onnx")):
            raise HTTPError(400, "choose a .pt model to build a TensorRT engine from")
        path = self.models.path(name)
        if not os.path.exists(path):
            raise HTTPError(404, f"{name} not found")
        if not ultralytics_available():
            raise HTTPError(400, "Ultralytics is not installed here: run the web UI in the Docker container")
        # TensorRT needs every bit of the Nano's 4 GB: unload the running model first
        self.ctl.set_autodrive(False)
        self.models.stop()
        cmd = [sys.executable, "-m", "duckie_yolo.robot.export_engine", "--weights", path,
               "--imgsz", str(int(body.get("imgsz", 320))), "--benchmark", "30"]
        self.jobs.start(f"TensorRT engine from {name}", cmd, cwd=self.workdir,
                        on_done=lambda rc: own(self.models.models_dir, recursive=True))
        return {"ok": True}

    def job(self, req, body):
        return self.jobs.status(with_log=True)

    def job_cancel(self, req, body):
        self.jobs.cancel()
        return {"ok": True}

    # ------------------------------------------------------------------ recording
    def record_start(self, req, body):
        r = self.ctl.robot
        meta = {"calibration": {"file": self.ctl.robot_config_path, "calibrated": dict(r.calibrated)},
                "model": self.models.name}
        sess = self.recorder.start(str(body.get("name", "")), float(body.get("hz", 3.0)), meta)
        return {"ok": True, "session": sess}

    def record_stop(self, req, body):
        self.recorder.stop()
        return {"ok": True}

    def sessions(self, req, body):
        return {"sessions": self.recorder.list()}

    def session_delete(self, req, body, name):
        self.recorder.delete(name)
        return {"ok": True}

    def session_recorrect(self, req, body, name):
        r = self.ctl.robot
        n = self.recorder.recorrect(name, self.ctl.rectify, {"file": self.ctl.robot_config_path,
                                                             "calibrated": dict(r.calibrated)})
        return {"ok": True, "frames": n}

    # ------------------------------------------------------------------ calibration
    def intr_start(self, req, body):
        self.ctl.set_view("raw")
        return {"ok": True, "capture": self.calib.intrinsics_start(int(body.get("cols", 7)), int(body.get("rows", 5)),
                                                                   int(body.get("target", 30)))}

    def intr_stop(self, req, body):
        self.calib.intrinsics_stop()
        return {"ok": True}

    def intr_compute(self, req, body):
        return self.calib.intrinsics_compute(float(body.get("square", 0.031)), str(body.get("model", "fisheye")))

    def extrinsics(self, req, body):
        return self.calib.extrinsics(float(body["x0"]), float(body["y0"]), int(body.get("cols", 7)),
                                     int(body.get("rows", 5)), float(body.get("square", 0.031)))

    def wheels_run(self, req, body):
        self.calib.wheels_run(str(body.get("kind")), float(body.get("v", 0.25)), float(body.get("omega", 2.0)),
                              float(body.get("duration", 3.0)))
        return {"ok": True}

    def wheels_apply(self, req, body):
        def opt(k):
            v = body.get(k)
            return None if v in (None, "") else float(v)

        return self.calib.wheels_apply(str(body.get("kind")), float(body.get("v", 0.25)), float(body.get("omega", 2.0)),
                                       float(body.get("duration", 3.0)), opt("drift"), opt("distance"), opt("turned_deg"))


def make_handler(app: WebUI):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "duckie-webui"

        def log_message(self, fmt, *args):  # keep the console for robot events
            pass

        # -------------------------------------------------------------- plumbing
        def _route(self, method: str):
            url = urlparse(self.path)
            self.query = {k: v[-1] for k, v in parse_qs(url.query).items()}
            for m, rx, fn in app.routes:
                if m != method:
                    continue
                mt = rx.match(url.path)
                if mt:
                    return fn, mt.groups(), url.path
            raise HTTPError(404, f"no route {method} {url.path}")

        def _send(self, code: int, ctype: str, data: bytes, extra: Optional[Dict[str, str]] = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def _json(self, code: int, obj: Any):
            self._send(code, "application/json", json.dumps(obj).encode())

        def _body(self) -> Dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return {}
            raw = self.rfile.read(n)
            try:
                d = json.loads(raw.decode() or "{}")
            except ValueError:
                raise HTTPError(400, "body must be JSON")
            if not isinstance(d, dict):
                raise HTTPError(400, "body must be a JSON object")
            return d

        def _handle(self, method: str):
            try:
                fn, groups, path = self._route(method)
                if fn is None:
                    return self._special(method, path, groups)
                body = self._body() if method in ("POST", "PUT", "DELETE") else {}
                res = fn(self, body, *groups)
                if isinstance(res, tuple):
                    self._send(*res)
                else:
                    self._json(200, res)
            except HTTPError as e:
                self._json(e.code, {"error": str(e)})
            except (ValueError, KeyError, FileNotFoundError, RuntimeError) as e:
                self._json(400, {"error": str(e) or type(e).__name__})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                traceback.print_exc()
                try:
                    self._json(500, {"error": f"{type(e).__name__}: {e}"})
                except Exception:
                    pass

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

        def do_PUT(self):
            self._handle("PUT")

        def do_DELETE(self):
            self._handle("DELETE")

        # -------------------------------------------------------------- long-lived responses
        def _special(self, method: str, path: str, groups):
            if path == "/stream.mjpg":
                return self._stream()
            if path == "/api/models/upload":
                n = int(self.headers.get("Content-Length") or 0)
                if n <= 0:
                    raise HTTPError(400, "empty upload")
                name = self.query.get("name", "")
                dst = app.models.save_upload(name, self.rfile, n)
                return self._json(200, {"ok": True, "name": os.path.basename(dst)})
            if path.endswith(".zip"):
                return self._zip(groups[0])
            raise HTTPError(404, "not found")

        def _stream(self):
            ctl = app.ctl
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            ctl.viewer_enter()
            last = None
            try:
                while True:
                    jpg = ctl.frames.wait_new(last, 2.0)
                    if jpg is None:
                        continue
                    last = jpg
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    time.sleep(0.5 / ctl.stream_fps)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                ctl.viewer_exit()

        def _zip(self, name: str):
            app.recorder._dir(name)  # validates before headers go out
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{name}.zip"')
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            app.recorder.write_zip(name, self.wfile)

    return Handler


def serve(app: WebUI, host: str = "0.0.0.0", port: int = 5555) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    httpd.daemon_threads = True
    return httpd
