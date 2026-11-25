"""Calibration from the browser: camera lens (intrinsics), floor (extrinsics) and wheels.

Thin glue around the same functions the command-line calibration scripts use, so both write
identical robot config files.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

from ..calibrate_extrinsics import compute_extrinsics
from ..calibrate_intrinsics import ViewSelector, calibrate_images, find_board, image_files
from ..calibrate_wheels import apply_straight, omega_gain_from
from ..robot_config import RobotConfig
from .util import own


class Calibration:
    def __init__(self, controller, calib_root: str = "calib"):
        self.ctl = controller
        self.root = calib_root
        self._lock = threading.Lock()
        self.capture: Optional[Dict[str, Any]] = None  # lens view capture state
        self.results: Dict[str, Any] = {}

    # ------------------------------------------------------------------ files shown in the page
    def image_names(self) -> Dict[str, str]:
        stem = os.path.splitext(os.path.basename(self.ctl.robot_config_path))[0]
        return {"intrinsics": f"{stem}_rectified_sample.jpg", "extrinsics": f"{stem}_extrinsics_check.jpg",
                "extrinsics_failed": f"{stem}_extrinsics_failed.jpg"}

    def image_path(self, name: str) -> str:
        if name not in self.image_names().values():
            raise FileNotFoundError(name)
        return os.path.join(os.path.dirname(os.path.abspath(self.ctl.robot_config_path)), name)

    # ------------------------------------------------------------------ lens: capture views
    def intrinsics_start(self, cols: int = 7, rows: int = 5, target: int = 30) -> Dict[str, Any]:
        with self._lock:
            if self.capture is not None and self.capture["active"]:
                return self.capture_status()
            d = os.path.join(self.root, "intr_" + time.strftime("%Y%m%d-%H%M%S"))
            os.makedirs(d, exist_ok=True)
            own(self.root)
            own(d)
            self.capture = {"active": True, "dir": d, "cols": int(cols), "rows": int(rows), "target": int(target),
                            "count": 0, "found": False, "corners": None, "t_seen": 0.0}
        self.ctl.overlay_fn = self._draw_capture
        threading.Thread(target=self._capture_worker, args=(self.capture,), daemon=True, name="intr-capture").start()
        return self.capture_status()

    def intrinsics_stop(self) -> None:
        with self._lock:
            if self.capture is not None:
                self.capture["active"] = False
        self.ctl.overlay_fn = None

    def _capture_worker(self, cap: Dict[str, Any]) -> None:
        pattern = (cap["cols"], cap["rows"])
        selector = ViewSelector()
        last_seq = -1
        while cap["active"] and cap["count"] < cap["target"]:
            raw, seq = self.ctl.latest_raw()
            if raw is None or seq == last_seq:
                time.sleep(0.03)
                continue
            last_seq = seq
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            # cheap rejection first: the full detector is slow on the Nano when no board is visible
            ok, _ = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_FAST_CHECK)
            corners = find_board(gray, pattern) if ok else None
            cap["found"], cap["corners"] = corners is not None, corners
            if corners is not None:
                cap["t_seen"] = time.time()
                if selector.accept(corners):
                    p = os.path.join(cap["dir"], f"intr_{cap['count']:03d}.png")
                    cv2.imwrite(p, raw)
                    own(p)
                    cap["count"] += 1
        cap["active"] = False
        if self.ctl.overlay_fn == self._draw_capture:
            self.ctl.overlay_fn = None

    def _draw_capture(self, img: np.ndarray) -> np.ndarray:
        cap = self.capture
        if cap is None:
            return img
        corners = cap.get("corners")
        fresh = corners is not None and time.time() - cap.get("t_seen", 0) < 0.5
        if fresh:
            cv2.drawChessboardCorners(img, (cap["cols"], cap["rows"]), corners, True)
        msg = f"lens calibration: {cap['count']}/{cap['target']} views  " + ("board found" if fresh else "no board")
        cv2.putText(img, msg, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, msg, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if fresh else (0, 200, 255), 1, cv2.LINE_AA)
        return img

    def capture_status(self) -> Optional[Dict[str, Any]]:
        cap = self.capture
        if cap is None:
            return None
        return {k: cap[k] for k in ("active", "dir", "cols", "rows", "target", "count", "found")}

    # ------------------------------------------------------------------ lens: compute
    def intrinsics_compute(self, square: float = 0.031, model: str = "fisheye", fov: float = 75.0) -> Dict[str, Any]:
        cap = self.capture
        if cap is None:
            raise ValueError("capture checkerboard views first")
        self.intrinsics_stop()
        files = image_files(cap["dir"])
        r = calibrate_images(files, self.ctl.robot_config_path, cap["cols"], cap["rows"], float(square), model, float(fov))
        own(self.ctl.robot_config_path)
        own(r["sample"])
        self.ctl.reload_robot_config()
        res = {"rms": round(r["rms"], 3), "views": r["views"], "files": r["files"], "image": os.path.basename(r["sample"]),
               "t": time.time()}
        self.results["intrinsics"] = res
        return res

    # ------------------------------------------------------------------ floor
    def extrinsics(self, x0: float, y0: float, cols: int = 7, rows: int = 5, square: float = 0.031) -> Dict[str, Any]:
        raw, _ = self.ctl.latest_raw()
        if raw is None:
            raise ValueError("no camera frame yet")
        robot = RobotConfig.load(self.ctl.robot_config_path)
        names = self.image_names()
        try:
            vis, mean_err, max_err = compute_extrinsics(raw.copy(), robot, int(cols), int(rows), float(square), float(x0), float(y0))
        except ValueError as e:
            p = self.image_path(names["extrinsics_failed"])
            cv2.imwrite(p, robot.camera_model().rectify(raw))
            own(p)
            self.results["extrinsics"] = {"error": str(e), "image": names["extrinsics_failed"], "t": time.time()}
            raise
        robot.save(self.ctl.robot_config_path)
        own(self.ctl.robot_config_path)
        p = self.image_path(names["extrinsics"])
        cv2.imwrite(p, vis)
        own(p)
        self.ctl.reload_robot_config()
        res = {"mean_mm": round(1000 * mean_err, 1), "max_mm": round(1000 * max_err, 1), "image": names["extrinsics"],
               "intrinsics_calibrated": bool(robot.calibrated.get("intrinsics")), "t": time.time()}
        self.results["extrinsics"] = res
        return res

    # ------------------------------------------------------------------ wheels
    def wheels_run(self, kind: str, v: float = 0.25, omega: float = 2.0, duration: float = 3.0, delay: float = 2.0) -> None:
        if kind == "straight":
            self.ctl.start_motion(float(v), 0.0, float(duration), delay)
        elif kind == "spin":
            self.ctl.start_motion(0.0, float(omega), float(duration), delay)
        else:
            raise ValueError("kind must be 'straight' or 'spin'")

    def wheels_apply(self, kind: str, v: float = 0.25, omega: float = 2.0, duration: float = 3.0,
                     drift: Optional[float] = None, distance: Optional[float] = None,
                     turned_deg: Optional[float] = None) -> Dict[str, Any]:
        robot = RobotConfig.load(self.ctl.robot_config_path)
        if kind == "straight":
            if drift is None and distance is None:
                raise ValueError("enter the drift and/or the distance you measured")
            apply_straight(robot.wheels, float(v), float(duration), drift, distance)
            robot.calibrated["wheels"] = True
            res = {"gain": round(robot.wheels.gain, 4), "trim": round(robot.wheels.trim, 4)}
        elif kind == "spin":
            if turned_deg is None or turned_deg <= 0:
                raise ValueError("enter the angle the robot turned (degrees)")
            robot.actuation.omega_gain = omega_gain_from(float(turned_deg), float(omega), float(duration))
            res = {"omega_gain": round(robot.actuation.omega_gain, 4)}
        else:
            raise ValueError("kind must be 'straight' or 'spin'")
        robot.save(self.ctl.robot_config_path)
        own(self.ctl.robot_config_path)
        self.ctl.reload_robot_config()
        self.results["wheels_" + kind] = dict(res, t=time.time())
        return res

    def status(self) -> Dict[str, Any]:
        return {"capture": self.capture_status(), "results": self.results}


def checkerboard_svg(inner_cols: int = 7, inner_rows: int = 5, square_mm: float = 31.0) -> str:
    """Printable checkerboard with real-world units (print at 100 % / "actual size")."""
    nx, ny = inner_cols + 1, inner_rows + 1
    page_w, page_h = 297.0, 210.0  # A4 landscape, mm
    bw, bh = nx * square_mm, ny * square_mm
    ox, oy = (page_w - bw) / 2, (page_h - bh) / 2 + 4
    rects = []
    for j in range(ny):
        for i in range(nx):
            if (i + j) % 2 == 0:
                rects.append(f'<rect x="{ox + i * square_mm:.2f}" y="{oy + j * square_mm:.2f}" '
                             f'width="{square_mm}" height="{square_mm}"/>')
    note = (f"{nx} x {ny} squares ({inner_cols} x {inner_rows} inner corners), {square_mm:g} mm each. "
            f"Print at 100% (actual size), then check that one square measures {square_mm:g} mm.")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{page_w}mm" height="{page_h}mm" '
            f'viewBox="0 0 {page_w} {page_h}"><rect width="100%" height="100%" fill="#fff"/>'
            f'<g fill="#000">{"".join(rects)}</g>'
            f'<text x="{page_w / 2}" y="{oy - 4:.1f}" font-family="sans-serif" font-size="4" text-anchor="middle">'
            f'{note}</text></svg>')
