"""Wire protocol between robot_io (on the robot) and its clients.

Compatible with gym-duckietown's ``Duckiebot-v0`` env (src/gym_duckietown/envs/duckiebot_env.py):
ZMQ PAIR socket (port 5555 here; Duckiebot-v0 itself expects 7777, so run robot_io with
``--port 7777`` for that env); the client sends JSON commands, the server answers each
``reset`` / ``action`` with a JSON header followed by the image buffer.

Requests:   {"command": "reset" | "action" | "stop" | "ping",
             "values": [left_duty, right_duty],          # action only, each in [-1, 1]
             "encoding": "raw" | "jpeg", "quality": 80,  # optional (default raw BGR)
             "after_seq": n,                             # optional: wait for a frame newer than n
             "req_id": k}                                # optional: echoed back
Replies:    {"dtype": "uint8", "shape": [h, w, 3], "encoding": ..., "seq": n, "stamp": t,
             "req_id": k, "wheels": [l, r]}  + one binary frame (raw BGR bytes or JPEG)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

DEFAULT_PORT = 5555


def encode_frame(frame_bgr: np.ndarray, encoding: str = "raw", quality: int = 80) -> Tuple[dict, bytes]:
    md = {"dtype": str(frame_bgr.dtype), "shape": list(frame_bgr.shape), "encoding": encoding}
    if encoding == "jpeg":
        import cv2

        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return md, buf.tobytes()
    md["encoding"] = "raw"
    return md, np.ascontiguousarray(frame_bgr).tobytes()


def decode_frame(md: dict, payload: bytes) -> np.ndarray:
    if md.get("encoding") == "jpeg":
        import cv2

        img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("JPEG decoding failed")
        return img
    return np.frombuffer(payload, dtype=md.get("dtype", "uint8")).reshape(md["shape"])
