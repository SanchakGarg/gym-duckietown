"""Automatic YOLO-seg labels from gym-duckietown's segmentation render.

``Simulator.render_obs(segment=True)`` renders the same camera view with:
  * asphalt / grass / floor            -> black
  * sky / background                   -> magenta (255, 0, 255)
  * lane markings                      -> their own colours: yellow (255,255,0), white (~245..255), red (255,0,0)
  * every object mesh                  -> a flat colour derived from a hash of the mesh name
    (``gen_segmentation_color`` in objmesh.py)

Caveats handled here:
  * duckiebots are not flat-coloured (their materials survive the segment render) and "duckie"
    and "duckiebot" even hash to the same colour: duckiebots are found by re-rendering with
    them hidden and taking the pixels that change.
  * some object colours are close to lane colours (bus/building ~ yellow, cone ~ red), so
    pixels matching any object colour are excluded from the lane classes.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from ..classes import DUCKIE, DUCKIEBOT, RED, WHITE, YELLOW

OBJECT_CLASS_BY_MESH = {"duckie": DUCKIE, "duckiebot": DUCKIEBOT}


def gen_segmentation_color(name: str) -> Tuple[int, int, int]:
    """Mirror of gym_duckietown.objmesh.ObjMesh.__init__.gen_segmentation_color."""
    hashed = "".join(str(ord(ch)) for ch in name)
    vals = [int(hashed[i : i + 3]) % 255 for i in range(0, len(hashed), 3)][:3]
    return tuple(vals)  # type: ignore[return-value]


def object_palette(mesh_names: Iterable[str]) -> Dict[str, Tuple[int, int, int]]:
    names = set(mesh_names) | {"duckie", "duckiebot"}
    return {n: gen_segmentation_color(n) for n in sorted(names)}


def env_mesh_names(env) -> List[str]:
    out = []
    for o in env.unwrapped.objects:
        mesh = getattr(o, "mesh", None)
        if mesh is not None and getattr(mesh, "mesh_name", None):
            out.append(mesh.mesh_name)
    return out


def color_match(img: np.ndarray, color: Tuple[int, int, int], tol: int = 6) -> np.ndarray:
    d = np.abs(img.astype(np.int16) - np.array(color, np.int16)[None, None, :]).max(axis=-1)
    return d <= tol


def seg_to_label_map(seg_rgb: np.ndarray, palette: Dict[str, Tuple[int, int, int]], tol: int = 6) -> np.ndarray:
    """(H, W) int16 label map; -1 = background / ignored objects.

    Lane markings keep the colours of the (photo/synthetic) tile texture in the segmentation
    render, with asphalt blacked out, so they are classified by hue/saturation/value.
    Thresholds were measured on segment renders of several maps, both styles, with and
    without domain randomization (white: S<=30, V>=218; yellow: H 23-31, S>=77; red: H>=176
    or <=2, S>=140). Anti-aliased edges are kept down to roughly half coverage.
    """
    img = seg_rgb.astype(np.int16)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    lm = np.full(seg_rgb.shape[:2], -1, np.int16)
    obj_any = np.zeros(seg_rgb.shape[:2], bool)
    for name, col in palette.items():
        m = color_match(seg_rgb, col, tol)
        obj_any |= m
        if name in OBJECT_CLASS_BY_MESH:
            lm[m] = DUCKIE  # duckiebots are split off later (identical colour)
    # objects whose hash colour resembles a marking (bus/building ~ yellow, cone ~ red):
    # also exclude their anti-aliased borders
    obj_zone = cv2.dilate(obj_any.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
    magenta = (R > 200) & (G < 60) & (B > 200)
    free = ~obj_zone & ~magenta
    hsv = cv2.cvtColor(np.ascontiguousarray(seg_rgb), cv2.COLOR_RGB2HSV)
    Hh, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    yellow = free & (Hh >= 18) & (Hh <= 36) & (S >= 60) & (V >= 100)
    red = free & ((Hh <= 8) | (Hh >= 170)) & (S >= 100) & (V >= 110)
    white = free & (S <= 40) & (V >= 150)
    lm[yellow] = YELLOW
    lm[red] = RED
    lm[white] = WHITE
    return lm


def render_label_map(env, palette: Optional[Dict[str, Tuple[int, int, int]]] = None, tol: int = 6) -> np.ndarray:
    """Render the segmentation view of the current pose and convert it to a label map."""
    u = env.unwrapped
    palette = palette or object_palette(env_mesh_names(env))
    seg = u.render_obs(segment=True).copy()
    lm = seg_to_label_map(seg, palette, tol)
    # Duckiebots keep their material colours in the segment render (dark red chassis, striped
    # battery, a bluish duck on top), so they are found by re-rendering with them hidden: every
    # pixel that changes belongs to a duckiebot (the render is otherwise deterministic).
    bots = [o for o in u.objects if o.kind == "duckiebot" and getattr(o, "visible", True)]
    if bots:
        for o in bots:
            o.visible = False
        try:
            seg_nobot = u.render_obs(segment=True)
        finally:
            for o in bots:
                o.visible = True
        lm[np.any(seg != seg_nobot, axis=-1)] = DUCKIEBOT
    return lm


def clean_label_map(lm: np.ndarray, kernel: int = 3) -> np.ndarray:
    """Remove isolated speckles per class (morphological open)."""
    out = np.full_like(lm, -1)
    k = np.ones((kernel, kernel), np.uint8)
    for c in np.unique(lm):
        if c < 0:
            continue
        m = cv2.morphologyEx((lm == c).astype(np.uint8), cv2.MORPH_OPEN, k)
        out[m.astype(bool)] = c
    return out


def label_map_to_polygons(
    lm: np.ndarray, min_area: float = 25.0, eps_px: float = 1.0, max_horizon_row: Optional[int] = None
) -> List[Tuple[int, np.ndarray]]:
    """Instance polygons (class id, (N,2) pixel coords) from a label map, one per connected component."""
    h, w = lm.shape
    out = []
    for c in np.unique(lm):
        if c < 0:
            continue
        m = (lm == c).astype(np.uint8)
        if max_horizon_row:
            m[:max_horizon_row] = 0
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            poly = cv2.approxPolyDP(cnt, eps_px, True).reshape(-1, 2)
            if len(poly) >= 3:
                out.append((int(c), poly.astype(np.float32)))
    return out


def polygons_to_yolo_lines(polys: List[Tuple[int, np.ndarray]], width: int, height: int) -> List[str]:
    lines = []
    for c, p in polys:
        q = p.copy()
        q[:, 0] = np.clip(q[:, 0] / width, 0, 1)
        q[:, 1] = np.clip(q[:, 1] / height, 0, 1)
        lines.append(" ".join([str(c)] + [f"{v:.5f}" for v in q.reshape(-1)]))
    return lines


def yolo_lines_to_label_map(lines: List[str], width: int, height: int) -> np.ndarray:
    """Rasterize YOLO-seg label lines back to a label map (for previews / round-trip tests)."""
    lm = np.full((height, width), -1, np.int16)
    for line in lines:
        vals = line.split()
        if len(vals) < 7:
            continue
        c = int(vals[0])
        p = np.array(vals[1:], np.float32).reshape(-1, 2) * np.array([width, height], np.float32)
        m = np.zeros((height, width), np.uint8)
        cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
        lm[m.astype(bool)] = c
    return lm
