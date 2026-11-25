"""Small helpers shared by the web UI modules."""
from __future__ import annotations

import os
import re
import shutil
from typing import Optional, Tuple

_owner: Optional[Tuple[int, int]] = None


def _target_owner() -> Optional[Tuple[int, int]]:
    """uid/gid that files we create should belong to.

    The web UI runs as root inside its container (GPU, TensorRT and Ultralytics' own installs
    just work that way), but its files land in the user's bind-mounted repo. They are handed
    to $DUCKIE_UID:$DUCKIE_GID (set by start_webui.sh) or the owner of the working directory.
    """
    global _owner
    if _owner is None:
        if not hasattr(os, "geteuid") or os.geteuid() != 0:
            _owner = (-1, -1)
        else:
            try:
                uid = int(os.environ["DUCKIE_UID"])
                gid = int(os.environ.get("DUCKIE_GID", uid))
            except (KeyError, ValueError):
                st = os.stat(os.getcwd())
                uid, gid = st.st_uid, st.st_gid
            _owner = (uid, gid)
    return None if _owner == (-1, -1) or _owner == (0, 0) else _owner


def own(path: str, recursive: bool = False) -> None:
    """chown a file/folder we created to the repo owner (no-op unless running as root)."""
    o = _target_owner()
    if o is None:
        return
    try:
        os.chown(path, *o)
        if recursive and os.path.isdir(path):
            for base, dirs, files in os.walk(path):
                for n in dirs + files:
                    os.chown(os.path.join(base, n), *o)
    except OSError:
        pass


def disk_free_gb(path: str) -> float:
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return -1.0


_FILE_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,120}$")


def safe_filename(name: str, exts: Tuple[str, ...]) -> str:
    """Validate a bare file name (no folders) with one of the allowed extensions."""
    base = os.path.basename(name or "")
    if base != name or not _FILE_OK.match(base) or not base.lower().endswith(exts):
        raise ValueError(f"invalid file name {name!r} (allowed: {', '.join(exts)})")
    return base
