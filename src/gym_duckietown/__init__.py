# coding=utf-8
__version__ = "6.2.0"

import platform

from zuper_commons.logs import ZLogger

from duckietown_world.resources import list_maps2

logger = ZLogger("gym-duckietown")
import os

import pyglet

on_mac = "Darwin" in platform.system()
# DUCKIETOWN_PYGLET_HEADLESS=0/1 overrides the platform default
# (e.g. set it to 0 on Linux to render through an X server such as Xvfb instead of EGL).
_headless_env = os.environ.get("DUCKIETOWN_PYGLET_HEADLESS")
if _headless_env is not None:
    pyglet.options["headless"] = _headless_env.strip().lower() in ("1", "true", "yes")
elif on_mac:
    pyglet.options["headless"] = False
else:
    pyglet.options["headless"] = True

path = os.path.dirname(os.path.dirname(__file__))
logger.debug(f"gym-duckietown version {__version__} path {path}\n")

from gym.envs.registration import register

from .utils import get_subdir_path


def reg_map_env(map_name0: str, map_file: str):
    gym_id = f"Duckietown-{map_name0}-v0"

    # logger.info('Registering gym environment id: %s' % gym_id)

    register(
        id=gym_id,
        entry_point="gym_duckietown.envs:DuckietownEnv",
        reward_threshold=400.0,
        kwargs={"map_name": map_file},
    )


for map_name, filename in list_maps2().items():
    # Register a gym environment for each map file available
    if "regress" not in filename:
        reg_map_env(map_name, filename)

register(id="MultiMap-v0", entry_point="gym_duckietown.envs:MultiMapEnv", reward_threshold=400.0)

register(id="Duckiebot-v0", entry_point="gym_duckietown.envs:DuckiebotEnv", reward_threshold=400.0)
