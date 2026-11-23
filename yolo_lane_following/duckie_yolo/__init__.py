"""duckie_yolo: YOLO-based lane following for Duckietown.

The core (perception, estimation, control, agent) depends only on numpy/OpenCV
(+ ultralytics for the YOLO segmenter). It never imports gym, ROS or any part of
the Duckietown robot software stack, so the exact same agent runs in
gym-duckietown and on a physical Duckiebot.
"""

from .classes import CLASS_NAMES, NUM_CLASSES

__version__ = "0.1.0"
__all__ = ["CLASS_NAMES", "NUM_CLASSES", "__version__"]
