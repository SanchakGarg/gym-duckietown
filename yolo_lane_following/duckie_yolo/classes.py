"""Class definitions shared by dataset generation, training and inference."""

YELLOW = 0  # yellow dashed center line
WHITE = 1  # solid white edge line
RED = 2  # red stop line
DUCKIE = 3
DUCKIEBOT = 4

CLASS_NAMES = ["yellow_lane", "white_lane", "red_stop", "duckie", "duckiebot"]
NUM_CLASSES = len(CLASS_NAMES)

LANE_CLASSES = (YELLOW, WHITE, RED)
OBSTACLE_CLASSES = (DUCKIE, DUCKIEBOT)

# BGR colors used only for visualisation overlays
VIZ_BGR = {
    YELLOW: (0, 215, 255),
    WHITE: (255, 255, 255),
    RED: (40, 40, 230),
    DUCKIE: (0, 140, 255),
    DUCKIEBOT: (255, 120, 40),
}
