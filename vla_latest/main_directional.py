import os
import sys
LOCAL_METACAR_PATH = os.path.join(os.path.dirname(__file__), "autodrive_api_python-1.0.0")
if os.path.isdir(LOCAL_METACAR_PATH) and LOCAL_METACAR_PATH not in sys.path:
    sys.path.insert(0, LOCAL_METACAR_PATH)

import logging
import keyboard

from metacar import SceneAPI, DirectionalControl

logging.basicConfig(
    filename="vla_autodrive.log",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
    encoding="utf-8",
)
logger = logging.getLogger(__name__)


def get_direction_control_from_keyboard() -> DirectionalControl:
    dc = DirectionalControl()
    if keyboard.is_pressed("up") or keyboard.is_pressed("w"):
        dc.forward = 0.5 if keyboard.is_pressed("shift") else 1.0
    elif keyboard.is_pressed("down") or keyboard.is_pressed("s"):
        dc.backward = 0.5 if keyboard.is_pressed("shift") else 1.0
    if keyboard.is_pressed("left") or keyboard.is_pressed("a"):
        dc.steering_angle = -1.0
    elif keyboard.is_pressed("right") or keyboard.is_pressed("d"):
        dc.steering_angle = 1.0
    return dc


def main():
    api = SceneAPI()
    keyboard.add_hotkey("space", api.retry_level)
    keyboard.add_hotkey("n", api.skip_level)
    api.connect()
    api.get_scene_static_data()
    for sim_car_msg, frames in api.main_loop():
        dc = get_direction_control_from_keyboard()
        api.set_vehicle_direction_control(dc)


if __name__ == "__main__":
    main()
