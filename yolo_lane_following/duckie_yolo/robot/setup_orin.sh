#!/usr/bin/env bash
# One-time setup of a Duckiebot with a Jetson Orin Nano (DB26J) - or a Jetson Nano (DB21) - for
# duckie_yolo. No ROS and no Duckietown software is used; every step that changes the system
# asks first.
#
#   cd gym-duckietown/yolo_lane_following && bash duckie_yolo/robot/setup_orin.sh
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # .../yolo_lane_following
ask() { read -r -p "$1 [y/N] " a; [[ "$a" =~ ^[Yy]$ ]]; }

echo "== duckie_yolo robot setup (package: $PKG_DIR)"

# 1. JetPack / L4T version -> which Ultralytics container to use
L4T="$(head -n1 /etc/nv_tegra_release 2>/dev/null || true)"
case "$L4T" in
  *R36*) JP=6 ;;
  *R35*) JP=5 ;;
  *R32*) JP=4 ;;
  *) JP=unknown ;;
esac
echo "-- L4T: ${L4T:-not a Jetson?}  -> JetPack $JP"
MODEL="$(tr -d '\0' </proc/device-tree/model 2>/dev/null || true)"
echo "-- board: ${MODEL:-unknown}"
I2C_BUS=1
[[ "$MODEL" == *Orin* ]] && I2C_BUS=7
echo "-- HAT I2C bus: $I2C_BUS (7 on Orin Nano / DB26J, 1 on Jetson Nano)"

# 2. Duckietown containers hold the camera (nvargus) and the motors (I2C): list them
if command -v docker >/dev/null 2>&1; then
  RUNNING="$(docker ps --format '{{.Names}}' | grep -Ei 'duckiebot-interface|car-interface|camera|wheels|dt-' || true)"
  if [[ -n "$RUNNING" ]]; then
    echo "-- these running containers may hold the camera / motors:"
    echo "$RUNNING" | sed 's/^/     /'
    if ask "   stop them and disable their auto-restart?"; then
      for c in $RUNNING; do docker update --restart=no "$c" >/dev/null; docker stop "$c"; done
    fi
  fi
fi

# 3. Native dependencies of robot_io (camera + motors)
NEED=()
python3 -c "import numpy" 2>/dev/null || NEED+=(python3-numpy)
python3 -c "import zmq" 2>/dev/null || NEED+=(python3-zmq)
python3 -c "import yaml" 2>/dev/null || NEED+=(python3-yaml)
command -v gst-launch-1.0 >/dev/null || NEED+=(gstreamer1.0-tools)
command -v i2cdetect >/dev/null || NEED+=(i2c-tools)
python3 -c "import cv2" 2>/dev/null || NEED+=(python3-opencv)
if (( ${#NEED[@]} )) && ask "-- apt-get install ${NEED[*]}?"; then
  sudo apt-get update && sudo apt-get install -y "${NEED[@]}"
fi
python3 -c "import smbus2" 2>/dev/null || { ask "-- pip install --user smbus2?" && python3 -m pip install --user smbus2; }
python3 -c "import Jetson.GPIO" 2>/dev/null || { ask "-- pip install --user Jetson.GPIO?" && python3 -m pip install --user Jetson.GPIO; }
if ! id -nG "$USER" | grep -qw i2c || ! id -nG "$USER" | grep -qw gpio; then
  if ask "-- add $USER to the i2c, gpio and video groups (needs re-login)?"; then
    sudo groupadd -f -r gpio
    sudo usermod -aG i2c,gpio,video "$USER"
  fi
fi
if ask "-- pip install --user --no-deps -e $PKG_DIR (duckie_yolo for robot_io)?"; then
  python3 -m pip install --user --no-deps -e "$PKG_DIR"
fi

# 4. Hardware check: the HAT's PCA9685 should answer at 0x60
if command -v i2cdetect >/dev/null; then
  if sudo i2cdetect -y -r "$I2C_BUS" 2>/dev/null | grep -qE '(^| )60( |$)'; then
    echo "-- PCA9685 found at 0x60 on bus $I2C_BUS"
  else
    echo "-- WARNING: no device at 0x60 on I2C bus $I2C_BUS (check the HAT / bus number)"
  fi
fi

# 5. Agent container (Ultralytics + TensorRT for this JetPack)
if [[ "$JP" == "5" || "$JP" == "6" ]] && command -v docker >/dev/null 2>&1; then
  if ask "-- build the agent container duckie-yolo-agent (base ultralytics:latest-jetson-jetpack$JP)?"; then
    docker build -t duckie-yolo-agent --build-arg JETPACK="$JP" -f "$PKG_DIR/duckie_yolo/robot/docker/Dockerfile.agent" "$PKG_DIR"
  fi
fi

# 6. Optional: start robot_io at boot
if ask "-- install a systemd service that runs robot_io at boot?"; then
  sudo tee /etc/systemd/system/duckie-robot-io.service >/dev/null <<EOF
[Unit]
Description=duckie_yolo robot_io (camera + wheels over ZMQ, no ROS)
After=network-online.target nvargus-daemon.service

[Service]
User=$USER
ExecStart=/usr/bin/python3 -m duckie_yolo.robot.robot_io --i2c-bus $I2C_BUS
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload && sudo systemctl enable --now duckie-robot-io
fi

cat <<EOF

== next steps
  1. wheels test (lift the robot!):  python3 -m duckie_yolo.robot.robot_io --i2c-bus $I2C_BUS --hw-test
  2. serve camera + wheels:          python3 -m duckie_yolo.robot.robot_io --i2c-bus $I2C_BUS
  3. calibrate (see README): intrinsics -> extrinsics -> wheels
  4. TensorRT engine + drive (inside the agent container):
       docker run --rm -it --runtime nvidia --network host --ipc host -v \$PWD:/work -w /work duckie-yolo-agent \\
         --weights models/best.engine --robot-config configs/robot_mybot.yaml --dry-run
EOF
