#!/usr/bin/env bash
# One-time setup of a Duckiebot Jetson for duckie_yolo: Jetson Nano (DB21, JetPack 4) or
# Orin Nano (DB26J, JetPack 5/6). No ROS and no Duckietown software; every step that changes
# the system asks first, and steps that are already done are skipped.
#
#   cd ~/gym-duckietown/yolo_lane_following && bash duckie_yolo/robot/setup_orin.sh
#
# Result: robot_io (camera + wheels) runs natively as a systemd service; the web UI runs in the
# GPU Docker container and is used from a browser at http://<robot-ip>:8080
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # .../yolo_lane_following
VENV="${VENV:-$HOME/dy}"
ask() { read -r -p "$1 [y/N] " a; [[ "$a" =~ ^[Yy]$ ]]; }

echo "== duckie_yolo robot setup (package: $PKG_DIR)"

# ---------------------------------------------------------------- 1. board / JetPack
L4T="$(head -n1 /etc/nv_tegra_release 2>/dev/null || true)"
case "$L4T" in
  *R36*) JP=6 ;;
  *R35*) JP=5 ;;
  *R32*) JP=4 ;;
  *) JP=unknown ;;
esac
MODEL="$(tr -d '\0' </proc/device-tree/model 2>/dev/null || true)"
I2C_BUS=1
[[ "$MODEL" == *Orin* ]] && I2C_BUS=7
echo "-- L4T: ${L4T:-not a Jetson?} -> JetPack $JP"
echo "-- board: ${MODEL:-unknown} -> HAT on I2C bus $I2C_BUS"

# duckie_yolo needs Python >= 3.8; JetPack 4 (Ubuntu 18.04) ships 3.6
if [[ "$JP" == "4" ]]; then PY=python3.8; else PY=python3; fi

# ---------------------------------------------------------------- 2. Duckietown containers
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

# ---------------------------------------------------------------- 3. system packages
NEED=()
command -v "$PY" >/dev/null || NEED+=("$PY")
[[ "$JP" == "4" ]] && { dpkg -s python3.8-venv >/dev/null 2>&1 || NEED+=(python3.8-venv python3.8-dev); }
[[ "$JP" != "4" ]] && { dpkg -s python3-venv >/dev/null 2>&1 || NEED+=(python3-venv); }
command -v gst-launch-1.0 >/dev/null || NEED+=(gstreamer1.0-tools)
command -v i2cdetect >/dev/null || NEED+=(i2c-tools)
if (( ${#NEED[@]} )) && ask "-- apt-get install ${NEED[*]}?"; then
  sudo apt-get update && sudo apt-get install -y "${NEED[@]}"
fi

# ---------------------------------------------------------------- 4. Python env for robot_io
if [[ ! -x "$VENV/bin/python" ]] && ask "-- create the Python env $VENV ($PY) for robot_io?"; then
  "$PY" -m venv "$VENV"
fi
if [[ -x "$VENV/bin/python" ]]; then
  if ! "$VENV/bin/python" -c "import numpy, zmq, yaml, smbus2, cv2, Jetson.GPIO" 2>/dev/null; then
    if ask "-- install robot_io's packages into $VENV?"; then
      "$VENV/bin/python" -m pip install -U pip
      # prebuilt wheels only: building numpy / OpenCV on the Jetson takes hours.
      # OpenCV < 5: the code is tested on 4.x. pip OpenCV has no GStreamer; the camera then uses a gst-launch pipe.
      "$VENV/bin/python" -m pip install --only-binary :all: numpy pyzmq pyyaml smbus2 "opencv-python-headless<5"
      # Jetson.GPIO 2.1+ uses the gpiochip character device, which JetPack 4's kernel does not support
      # for this; 2.0.x (sysfs) is what JetPack 4 itself ships. Old versions are source-only (pure Python).
      if [[ "$JP" == "4" ]]; then "$VENV/bin/python" -m pip install "Jetson.GPIO<2.1"; else "$VENV/bin/python" -m pip install Jetson.GPIO; fi
    fi
  fi
  "$VENV/bin/python" -c "import numpy, zmq, yaml, smbus2, cv2, Jetson.GPIO; print('-- robot_io env ok, OpenCV', cv2.__version__)" \
    || echo "!! robot_io env incomplete (see above)"
fi

# ---------------------------------------------------------------- 5. permissions
if ! id -nG "$USER" | grep -qw i2c || ! id -nG "$USER" | grep -qw gpio; then
  if ask "-- add $USER to the i2c, gpio and video groups (log out and in afterwards)?"; then
    sudo groupadd -f -r gpio
    sudo usermod -aG i2c,gpio,video "$USER"
  fi
fi
if command -v docker >/dev/null 2>&1 && ! id -nG "$USER" | grep -qw docker; then
  if ask "-- add $USER to the docker group (run docker without sudo; log out and in afterwards)?"; then
    sudo usermod -aG docker "$USER"
  fi
fi

# ---------------------------------------------------------------- 6. hardware check
if command -v i2cdetect >/dev/null; then
  if sudo i2cdetect -y -r "$I2C_BUS" 2>/dev/null | grep -qE '(^| )60( |$)'; then
    echo "-- motor controller (PCA9685) found at 0x60 on bus $I2C_BUS"
  else
    echo "!! no device at 0x60 on I2C bus $I2C_BUS (check the HAT / bus number)"
  fi
fi

# ---------------------------------------------------------------- 7. robot_io as a service
if [[ -x "$VENV/bin/python" ]] && ask "-- run robot_io as a service (starts at boot, restarts if it crashes)?"; then
  echo "   Extra robot_io options, from the wheel test (--hw-test) and the first camera image:"
  echo "   e.g. --invert-left / --invert-right if a wheel turned the wrong way, --flip-method 2 if the image is upside down."
  read -r -p "   extra options (empty for none): " EXTRA
  if (ss -ltn 2>/dev/null || true) | grep -q ':5555 '; then
    echo "   NOTE: something already listens on :5555 (robot_io started by hand, e.g. in tmux?). Stop it first:"
    echo "         tmux attach -t io, then Ctrl-C."
    read -r -p "   press Enter when it is stopped... " _
  fi
  sudo tee /etc/systemd/system/duckie-robot-io.service >/dev/null <<EOF
[Unit]
Description=duckie_yolo robot_io (camera + wheels over ZMQ, no ROS)
After=network-online.target nvargus-daemon.service

[Service]
User=$USER
WorkingDirectory=$PKG_DIR
ExecStart=$VENV/bin/python -m duckie_yolo.robot.robot_io --i2c-bus $I2C_BUS $EXTRA
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload && sudo systemctl enable --now duckie-robot-io
  sleep 3
  systemctl --no-pager --lines=5 status duckie-robot-io || true
fi

# ---------------------------------------------------------------- 8. performance: all of the 4 GB for the GPU
if command -v nvpmodel >/dev/null && ask "-- max performance: highest power mode + max clocks at every boot (needs a good battery / 5V 4A supply)?"; then
  sudo nvpmodel -m 0 || true
  sudo tee /etc/systemd/system/duckie-jetson-clocks.service >/dev/null <<EOF
[Unit]
Description=Max Jetson clocks for duckie_yolo
After=nvpmodel.service

[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload && sudo systemctl enable --now duckie-jetson-clocks || true
fi
if ! swapon --show=NAME --noheadings 2>/dev/null | grep -q '^/swapfile$'; then
  if ask "-- add a 4 GB swap file (prevents out-of-memory while building TensorRT engines)?"; then
    sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
    grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
  fi
fi
if [[ "$(systemctl get-default 2>/dev/null)" == "graphical.target" ]]; then
  if ask "-- boot without the desktop (frees several hundred MB of RAM for the model; undo: sudo systemctl set-default graphical.target)?"; then
    sudo systemctl set-default multi-user.target
    echo "   takes effect at the next reboot"
  fi
fi

# ---------------------------------------------------------------- 9. web UI (GPU container)
if command -v docker >/dev/null 2>&1 && ask "-- build and start the web UI container (first build downloads several GB)?"; then
  if id -nG "$USER" | grep -qw docker; then
    bash "$PKG_DIR/duckie_yolo/robot/webui/start_webui.sh"
  else
    sudo -E bash "$PKG_DIR/duckie_yolo/robot/webui/start_webui.sh"
  fi
fi

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
cat <<EOF

== done
  Open http://${IP:-<robot-ip>}:8080 in a browser on your laptop: camera, driving + recording,
  models, calibration and the dataset guidelines are all there.

  robot_io by hand (instead of the service):  $VENV/bin/python -m duckie_yolo.robot.robot_io --i2c-bus $I2C_BUS
  wheel test (LIFT THE ROBOT):                $VENV/bin/python -m duckie_yolo.robot.robot_io --i2c-bus $I2C_BUS --hw-test
  web UI after a git pull:                    bash duckie_yolo/robot/webui/start_webui.sh
EOF
