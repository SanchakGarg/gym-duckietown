#!/usr/bin/env bash
# Start (or restart) the Duckiebot web UI in its GPU Docker container on the Jetson.
#
#   cd ~/gym-duckietown/yolo_lane_following && bash duckie_yolo/robot/webui/start_webui.sh
#   then open http://<robot-ip>:5555 in a browser
#
# The container:
#   * uses the GPU (--runtime nvidia) with NO memory / CPU limits. On a Jetson the GPU shares the
#     board's RAM, so "all GPU memory" means: no container limits + as little else running as possible
#     (setup_orin.sh can add swap, max clocks and a headless boot)
#   * shares the host network: it reaches robot_io on localhost:7777 and serves :5555
#   * restarts by itself after crashes and reboots (--restart unless-stopped)
#   * runs the code from this folder (bind mount): after `git pull` just run this script again
#
# Options (environment variables):
#   PORT=5555  ROBOT_CONFIG=configs/robot_mybot.yaml  IMAGE=duckie-yolo-agent  REBUILD=1  FOLLOW_LOGS=1
#   ROBOT_IO_HOST=localhost
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # .../yolo_lane_following
IMAGE="${IMAGE:-duckie-yolo-agent}"
NAME="${NAME:-duckie-webui}"
PORT="${PORT:-5555}"
ROBOT_CONFIG="${ROBOT_CONFIG:-configs/robot_mybot.yaml}"
ROBOT_IO_HOST="${ROBOT_IO_HOST:-localhost}"

L4T="$(head -n1 /etc/nv_tegra_release 2>/dev/null || true)"
case "$L4T" in
  *R36*) JP=6 ;;
  *R35*) JP=5 ;;
  *R32*) JP=4 ;;
  *) echo "!! not a Jetson (no /etc/nv_tegra_release): this script is meant for the robot"; exit 1 ;;
esac

if ! docker info >/dev/null 2>&1; then
  echo "!! cannot talk to Docker. Is it running, and are you in the 'docker' group? (try: sudo usermod -aG docker \$USER, then log in again)"
  exit 1
fi

if [[ "${REBUILD:-0}" == "1" ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "== building $IMAGE for JetPack $JP (first time: large download, can take 20+ minutes)"
  docker build -t "$IMAGE" --build-arg JETPACK="$JP" -f "$PKG_DIR/duckie_yolo/robot/docker/Dockerfile.agent" "$PKG_DIR"
fi

mkdir -p "$PKG_DIR/models" "$PKG_DIR/data/real" "$PKG_DIR/calib" "$PKG_DIR/configs"

if docker container inspect "$NAME" >/dev/null 2>&1; then
  echo "== replacing the running $NAME container"
  docker rm -f "$NAME" >/dev/null
fi

docker run -d --name "$NAME" \
  --restart unless-stopped \
  --runtime nvidia \
  --network host \
  --ipc host \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e DUCKIE_UID="${SUDO_UID:-$(id -u)}" -e DUCKIE_GID="${SUDO_GID:-$(id -g)}" \
  -v /etc/localtime:/etc/localtime:ro \
  -v "$PKG_DIR":/work -w /work \
  --entrypoint python3 \
  "$IMAGE" -m duckie_yolo.robot.webui --port "$PORT" --robot-host "$ROBOT_IO_HOST" --robot-config "$ROBOT_CONFIG" >/dev/null

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "== web UI starting: open http://${IP:-<robot-ip>}:$PORT"
echo "   logs:  docker logs -f $NAME      stop:  docker stop $NAME      (it restarts at boot until stopped)"
if ! (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ':7777 '; then
  echo "!! nothing is listening on :7777 - start robot_io first (systemctl start duckie-robot-io, or by hand)"
fi
if [[ "${FOLLOW_LOGS:-1}" == "1" ]]; then
  echo "   (showing logs, Ctrl-C only stops watching; the web UI keeps running)"
  sleep 2
  timeout 25 docker logs -f "$NAME" || true
fi
