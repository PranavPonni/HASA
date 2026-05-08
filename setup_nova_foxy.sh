#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/Adjuvo/senseglove_ros.git"
BRANCH="jazzy"
WS_DIR="${1:-$HOME/HASA/senseglove_ros2_ws}"

echo "[1/6] Preparing workspace: ${WS_DIR}"
mkdir -p "${WS_DIR}"

if [[ ! -d "${WS_DIR}/.git" ]]; then
  echo "[2/6] Cloning ${REPO_URL} (branch: ${BRANCH})"
  git clone --branch "${BRANCH}" --single-branch "${REPO_URL}" "${WS_DIR}"
else
  echo "[2/6] Updating existing repository"
  git -C "${WS_DIR}" fetch origin "${BRANCH}"
  git -C "${WS_DIR}" checkout "${BRANCH}"
  git -C "${WS_DIR}" pull --ff-only origin "${BRANCH}"
fi

echo "[3/6] Applying Foxy compatibility patches"
find "${WS_DIR}" -name CMakeLists.txt -type f -print0 | while IFS= read -r -d '' file; do
  sed -i 's/cmake_minimum_required(VERSION[[:space:]]*3\.21\.\.\.3\.25)/cmake_minimum_required(VERSION 3.16)/g' "$file"
  sed -i 's/cmake_minimum_required(VERSION[[:space:]]*3\.21)/cmake_minimum_required(VERSION 3.16)/g' "$file"
done

FD_NODE="${WS_DIR}/senseglove/senseglove_interaction/senseglove_interaction/finger_distance/finger_distance_node.py"
if [[ -f "${FD_NODE}" ]]; then
  sed -i 's/params: list\[Parameter\]/params/g' "${FD_NODE}"
fi

echo "[4/6] Installing ROS dependencies (Foxy)"
# Foxy setup scripts may reference unset vars; avoid nounset failure.
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"
set +u
source /opt/ros/foxy/setup.bash
set -u
rosdep update
if ! rosdep install --from-paths "${WS_DIR}/senseglove" --ignore-src -r -y --rosdistro foxy; then
  echo "[warn] rosdep could not resolve all keys on EOL Foxy. Trying apt fallback packages..."
  sudo apt-get update
  sudo apt-get install -y \
    ros-foxy-control-msgs \
    ros-foxy-controller-manager \
    ros-foxy-controller-interface \
    ros-foxy-hardware-interface \
    ros-foxy-ros2-control \
    ros-foxy-ros2-controllers \
    ros-foxy-xacro \
    python3-pytest
fi

echo "[5/6] Building workspace"
cd "${WS_DIR}"
colcon build --symlink-install

echo "[6/6] Done"
echo
echo "Use this in each Nova terminal:"
echo "  source /opt/ros/foxy/setup.bash"
echo "  source ${WS_DIR}/install/setup.bash"
echo
echo "If your gloves are Bluetooth Classic, pair and bind rfcomm ports first per SenseGlove docs."
