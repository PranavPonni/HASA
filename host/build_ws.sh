#!/bin/bash
WS_DIR=$(pwd)

if [ -d "$WS_DIR/devel" ] && [ -d "$WS_DIR/build" ]; then
    cd "$WS_DIR"
    catkin_make
else
    cd "$WS_DIR/src"
    catkin_init_workspace
    cd "$WS_DIR"
    catkin_make
fi

if ! grep -q "source_hasa" ~/.bashrc; then
    alias source_hasa="source ${WS_DIR}/devel/setup.bash"
    echo "alias source_hasa=\"source ${WS_DIR}/devel/setup.bash\"" >> ~/.bashrc
    echo "source_hasa alias has been set."
fi