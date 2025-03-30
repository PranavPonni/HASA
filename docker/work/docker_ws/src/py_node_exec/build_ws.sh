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
