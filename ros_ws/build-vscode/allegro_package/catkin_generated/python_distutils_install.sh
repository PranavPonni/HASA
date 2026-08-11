#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/handlingteam2/HASA/ros_ws/src/allegro_package"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/usr/local/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/usr/local/lib/python3/dist-packages:/home/handlingteam2/HASA/ros_ws/build-vscode/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/handlingteam2/HASA/ros_ws/build-vscode" \
    "/usr/bin/python3" \
    "/home/handlingteam2/HASA/ros_ws/src/allegro_package/setup.py" \
     \
    build --build-base "/home/handlingteam2/HASA/ros_ws/build-vscode/allegro_package" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/usr/local" --install-scripts="/usr/local/bin"
