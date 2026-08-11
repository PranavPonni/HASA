# Install script for directory: /home/handlingteam2/HASA/ros_ws/src/xela_package/xela_server_ros

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/xela_server_ros/msg" TYPE FILE FILES
    "/home/handlingteam2/HASA/ros_ws/src/xela_package/xela_server_ros/msg/SensorFull.msg"
    "/home/handlingteam2/HASA/ros_ws/src/xela_package/xela_server_ros/msg/SensStream.msg"
    "/home/handlingteam2/HASA/ros_ws/src/xela_package/xela_server_ros/msg/Taxel.msg"
    "/home/handlingteam2/HASA/ros_ws/src/xela_package/xela_server_ros/msg/Forces.msg"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/xela_server_ros/srv" TYPE FILE FILES "/home/handlingteam2/HASA/ros_ws/src/xela_package/xela_server_ros/srv/XelaSensorStream.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/xela_server_ros/cmake" TYPE FILE FILES "/home/handlingteam2/HASA/ros_ws/build-vscode/xela_package/xela_server_ros/catkin_generated/installspace/xela_server_ros-msg-paths.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/handlingteam2/HASA/ros_ws/build-vscode/devel/include/xela_server_ros")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/home/handlingteam2/HASA/ros_ws/build-vscode/devel/share/roseus/ros/xela_server_ros")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/home/handlingteam2/HASA/ros_ws/build-vscode/devel/share/common-lisp/ros/xela_server_ros")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/home/handlingteam2/HASA/ros_ws/build-vscode/devel/share/gennodejs/ros/xela_server_ros")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/usr/bin/python3" -m compileall "/home/handlingteam2/HASA/ros_ws/build-vscode/devel/lib/python3/dist-packages/xela_server_ros")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages" TYPE DIRECTORY FILES "/home/handlingteam2/HASA/ros_ws/build-vscode/devel/lib/python3/dist-packages/xela_server_ros")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/handlingteam2/HASA/ros_ws/build-vscode/xela_package/xela_server_ros/catkin_generated/installspace/xela_server_ros.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/xela_server_ros/cmake" TYPE FILE FILES "/home/handlingteam2/HASA/ros_ws/build-vscode/xela_package/xela_server_ros/catkin_generated/installspace/xela_server_ros-msg-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/xela_server_ros/cmake" TYPE FILE FILES
    "/home/handlingteam2/HASA/ros_ws/build-vscode/xela_package/xela_server_ros/catkin_generated/installspace/xela_server_rosConfig.cmake"
    "/home/handlingteam2/HASA/ros_ws/build-vscode/xela_package/xela_server_ros/catkin_generated/installspace/xela_server_rosConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/xela_server_ros" TYPE FILE FILES "/home/handlingteam2/HASA/ros_ws/src/xela_package/xela_server_ros/package.xml")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/xela_server_ros" TYPE DIRECTORY FILES "/home/handlingteam2/HASA/ros_ws/src/xela_package/xela_server_ros/scripts/" USE_SOURCE_PERMISSIONS)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/xela_server_ros/launch" TYPE DIRECTORY FILES "/home/handlingteam2/HASA/ros_ws/src/xela_package/xela_server_ros/launch/" USE_SOURCE_PERMISSIONS)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/xela_server_ros/launch" TYPE DIRECTORY FILES "/home/handlingteam2/HASA/ros_ws/src/xela_package/xela_server_ros/config/" USE_SOURCE_PERMISSIONS)
endif()

