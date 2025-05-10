cmake_minimum_required(VERSION 2.8.3)
project(xela_py)
find_package(catkin REQUIRED COMPONENTS
  rospy
  xela_server_ros
)
catkin_python_setup()
catkin_package()