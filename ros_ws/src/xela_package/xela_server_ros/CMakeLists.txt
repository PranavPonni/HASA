cmake_minimum_required(VERSION 3.5)
project(xela_server_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  message_generation
)
################################################
## Declare ROS messages, services and actions ##
################################################
add_message_files(
  FILES
  "SensorFull.msg"
  "SensStream.msg"
  "Taxel.msg"
  "Forces.msg"
)

add_service_files(
  FILES
  "XelaSensorStream.srv"
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs

)
catkin_package(
  CATKIN_DEPENDS 
    rospy 
    std_msgs 
    geometry_msgs
    message_runtime
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

install(
  DIRECTORY scripts/
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  USE_SOURCE_PERMISSIONS
)

install(
  DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  USE_SOURCE_PERMISSIONS
)

install(
  DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  USE_SOURCE_PERMISSIONS
)
