// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from manus_ros2_msgs:msg/ManusGlove.idl
// generated code does not contain a copyright notice

#ifndef MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_GLOVE__STRUCT_H_
#define MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_GLOVE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'side'
#include "rosidl_runtime_c/string.h"
// Member 'raw_nodes'
#include "manus_ros2_msgs/msg/detail/manus_raw_node__struct.h"
// Member 'ergonomics'
#include "manus_ros2_msgs/msg/detail/manus_ergonomics__struct.h"
// Member 'raw_sensor_orientation'
#include "geometry_msgs/msg/detail/quaternion__struct.h"
// Member 'raw_sensor'
#include "geometry_msgs/msg/detail/pose__struct.h"

// Struct defined in msg/ManusGlove in the package manus_ros2_msgs.
typedef struct manus_ros2_msgs__msg__ManusGlove
{
  int32_t glove_id;
  rosidl_runtime_c__String side;
  int32_t raw_node_count;
  manus_ros2_msgs__msg__ManusRawNode__Sequence raw_nodes;
  int32_t ergonomics_count;
  manus_ros2_msgs__msg__ManusErgonomics__Sequence ergonomics;
  geometry_msgs__msg__Quaternion raw_sensor_orientation;
  int32_t raw_sensor_count;
  geometry_msgs__msg__Pose__Sequence raw_sensor;
} manus_ros2_msgs__msg__ManusGlove;

// Struct for a sequence of manus_ros2_msgs__msg__ManusGlove.
typedef struct manus_ros2_msgs__msg__ManusGlove__Sequence
{
  manus_ros2_msgs__msg__ManusGlove * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} manus_ros2_msgs__msg__ManusGlove__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_GLOVE__STRUCT_H_
