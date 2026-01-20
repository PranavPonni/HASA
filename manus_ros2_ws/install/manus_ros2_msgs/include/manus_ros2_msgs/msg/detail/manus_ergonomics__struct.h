// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from manus_ros2_msgs:msg/ManusErgonomics.idl
// generated code does not contain a copyright notice

#ifndef MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_ERGONOMICS__STRUCT_H_
#define MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_ERGONOMICS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'type'
#include "rosidl_runtime_c/string.h"

// Struct defined in msg/ManusErgonomics in the package manus_ros2_msgs.
typedef struct manus_ros2_msgs__msg__ManusErgonomics
{
  rosidl_runtime_c__String type;
  float value;
} manus_ros2_msgs__msg__ManusErgonomics;

// Struct for a sequence of manus_ros2_msgs__msg__ManusErgonomics.
typedef struct manus_ros2_msgs__msg__ManusErgonomics__Sequence
{
  manus_ros2_msgs__msg__ManusErgonomics * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} manus_ros2_msgs__msg__ManusErgonomics__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_ERGONOMICS__STRUCT_H_
