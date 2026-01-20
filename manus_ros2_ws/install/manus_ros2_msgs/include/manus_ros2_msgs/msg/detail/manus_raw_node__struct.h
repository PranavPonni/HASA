// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from manus_ros2_msgs:msg/ManusRawNode.idl
// generated code does not contain a copyright notice

#ifndef MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_RAW_NODE__STRUCT_H_
#define MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_RAW_NODE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'joint_type'
// Member 'chain_type'
#include "rosidl_runtime_c/string.h"
// Member 'pose'
#include "geometry_msgs/msg/detail/pose__struct.h"

// Struct defined in msg/ManusRawNode in the package manus_ros2_msgs.
typedef struct manus_ros2_msgs__msg__ManusRawNode
{
  int32_t node_id;
  int32_t parent_node_id;
  rosidl_runtime_c__String joint_type;
  rosidl_runtime_c__String chain_type;
  geometry_msgs__msg__Pose pose;
} manus_ros2_msgs__msg__ManusRawNode;

// Struct for a sequence of manus_ros2_msgs__msg__ManusRawNode.
typedef struct manus_ros2_msgs__msg__ManusRawNode__Sequence
{
  manus_ros2_msgs__msg__ManusRawNode * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} manus_ros2_msgs__msg__ManusRawNode__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_RAW_NODE__STRUCT_H_
