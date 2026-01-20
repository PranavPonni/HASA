// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from manus_ros2_msgs:msg/ManusVibrationCommand.idl
// generated code does not contain a copyright notice

#ifndef MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_VIBRATION_COMMAND__STRUCT_H_
#define MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_VIBRATION_COMMAND__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Struct defined in msg/ManusVibrationCommand in the package manus_ros2_msgs.
typedef struct manus_ros2_msgs__msg__ManusVibrationCommand
{
  float intensities[5];
} manus_ros2_msgs__msg__ManusVibrationCommand;

// Struct for a sequence of manus_ros2_msgs__msg__ManusVibrationCommand.
typedef struct manus_ros2_msgs__msg__ManusVibrationCommand__Sequence
{
  manus_ros2_msgs__msg__ManusVibrationCommand * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} manus_ros2_msgs__msg__ManusVibrationCommand__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_VIBRATION_COMMAND__STRUCT_H_
