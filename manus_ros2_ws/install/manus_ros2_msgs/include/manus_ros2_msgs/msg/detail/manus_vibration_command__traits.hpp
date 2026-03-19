// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from manus_ros2_msgs:msg/ManusVibrationCommand.idl
// generated code does not contain a copyright notice

#ifndef MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_VIBRATION_COMMAND__TRAITS_HPP_
#define MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_VIBRATION_COMMAND__TRAITS_HPP_

#include "manus_ros2_msgs/msg/detail/manus_vibration_command__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<manus_ros2_msgs::msg::ManusVibrationCommand>()
{
  return "manus_ros2_msgs::msg::ManusVibrationCommand";
}

template<>
inline const char * name<manus_ros2_msgs::msg::ManusVibrationCommand>()
{
  return "manus_ros2_msgs/msg/ManusVibrationCommand";
}

template<>
struct has_fixed_size<manus_ros2_msgs::msg::ManusVibrationCommand>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<manus_ros2_msgs::msg::ManusVibrationCommand>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<manus_ros2_msgs::msg::ManusVibrationCommand>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_VIBRATION_COMMAND__TRAITS_HPP_
