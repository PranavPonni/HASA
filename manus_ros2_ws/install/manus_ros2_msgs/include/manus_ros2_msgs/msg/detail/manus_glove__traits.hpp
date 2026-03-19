// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from manus_ros2_msgs:msg/ManusGlove.idl
// generated code does not contain a copyright notice

#ifndef MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_GLOVE__TRAITS_HPP_
#define MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_GLOVE__TRAITS_HPP_

#include "manus_ros2_msgs/msg/detail/manus_glove__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

// Include directives for member types
// Member 'raw_sensor_orientation'
#include "geometry_msgs/msg/detail/quaternion__traits.hpp"

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<manus_ros2_msgs::msg::ManusGlove>()
{
  return "manus_ros2_msgs::msg::ManusGlove";
}

template<>
inline const char * name<manus_ros2_msgs::msg::ManusGlove>()
{
  return "manus_ros2_msgs/msg/ManusGlove";
}

template<>
struct has_fixed_size<manus_ros2_msgs::msg::ManusGlove>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<manus_ros2_msgs::msg::ManusGlove>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<manus_ros2_msgs::msg::ManusGlove>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_GLOVE__TRAITS_HPP_
