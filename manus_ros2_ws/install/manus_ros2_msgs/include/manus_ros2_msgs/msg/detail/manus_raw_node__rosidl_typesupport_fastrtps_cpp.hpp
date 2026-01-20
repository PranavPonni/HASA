// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__rosidl_typesupport_fastrtps_cpp.hpp.em
// with input from manus_ros2_msgs:msg/ManusRawNode.idl
// generated code does not contain a copyright notice

#ifndef MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_RAW_NODE__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
#define MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_RAW_NODE__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_

#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "manus_ros2_msgs/msg/rosidl_typesupport_fastrtps_cpp__visibility_control.h"
#include "manus_ros2_msgs/msg/detail/manus_raw_node__struct.hpp"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

#include "fastcdr/Cdr.h"

namespace manus_ros2_msgs
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_manus_ros2_msgs
cdr_serialize(
  const manus_ros2_msgs::msg::ManusRawNode & ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_manus_ros2_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  manus_ros2_msgs::msg::ManusRawNode & ros_message);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_manus_ros2_msgs
get_serialized_size(
  const manus_ros2_msgs::msg::ManusRawNode & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_manus_ros2_msgs
max_serialized_size_ManusRawNode(
  bool & full_bounded,
  size_t current_alignment);

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace manus_ros2_msgs

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_manus_ros2_msgs
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, manus_ros2_msgs, msg, ManusRawNode)();

#ifdef __cplusplus
}
#endif

#endif  // MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_RAW_NODE__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
