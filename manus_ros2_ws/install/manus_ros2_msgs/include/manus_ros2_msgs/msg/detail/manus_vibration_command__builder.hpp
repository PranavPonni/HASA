// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from manus_ros2_msgs:msg/ManusVibrationCommand.idl
// generated code does not contain a copyright notice

#ifndef MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_VIBRATION_COMMAND__BUILDER_HPP_
#define MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_VIBRATION_COMMAND__BUILDER_HPP_

#include "manus_ros2_msgs/msg/detail/manus_vibration_command__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace manus_ros2_msgs
{

namespace msg
{

namespace builder
{

class Init_ManusVibrationCommand_intensities
{
public:
  Init_ManusVibrationCommand_intensities()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::manus_ros2_msgs::msg::ManusVibrationCommand intensities(::manus_ros2_msgs::msg::ManusVibrationCommand::_intensities_type arg)
  {
    msg_.intensities = std::move(arg);
    return std::move(msg_);
  }

private:
  ::manus_ros2_msgs::msg::ManusVibrationCommand msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::manus_ros2_msgs::msg::ManusVibrationCommand>()
{
  return manus_ros2_msgs::msg::builder::Init_ManusVibrationCommand_intensities();
}

}  // namespace manus_ros2_msgs

#endif  // MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_VIBRATION_COMMAND__BUILDER_HPP_
