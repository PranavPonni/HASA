// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from manus_ros2_msgs:msg/ManusErgonomics.idl
// generated code does not contain a copyright notice

#ifndef MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_ERGONOMICS__BUILDER_HPP_
#define MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_ERGONOMICS__BUILDER_HPP_

#include "manus_ros2_msgs/msg/detail/manus_ergonomics__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace manus_ros2_msgs
{

namespace msg
{

namespace builder
{

class Init_ManusErgonomics_value
{
public:
  explicit Init_ManusErgonomics_value(::manus_ros2_msgs::msg::ManusErgonomics & msg)
  : msg_(msg)
  {}
  ::manus_ros2_msgs::msg::ManusErgonomics value(::manus_ros2_msgs::msg::ManusErgonomics::_value_type arg)
  {
    msg_.value = std::move(arg);
    return std::move(msg_);
  }

private:
  ::manus_ros2_msgs::msg::ManusErgonomics msg_;
};

class Init_ManusErgonomics_type
{
public:
  Init_ManusErgonomics_type()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ManusErgonomics_value type(::manus_ros2_msgs::msg::ManusErgonomics::_type_type arg)
  {
    msg_.type = std::move(arg);
    return Init_ManusErgonomics_value(msg_);
  }

private:
  ::manus_ros2_msgs::msg::ManusErgonomics msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::manus_ros2_msgs::msg::ManusErgonomics>()
{
  return manus_ros2_msgs::msg::builder::Init_ManusErgonomics_type();
}

}  // namespace manus_ros2_msgs

#endif  // MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_ERGONOMICS__BUILDER_HPP_
