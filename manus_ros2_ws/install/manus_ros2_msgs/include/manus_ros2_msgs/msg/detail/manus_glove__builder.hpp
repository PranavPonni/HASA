// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from manus_ros2_msgs:msg/ManusGlove.idl
// generated code does not contain a copyright notice

#ifndef MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_GLOVE__BUILDER_HPP_
#define MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_GLOVE__BUILDER_HPP_

#include "manus_ros2_msgs/msg/detail/manus_glove__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace manus_ros2_msgs
{

namespace msg
{

namespace builder
{

class Init_ManusGlove_raw_sensor
{
public:
  explicit Init_ManusGlove_raw_sensor(::manus_ros2_msgs::msg::ManusGlove & msg)
  : msg_(msg)
  {}
  ::manus_ros2_msgs::msg::ManusGlove raw_sensor(::manus_ros2_msgs::msg::ManusGlove::_raw_sensor_type arg)
  {
    msg_.raw_sensor = std::move(arg);
    return std::move(msg_);
  }

private:
  ::manus_ros2_msgs::msg::ManusGlove msg_;
};

class Init_ManusGlove_raw_sensor_count
{
public:
  explicit Init_ManusGlove_raw_sensor_count(::manus_ros2_msgs::msg::ManusGlove & msg)
  : msg_(msg)
  {}
  Init_ManusGlove_raw_sensor raw_sensor_count(::manus_ros2_msgs::msg::ManusGlove::_raw_sensor_count_type arg)
  {
    msg_.raw_sensor_count = std::move(arg);
    return Init_ManusGlove_raw_sensor(msg_);
  }

private:
  ::manus_ros2_msgs::msg::ManusGlove msg_;
};

class Init_ManusGlove_raw_sensor_orientation
{
public:
  explicit Init_ManusGlove_raw_sensor_orientation(::manus_ros2_msgs::msg::ManusGlove & msg)
  : msg_(msg)
  {}
  Init_ManusGlove_raw_sensor_count raw_sensor_orientation(::manus_ros2_msgs::msg::ManusGlove::_raw_sensor_orientation_type arg)
  {
    msg_.raw_sensor_orientation = std::move(arg);
    return Init_ManusGlove_raw_sensor_count(msg_);
  }

private:
  ::manus_ros2_msgs::msg::ManusGlove msg_;
};

class Init_ManusGlove_ergonomics
{
public:
  explicit Init_ManusGlove_ergonomics(::manus_ros2_msgs::msg::ManusGlove & msg)
  : msg_(msg)
  {}
  Init_ManusGlove_raw_sensor_orientation ergonomics(::manus_ros2_msgs::msg::ManusGlove::_ergonomics_type arg)
  {
    msg_.ergonomics = std::move(arg);
    return Init_ManusGlove_raw_sensor_orientation(msg_);
  }

private:
  ::manus_ros2_msgs::msg::ManusGlove msg_;
};

class Init_ManusGlove_ergonomics_count
{
public:
  explicit Init_ManusGlove_ergonomics_count(::manus_ros2_msgs::msg::ManusGlove & msg)
  : msg_(msg)
  {}
  Init_ManusGlove_ergonomics ergonomics_count(::manus_ros2_msgs::msg::ManusGlove::_ergonomics_count_type arg)
  {
    msg_.ergonomics_count = std::move(arg);
    return Init_ManusGlove_ergonomics(msg_);
  }

private:
  ::manus_ros2_msgs::msg::ManusGlove msg_;
};

class Init_ManusGlove_raw_nodes
{
public:
  explicit Init_ManusGlove_raw_nodes(::manus_ros2_msgs::msg::ManusGlove & msg)
  : msg_(msg)
  {}
  Init_ManusGlove_ergonomics_count raw_nodes(::manus_ros2_msgs::msg::ManusGlove::_raw_nodes_type arg)
  {
    msg_.raw_nodes = std::move(arg);
    return Init_ManusGlove_ergonomics_count(msg_);
  }

private:
  ::manus_ros2_msgs::msg::ManusGlove msg_;
};

class Init_ManusGlove_raw_node_count
{
public:
  explicit Init_ManusGlove_raw_node_count(::manus_ros2_msgs::msg::ManusGlove & msg)
  : msg_(msg)
  {}
  Init_ManusGlove_raw_nodes raw_node_count(::manus_ros2_msgs::msg::ManusGlove::_raw_node_count_type arg)
  {
    msg_.raw_node_count = std::move(arg);
    return Init_ManusGlove_raw_nodes(msg_);
  }

private:
  ::manus_ros2_msgs::msg::ManusGlove msg_;
};

class Init_ManusGlove_side
{
public:
  explicit Init_ManusGlove_side(::manus_ros2_msgs::msg::ManusGlove & msg)
  : msg_(msg)
  {}
  Init_ManusGlove_raw_node_count side(::manus_ros2_msgs::msg::ManusGlove::_side_type arg)
  {
    msg_.side = std::move(arg);
    return Init_ManusGlove_raw_node_count(msg_);
  }

private:
  ::manus_ros2_msgs::msg::ManusGlove msg_;
};

class Init_ManusGlove_glove_id
{
public:
  Init_ManusGlove_glove_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ManusGlove_side glove_id(::manus_ros2_msgs::msg::ManusGlove::_glove_id_type arg)
  {
    msg_.glove_id = std::move(arg);
    return Init_ManusGlove_side(msg_);
  }

private:
  ::manus_ros2_msgs::msg::ManusGlove msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::manus_ros2_msgs::msg::ManusGlove>()
{
  return manus_ros2_msgs::msg::builder::Init_ManusGlove_glove_id();
}

}  // namespace manus_ros2_msgs

#endif  // MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_GLOVE__BUILDER_HPP_
