// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from manus_ros2_msgs:msg/ManusRawNode.idl
// generated code does not contain a copyright notice

#ifndef MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_RAW_NODE__BUILDER_HPP_
#define MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_RAW_NODE__BUILDER_HPP_

#include "manus_ros2_msgs/msg/detail/manus_raw_node__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace manus_ros2_msgs
{

namespace msg
{

namespace builder
{

class Init_ManusRawNode_pose
{
public:
  explicit Init_ManusRawNode_pose(::manus_ros2_msgs::msg::ManusRawNode & msg)
  : msg_(msg)
  {}
  ::manus_ros2_msgs::msg::ManusRawNode pose(::manus_ros2_msgs::msg::ManusRawNode::_pose_type arg)
  {
    msg_.pose = std::move(arg);
    return std::move(msg_);
  }

private:
  ::manus_ros2_msgs::msg::ManusRawNode msg_;
};

class Init_ManusRawNode_chain_type
{
public:
  explicit Init_ManusRawNode_chain_type(::manus_ros2_msgs::msg::ManusRawNode & msg)
  : msg_(msg)
  {}
  Init_ManusRawNode_pose chain_type(::manus_ros2_msgs::msg::ManusRawNode::_chain_type_type arg)
  {
    msg_.chain_type = std::move(arg);
    return Init_ManusRawNode_pose(msg_);
  }

private:
  ::manus_ros2_msgs::msg::ManusRawNode msg_;
};

class Init_ManusRawNode_joint_type
{
public:
  explicit Init_ManusRawNode_joint_type(::manus_ros2_msgs::msg::ManusRawNode & msg)
  : msg_(msg)
  {}
  Init_ManusRawNode_chain_type joint_type(::manus_ros2_msgs::msg::ManusRawNode::_joint_type_type arg)
  {
    msg_.joint_type = std::move(arg);
    return Init_ManusRawNode_chain_type(msg_);
  }

private:
  ::manus_ros2_msgs::msg::ManusRawNode msg_;
};

class Init_ManusRawNode_parent_node_id
{
public:
  explicit Init_ManusRawNode_parent_node_id(::manus_ros2_msgs::msg::ManusRawNode & msg)
  : msg_(msg)
  {}
  Init_ManusRawNode_joint_type parent_node_id(::manus_ros2_msgs::msg::ManusRawNode::_parent_node_id_type arg)
  {
    msg_.parent_node_id = std::move(arg);
    return Init_ManusRawNode_joint_type(msg_);
  }

private:
  ::manus_ros2_msgs::msg::ManusRawNode msg_;
};

class Init_ManusRawNode_node_id
{
public:
  Init_ManusRawNode_node_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ManusRawNode_parent_node_id node_id(::manus_ros2_msgs::msg::ManusRawNode::_node_id_type arg)
  {
    msg_.node_id = std::move(arg);
    return Init_ManusRawNode_parent_node_id(msg_);
  }

private:
  ::manus_ros2_msgs::msg::ManusRawNode msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::manus_ros2_msgs::msg::ManusRawNode>()
{
  return manus_ros2_msgs::msg::builder::Init_ManusRawNode_node_id();
}

}  // namespace manus_ros2_msgs

#endif  // MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_RAW_NODE__BUILDER_HPP_
