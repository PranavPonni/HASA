// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from manus_ros2_msgs:msg/ManusRawNode.idl
// generated code does not contain a copyright notice

#ifndef MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_RAW_NODE__STRUCT_HPP_
#define MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_RAW_NODE__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


// Include directives for member types
// Member 'pose'
#include "geometry_msgs/msg/detail/pose__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__manus_ros2_msgs__msg__ManusRawNode __attribute__((deprecated))
#else
# define DEPRECATED__manus_ros2_msgs__msg__ManusRawNode __declspec(deprecated)
#endif

namespace manus_ros2_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ManusRawNode_
{
  using Type = ManusRawNode_<ContainerAllocator>;

  explicit ManusRawNode_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : pose(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->node_id = 0l;
      this->parent_node_id = 0l;
      this->joint_type = "";
      this->chain_type = "";
    }
  }

  explicit ManusRawNode_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : joint_type(_alloc),
    chain_type(_alloc),
    pose(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->node_id = 0l;
      this->parent_node_id = 0l;
      this->joint_type = "";
      this->chain_type = "";
    }
  }

  // field types and members
  using _node_id_type =
    int32_t;
  _node_id_type node_id;
  using _parent_node_id_type =
    int32_t;
  _parent_node_id_type parent_node_id;
  using _joint_type_type =
    std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other>;
  _joint_type_type joint_type;
  using _chain_type_type =
    std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other>;
  _chain_type_type chain_type;
  using _pose_type =
    geometry_msgs::msg::Pose_<ContainerAllocator>;
  _pose_type pose;

  // setters for named parameter idiom
  Type & set__node_id(
    const int32_t & _arg)
  {
    this->node_id = _arg;
    return *this;
  }
  Type & set__parent_node_id(
    const int32_t & _arg)
  {
    this->parent_node_id = _arg;
    return *this;
  }
  Type & set__joint_type(
    const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other> & _arg)
  {
    this->joint_type = _arg;
    return *this;
  }
  Type & set__chain_type(
    const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other> & _arg)
  {
    this->chain_type = _arg;
    return *this;
  }
  Type & set__pose(
    const geometry_msgs::msg::Pose_<ContainerAllocator> & _arg)
  {
    this->pose = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    manus_ros2_msgs::msg::ManusRawNode_<ContainerAllocator> *;
  using ConstRawPtr =
    const manus_ros2_msgs::msg::ManusRawNode_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<manus_ros2_msgs::msg::ManusRawNode_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<manus_ros2_msgs::msg::ManusRawNode_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      manus_ros2_msgs::msg::ManusRawNode_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<manus_ros2_msgs::msg::ManusRawNode_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      manus_ros2_msgs::msg::ManusRawNode_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<manus_ros2_msgs::msg::ManusRawNode_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<manus_ros2_msgs::msg::ManusRawNode_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<manus_ros2_msgs::msg::ManusRawNode_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__manus_ros2_msgs__msg__ManusRawNode
    std::shared_ptr<manus_ros2_msgs::msg::ManusRawNode_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__manus_ros2_msgs__msg__ManusRawNode
    std::shared_ptr<manus_ros2_msgs::msg::ManusRawNode_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ManusRawNode_ & other) const
  {
    if (this->node_id != other.node_id) {
      return false;
    }
    if (this->parent_node_id != other.parent_node_id) {
      return false;
    }
    if (this->joint_type != other.joint_type) {
      return false;
    }
    if (this->chain_type != other.chain_type) {
      return false;
    }
    if (this->pose != other.pose) {
      return false;
    }
    return true;
  }
  bool operator!=(const ManusRawNode_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ManusRawNode_

// alias to use template instance with default allocator
using ManusRawNode =
  manus_ros2_msgs::msg::ManusRawNode_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace manus_ros2_msgs

#endif  // MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_RAW_NODE__STRUCT_HPP_
