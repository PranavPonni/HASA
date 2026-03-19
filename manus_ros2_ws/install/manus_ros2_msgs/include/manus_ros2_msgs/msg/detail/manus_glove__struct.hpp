// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from manus_ros2_msgs:msg/ManusGlove.idl
// generated code does not contain a copyright notice

#ifndef MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_GLOVE__STRUCT_HPP_
#define MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_GLOVE__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


// Include directives for member types
// Member 'raw_nodes'
#include "manus_ros2_msgs/msg/detail/manus_raw_node__struct.hpp"
// Member 'ergonomics'
#include "manus_ros2_msgs/msg/detail/manus_ergonomics__struct.hpp"
// Member 'raw_sensor_orientation'
#include "geometry_msgs/msg/detail/quaternion__struct.hpp"
// Member 'raw_sensor'
#include "geometry_msgs/msg/detail/pose__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__manus_ros2_msgs__msg__ManusGlove __attribute__((deprecated))
#else
# define DEPRECATED__manus_ros2_msgs__msg__ManusGlove __declspec(deprecated)
#endif

namespace manus_ros2_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ManusGlove_
{
  using Type = ManusGlove_<ContainerAllocator>;

  explicit ManusGlove_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : raw_sensor_orientation(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->glove_id = 0l;
      this->side = "";
      this->raw_node_count = 0l;
      this->ergonomics_count = 0l;
      this->raw_sensor_count = 0l;
    }
  }

  explicit ManusGlove_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : side(_alloc),
    raw_sensor_orientation(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->glove_id = 0l;
      this->side = "";
      this->raw_node_count = 0l;
      this->ergonomics_count = 0l;
      this->raw_sensor_count = 0l;
    }
  }

  // field types and members
  using _glove_id_type =
    int32_t;
  _glove_id_type glove_id;
  using _side_type =
    std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other>;
  _side_type side;
  using _raw_node_count_type =
    int32_t;
  _raw_node_count_type raw_node_count;
  using _raw_nodes_type =
    std::vector<manus_ros2_msgs::msg::ManusRawNode_<ContainerAllocator>, typename ContainerAllocator::template rebind<manus_ros2_msgs::msg::ManusRawNode_<ContainerAllocator>>::other>;
  _raw_nodes_type raw_nodes;
  using _ergonomics_count_type =
    int32_t;
  _ergonomics_count_type ergonomics_count;
  using _ergonomics_type =
    std::vector<manus_ros2_msgs::msg::ManusErgonomics_<ContainerAllocator>, typename ContainerAllocator::template rebind<manus_ros2_msgs::msg::ManusErgonomics_<ContainerAllocator>>::other>;
  _ergonomics_type ergonomics;
  using _raw_sensor_orientation_type =
    geometry_msgs::msg::Quaternion_<ContainerAllocator>;
  _raw_sensor_orientation_type raw_sensor_orientation;
  using _raw_sensor_count_type =
    int32_t;
  _raw_sensor_count_type raw_sensor_count;
  using _raw_sensor_type =
    std::vector<geometry_msgs::msg::Pose_<ContainerAllocator>, typename ContainerAllocator::template rebind<geometry_msgs::msg::Pose_<ContainerAllocator>>::other>;
  _raw_sensor_type raw_sensor;

  // setters for named parameter idiom
  Type & set__glove_id(
    const int32_t & _arg)
  {
    this->glove_id = _arg;
    return *this;
  }
  Type & set__side(
    const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other> & _arg)
  {
    this->side = _arg;
    return *this;
  }
  Type & set__raw_node_count(
    const int32_t & _arg)
  {
    this->raw_node_count = _arg;
    return *this;
  }
  Type & set__raw_nodes(
    const std::vector<manus_ros2_msgs::msg::ManusRawNode_<ContainerAllocator>, typename ContainerAllocator::template rebind<manus_ros2_msgs::msg::ManusRawNode_<ContainerAllocator>>::other> & _arg)
  {
    this->raw_nodes = _arg;
    return *this;
  }
  Type & set__ergonomics_count(
    const int32_t & _arg)
  {
    this->ergonomics_count = _arg;
    return *this;
  }
  Type & set__ergonomics(
    const std::vector<manus_ros2_msgs::msg::ManusErgonomics_<ContainerAllocator>, typename ContainerAllocator::template rebind<manus_ros2_msgs::msg::ManusErgonomics_<ContainerAllocator>>::other> & _arg)
  {
    this->ergonomics = _arg;
    return *this;
  }
  Type & set__raw_sensor_orientation(
    const geometry_msgs::msg::Quaternion_<ContainerAllocator> & _arg)
  {
    this->raw_sensor_orientation = _arg;
    return *this;
  }
  Type & set__raw_sensor_count(
    const int32_t & _arg)
  {
    this->raw_sensor_count = _arg;
    return *this;
  }
  Type & set__raw_sensor(
    const std::vector<geometry_msgs::msg::Pose_<ContainerAllocator>, typename ContainerAllocator::template rebind<geometry_msgs::msg::Pose_<ContainerAllocator>>::other> & _arg)
  {
    this->raw_sensor = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    manus_ros2_msgs::msg::ManusGlove_<ContainerAllocator> *;
  using ConstRawPtr =
    const manus_ros2_msgs::msg::ManusGlove_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<manus_ros2_msgs::msg::ManusGlove_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<manus_ros2_msgs::msg::ManusGlove_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      manus_ros2_msgs::msg::ManusGlove_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<manus_ros2_msgs::msg::ManusGlove_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      manus_ros2_msgs::msg::ManusGlove_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<manus_ros2_msgs::msg::ManusGlove_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<manus_ros2_msgs::msg::ManusGlove_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<manus_ros2_msgs::msg::ManusGlove_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__manus_ros2_msgs__msg__ManusGlove
    std::shared_ptr<manus_ros2_msgs::msg::ManusGlove_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__manus_ros2_msgs__msg__ManusGlove
    std::shared_ptr<manus_ros2_msgs::msg::ManusGlove_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ManusGlove_ & other) const
  {
    if (this->glove_id != other.glove_id) {
      return false;
    }
    if (this->side != other.side) {
      return false;
    }
    if (this->raw_node_count != other.raw_node_count) {
      return false;
    }
    if (this->raw_nodes != other.raw_nodes) {
      return false;
    }
    if (this->ergonomics_count != other.ergonomics_count) {
      return false;
    }
    if (this->ergonomics != other.ergonomics) {
      return false;
    }
    if (this->raw_sensor_orientation != other.raw_sensor_orientation) {
      return false;
    }
    if (this->raw_sensor_count != other.raw_sensor_count) {
      return false;
    }
    if (this->raw_sensor != other.raw_sensor) {
      return false;
    }
    return true;
  }
  bool operator!=(const ManusGlove_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ManusGlove_

// alias to use template instance with default allocator
using ManusGlove =
  manus_ros2_msgs::msg::ManusGlove_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace manus_ros2_msgs

#endif  // MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_GLOVE__STRUCT_HPP_
