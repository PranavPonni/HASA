// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from manus_ros2_msgs:msg/ManusErgonomics.idl
// generated code does not contain a copyright notice

#ifndef MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_ERGONOMICS__STRUCT_HPP_
#define MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_ERGONOMICS__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


#ifndef _WIN32
# define DEPRECATED__manus_ros2_msgs__msg__ManusErgonomics __attribute__((deprecated))
#else
# define DEPRECATED__manus_ros2_msgs__msg__ManusErgonomics __declspec(deprecated)
#endif

namespace manus_ros2_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ManusErgonomics_
{
  using Type = ManusErgonomics_<ContainerAllocator>;

  explicit ManusErgonomics_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->type = "";
      this->value = 0.0f;
    }
  }

  explicit ManusErgonomics_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : type(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->type = "";
      this->value = 0.0f;
    }
  }

  // field types and members
  using _type_type =
    std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other>;
  _type_type type;
  using _value_type =
    float;
  _value_type value;

  // setters for named parameter idiom
  Type & set__type(
    const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other> & _arg)
  {
    this->type = _arg;
    return *this;
  }
  Type & set__value(
    const float & _arg)
  {
    this->value = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    manus_ros2_msgs::msg::ManusErgonomics_<ContainerAllocator> *;
  using ConstRawPtr =
    const manus_ros2_msgs::msg::ManusErgonomics_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<manus_ros2_msgs::msg::ManusErgonomics_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<manus_ros2_msgs::msg::ManusErgonomics_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      manus_ros2_msgs::msg::ManusErgonomics_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<manus_ros2_msgs::msg::ManusErgonomics_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      manus_ros2_msgs::msg::ManusErgonomics_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<manus_ros2_msgs::msg::ManusErgonomics_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<manus_ros2_msgs::msg::ManusErgonomics_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<manus_ros2_msgs::msg::ManusErgonomics_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__manus_ros2_msgs__msg__ManusErgonomics
    std::shared_ptr<manus_ros2_msgs::msg::ManusErgonomics_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__manus_ros2_msgs__msg__ManusErgonomics
    std::shared_ptr<manus_ros2_msgs::msg::ManusErgonomics_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ManusErgonomics_ & other) const
  {
    if (this->type != other.type) {
      return false;
    }
    if (this->value != other.value) {
      return false;
    }
    return true;
  }
  bool operator!=(const ManusErgonomics_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ManusErgonomics_

// alias to use template instance with default allocator
using ManusErgonomics =
  manus_ros2_msgs::msg::ManusErgonomics_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace manus_ros2_msgs

#endif  // MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_ERGONOMICS__STRUCT_HPP_
