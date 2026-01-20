// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from manus_ros2_msgs:msg/ManusVibrationCommand.idl
// generated code does not contain a copyright notice

#ifndef MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_VIBRATION_COMMAND__STRUCT_HPP_
#define MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_VIBRATION_COMMAND__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


#ifndef _WIN32
# define DEPRECATED__manus_ros2_msgs__msg__ManusVibrationCommand __attribute__((deprecated))
#else
# define DEPRECATED__manus_ros2_msgs__msg__ManusVibrationCommand __declspec(deprecated)
#endif

namespace manus_ros2_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ManusVibrationCommand_
{
  using Type = ManusVibrationCommand_<ContainerAllocator>;

  explicit ManusVibrationCommand_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<float, 5>::iterator, float>(this->intensities.begin(), this->intensities.end(), 0.0f);
    }
  }

  explicit ManusVibrationCommand_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : intensities(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<float, 5>::iterator, float>(this->intensities.begin(), this->intensities.end(), 0.0f);
    }
  }

  // field types and members
  using _intensities_type =
    std::array<float, 5>;
  _intensities_type intensities;

  // setters for named parameter idiom
  Type & set__intensities(
    const std::array<float, 5> & _arg)
  {
    this->intensities = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    manus_ros2_msgs::msg::ManusVibrationCommand_<ContainerAllocator> *;
  using ConstRawPtr =
    const manus_ros2_msgs::msg::ManusVibrationCommand_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<manus_ros2_msgs::msg::ManusVibrationCommand_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<manus_ros2_msgs::msg::ManusVibrationCommand_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      manus_ros2_msgs::msg::ManusVibrationCommand_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<manus_ros2_msgs::msg::ManusVibrationCommand_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      manus_ros2_msgs::msg::ManusVibrationCommand_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<manus_ros2_msgs::msg::ManusVibrationCommand_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<manus_ros2_msgs::msg::ManusVibrationCommand_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<manus_ros2_msgs::msg::ManusVibrationCommand_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__manus_ros2_msgs__msg__ManusVibrationCommand
    std::shared_ptr<manus_ros2_msgs::msg::ManusVibrationCommand_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__manus_ros2_msgs__msg__ManusVibrationCommand
    std::shared_ptr<manus_ros2_msgs::msg::ManusVibrationCommand_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ManusVibrationCommand_ & other) const
  {
    if (this->intensities != other.intensities) {
      return false;
    }
    return true;
  }
  bool operator!=(const ManusVibrationCommand_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ManusVibrationCommand_

// alias to use template instance with default allocator
using ManusVibrationCommand =
  manus_ros2_msgs::msg::ManusVibrationCommand_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace manus_ros2_msgs

#endif  // MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_VIBRATION_COMMAND__STRUCT_HPP_
