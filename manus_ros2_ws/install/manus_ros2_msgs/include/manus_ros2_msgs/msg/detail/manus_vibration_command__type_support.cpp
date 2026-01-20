// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from manus_ros2_msgs:msg/ManusVibrationCommand.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "manus_ros2_msgs/msg/detail/manus_vibration_command__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace manus_ros2_msgs
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void ManusVibrationCommand_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) manus_ros2_msgs::msg::ManusVibrationCommand(_init);
}

void ManusVibrationCommand_fini_function(void * message_memory)
{
  auto typed_message = static_cast<manus_ros2_msgs::msg::ManusVibrationCommand *>(message_memory);
  typed_message->~ManusVibrationCommand();
}

size_t size_function__ManusVibrationCommand__intensities(const void * untyped_member)
{
  (void)untyped_member;
  return 5;
}

const void * get_const_function__ManusVibrationCommand__intensities(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<float, 5> *>(untyped_member);
  return &member[index];
}

void * get_function__ManusVibrationCommand__intensities(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<float, 5> *>(untyped_member);
  return &member[index];
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember ManusVibrationCommand_message_member_array[1] = {
  {
    "intensities",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    5,  // array size
    false,  // is upper bound
    offsetof(manus_ros2_msgs::msg::ManusVibrationCommand, intensities),  // bytes offset in struct
    nullptr,  // default value
    size_function__ManusVibrationCommand__intensities,  // size() function pointer
    get_const_function__ManusVibrationCommand__intensities,  // get_const(index) function pointer
    get_function__ManusVibrationCommand__intensities,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers ManusVibrationCommand_message_members = {
  "manus_ros2_msgs::msg",  // message namespace
  "ManusVibrationCommand",  // message name
  1,  // number of fields
  sizeof(manus_ros2_msgs::msg::ManusVibrationCommand),
  ManusVibrationCommand_message_member_array,  // message members
  ManusVibrationCommand_init_function,  // function to initialize message memory (memory has to be allocated)
  ManusVibrationCommand_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t ManusVibrationCommand_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &ManusVibrationCommand_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace manus_ros2_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<manus_ros2_msgs::msg::ManusVibrationCommand>()
{
  return &::manus_ros2_msgs::msg::rosidl_typesupport_introspection_cpp::ManusVibrationCommand_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, manus_ros2_msgs, msg, ManusVibrationCommand)() {
  return &::manus_ros2_msgs::msg::rosidl_typesupport_introspection_cpp::ManusVibrationCommand_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
