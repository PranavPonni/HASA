// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from manus_ros2_msgs:msg/ManusErgonomics.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "manus_ros2_msgs/msg/detail/manus_ergonomics__struct.hpp"
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

void ManusErgonomics_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) manus_ros2_msgs::msg::ManusErgonomics(_init);
}

void ManusErgonomics_fini_function(void * message_memory)
{
  auto typed_message = static_cast<manus_ros2_msgs::msg::ManusErgonomics *>(message_memory);
  typed_message->~ManusErgonomics();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember ManusErgonomics_message_member_array[2] = {
  {
    "type",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(manus_ros2_msgs::msg::ManusErgonomics, type),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "value",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(manus_ros2_msgs::msg::ManusErgonomics, value),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers ManusErgonomics_message_members = {
  "manus_ros2_msgs::msg",  // message namespace
  "ManusErgonomics",  // message name
  2,  // number of fields
  sizeof(manus_ros2_msgs::msg::ManusErgonomics),
  ManusErgonomics_message_member_array,  // message members
  ManusErgonomics_init_function,  // function to initialize message memory (memory has to be allocated)
  ManusErgonomics_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t ManusErgonomics_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &ManusErgonomics_message_members,
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
get_message_type_support_handle<manus_ros2_msgs::msg::ManusErgonomics>()
{
  return &::manus_ros2_msgs::msg::rosidl_typesupport_introspection_cpp::ManusErgonomics_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, manus_ros2_msgs, msg, ManusErgonomics)() {
  return &::manus_ros2_msgs::msg::rosidl_typesupport_introspection_cpp::ManusErgonomics_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
