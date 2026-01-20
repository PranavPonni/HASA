// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from manus_ros2_msgs:msg/ManusGlove.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "manus_ros2_msgs/msg/detail/manus_glove__struct.hpp"
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

void ManusGlove_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) manus_ros2_msgs::msg::ManusGlove(_init);
}

void ManusGlove_fini_function(void * message_memory)
{
  auto typed_message = static_cast<manus_ros2_msgs::msg::ManusGlove *>(message_memory);
  typed_message->~ManusGlove();
}

size_t size_function__ManusGlove__raw_nodes(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<manus_ros2_msgs::msg::ManusRawNode> *>(untyped_member);
  return member->size();
}

const void * get_const_function__ManusGlove__raw_nodes(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<manus_ros2_msgs::msg::ManusRawNode> *>(untyped_member);
  return &member[index];
}

void * get_function__ManusGlove__raw_nodes(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<manus_ros2_msgs::msg::ManusRawNode> *>(untyped_member);
  return &member[index];
}

void resize_function__ManusGlove__raw_nodes(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<manus_ros2_msgs::msg::ManusRawNode> *>(untyped_member);
  member->resize(size);
}

size_t size_function__ManusGlove__ergonomics(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<manus_ros2_msgs::msg::ManusErgonomics> *>(untyped_member);
  return member->size();
}

const void * get_const_function__ManusGlove__ergonomics(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<manus_ros2_msgs::msg::ManusErgonomics> *>(untyped_member);
  return &member[index];
}

void * get_function__ManusGlove__ergonomics(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<manus_ros2_msgs::msg::ManusErgonomics> *>(untyped_member);
  return &member[index];
}

void resize_function__ManusGlove__ergonomics(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<manus_ros2_msgs::msg::ManusErgonomics> *>(untyped_member);
  member->resize(size);
}

size_t size_function__ManusGlove__raw_sensor(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<geometry_msgs::msg::Pose> *>(untyped_member);
  return member->size();
}

const void * get_const_function__ManusGlove__raw_sensor(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<geometry_msgs::msg::Pose> *>(untyped_member);
  return &member[index];
}

void * get_function__ManusGlove__raw_sensor(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<geometry_msgs::msg::Pose> *>(untyped_member);
  return &member[index];
}

void resize_function__ManusGlove__raw_sensor(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<geometry_msgs::msg::Pose> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember ManusGlove_message_member_array[9] = {
  {
    "glove_id",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(manus_ros2_msgs::msg::ManusGlove, glove_id),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "side",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(manus_ros2_msgs::msg::ManusGlove, side),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "raw_node_count",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(manus_ros2_msgs::msg::ManusGlove, raw_node_count),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "raw_nodes",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<manus_ros2_msgs::msg::ManusRawNode>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(manus_ros2_msgs::msg::ManusGlove, raw_nodes),  // bytes offset in struct
    nullptr,  // default value
    size_function__ManusGlove__raw_nodes,  // size() function pointer
    get_const_function__ManusGlove__raw_nodes,  // get_const(index) function pointer
    get_function__ManusGlove__raw_nodes,  // get(index) function pointer
    resize_function__ManusGlove__raw_nodes  // resize(index) function pointer
  },
  {
    "ergonomics_count",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(manus_ros2_msgs::msg::ManusGlove, ergonomics_count),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "ergonomics",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<manus_ros2_msgs::msg::ManusErgonomics>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(manus_ros2_msgs::msg::ManusGlove, ergonomics),  // bytes offset in struct
    nullptr,  // default value
    size_function__ManusGlove__ergonomics,  // size() function pointer
    get_const_function__ManusGlove__ergonomics,  // get_const(index) function pointer
    get_function__ManusGlove__ergonomics,  // get(index) function pointer
    resize_function__ManusGlove__ergonomics  // resize(index) function pointer
  },
  {
    "raw_sensor_orientation",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<geometry_msgs::msg::Quaternion>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(manus_ros2_msgs::msg::ManusGlove, raw_sensor_orientation),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "raw_sensor_count",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(manus_ros2_msgs::msg::ManusGlove, raw_sensor_count),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "raw_sensor",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<geometry_msgs::msg::Pose>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(manus_ros2_msgs::msg::ManusGlove, raw_sensor),  // bytes offset in struct
    nullptr,  // default value
    size_function__ManusGlove__raw_sensor,  // size() function pointer
    get_const_function__ManusGlove__raw_sensor,  // get_const(index) function pointer
    get_function__ManusGlove__raw_sensor,  // get(index) function pointer
    resize_function__ManusGlove__raw_sensor  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers ManusGlove_message_members = {
  "manus_ros2_msgs::msg",  // message namespace
  "ManusGlove",  // message name
  9,  // number of fields
  sizeof(manus_ros2_msgs::msg::ManusGlove),
  ManusGlove_message_member_array,  // message members
  ManusGlove_init_function,  // function to initialize message memory (memory has to be allocated)
  ManusGlove_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t ManusGlove_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &ManusGlove_message_members,
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
get_message_type_support_handle<manus_ros2_msgs::msg::ManusGlove>()
{
  return &::manus_ros2_msgs::msg::rosidl_typesupport_introspection_cpp::ManusGlove_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, manus_ros2_msgs, msg, ManusGlove)() {
  return &::manus_ros2_msgs::msg::rosidl_typesupport_introspection_cpp::ManusGlove_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
