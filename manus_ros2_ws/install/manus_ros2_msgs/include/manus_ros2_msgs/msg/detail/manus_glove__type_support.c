// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from manus_ros2_msgs:msg/ManusGlove.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "manus_ros2_msgs/msg/detail/manus_glove__rosidl_typesupport_introspection_c.h"
#include "manus_ros2_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "manus_ros2_msgs/msg/detail/manus_glove__functions.h"
#include "manus_ros2_msgs/msg/detail/manus_glove__struct.h"


// Include directives for member types
// Member `side`
#include "rosidl_runtime_c/string_functions.h"
// Member `raw_nodes`
#include "manus_ros2_msgs/msg/manus_raw_node.h"
// Member `raw_nodes`
#include "manus_ros2_msgs/msg/detail/manus_raw_node__rosidl_typesupport_introspection_c.h"
// Member `ergonomics`
#include "manus_ros2_msgs/msg/manus_ergonomics.h"
// Member `ergonomics`
#include "manus_ros2_msgs/msg/detail/manus_ergonomics__rosidl_typesupport_introspection_c.h"
// Member `raw_sensor_orientation`
#include "geometry_msgs/msg/quaternion.h"
// Member `raw_sensor_orientation`
#include "geometry_msgs/msg/detail/quaternion__rosidl_typesupport_introspection_c.h"
// Member `raw_sensor`
#include "geometry_msgs/msg/pose.h"
// Member `raw_sensor`
#include "geometry_msgs/msg/detail/pose__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void ManusGlove__rosidl_typesupport_introspection_c__ManusGlove_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  manus_ros2_msgs__msg__ManusGlove__init(message_memory);
}

void ManusGlove__rosidl_typesupport_introspection_c__ManusGlove_fini_function(void * message_memory)
{
  manus_ros2_msgs__msg__ManusGlove__fini(message_memory);
}

size_t ManusGlove__rosidl_typesupport_introspection_c__size_function__ManusRawNode__raw_nodes(
  const void * untyped_member)
{
  const manus_ros2_msgs__msg__ManusRawNode__Sequence * member =
    (const manus_ros2_msgs__msg__ManusRawNode__Sequence *)(untyped_member);
  return member->size;
}

const void * ManusGlove__rosidl_typesupport_introspection_c__get_const_function__ManusRawNode__raw_nodes(
  const void * untyped_member, size_t index)
{
  const manus_ros2_msgs__msg__ManusRawNode__Sequence * member =
    (const manus_ros2_msgs__msg__ManusRawNode__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ManusGlove__rosidl_typesupport_introspection_c__get_function__ManusRawNode__raw_nodes(
  void * untyped_member, size_t index)
{
  manus_ros2_msgs__msg__ManusRawNode__Sequence * member =
    (manus_ros2_msgs__msg__ManusRawNode__Sequence *)(untyped_member);
  return &member->data[index];
}

bool ManusGlove__rosidl_typesupport_introspection_c__resize_function__ManusRawNode__raw_nodes(
  void * untyped_member, size_t size)
{
  manus_ros2_msgs__msg__ManusRawNode__Sequence * member =
    (manus_ros2_msgs__msg__ManusRawNode__Sequence *)(untyped_member);
  manus_ros2_msgs__msg__ManusRawNode__Sequence__fini(member);
  return manus_ros2_msgs__msg__ManusRawNode__Sequence__init(member, size);
}

size_t ManusGlove__rosidl_typesupport_introspection_c__size_function__ManusErgonomics__ergonomics(
  const void * untyped_member)
{
  const manus_ros2_msgs__msg__ManusErgonomics__Sequence * member =
    (const manus_ros2_msgs__msg__ManusErgonomics__Sequence *)(untyped_member);
  return member->size;
}

const void * ManusGlove__rosidl_typesupport_introspection_c__get_const_function__ManusErgonomics__ergonomics(
  const void * untyped_member, size_t index)
{
  const manus_ros2_msgs__msg__ManusErgonomics__Sequence * member =
    (const manus_ros2_msgs__msg__ManusErgonomics__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ManusGlove__rosidl_typesupport_introspection_c__get_function__ManusErgonomics__ergonomics(
  void * untyped_member, size_t index)
{
  manus_ros2_msgs__msg__ManusErgonomics__Sequence * member =
    (manus_ros2_msgs__msg__ManusErgonomics__Sequence *)(untyped_member);
  return &member->data[index];
}

bool ManusGlove__rosidl_typesupport_introspection_c__resize_function__ManusErgonomics__ergonomics(
  void * untyped_member, size_t size)
{
  manus_ros2_msgs__msg__ManusErgonomics__Sequence * member =
    (manus_ros2_msgs__msg__ManusErgonomics__Sequence *)(untyped_member);
  manus_ros2_msgs__msg__ManusErgonomics__Sequence__fini(member);
  return manus_ros2_msgs__msg__ManusErgonomics__Sequence__init(member, size);
}

size_t ManusGlove__rosidl_typesupport_introspection_c__size_function__Pose__raw_sensor(
  const void * untyped_member)
{
  const geometry_msgs__msg__Pose__Sequence * member =
    (const geometry_msgs__msg__Pose__Sequence *)(untyped_member);
  return member->size;
}

const void * ManusGlove__rosidl_typesupport_introspection_c__get_const_function__Pose__raw_sensor(
  const void * untyped_member, size_t index)
{
  const geometry_msgs__msg__Pose__Sequence * member =
    (const geometry_msgs__msg__Pose__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ManusGlove__rosidl_typesupport_introspection_c__get_function__Pose__raw_sensor(
  void * untyped_member, size_t index)
{
  geometry_msgs__msg__Pose__Sequence * member =
    (geometry_msgs__msg__Pose__Sequence *)(untyped_member);
  return &member->data[index];
}

bool ManusGlove__rosidl_typesupport_introspection_c__resize_function__Pose__raw_sensor(
  void * untyped_member, size_t size)
{
  geometry_msgs__msg__Pose__Sequence * member =
    (geometry_msgs__msg__Pose__Sequence *)(untyped_member);
  geometry_msgs__msg__Pose__Sequence__fini(member);
  return geometry_msgs__msg__Pose__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember ManusGlove__rosidl_typesupport_introspection_c__ManusGlove_message_member_array[9] = {
  {
    "glove_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(manus_ros2_msgs__msg__ManusGlove, glove_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "side",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(manus_ros2_msgs__msg__ManusGlove, side),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "raw_node_count",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(manus_ros2_msgs__msg__ManusGlove, raw_node_count),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "raw_nodes",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(manus_ros2_msgs__msg__ManusGlove, raw_nodes),  // bytes offset in struct
    NULL,  // default value
    ManusGlove__rosidl_typesupport_introspection_c__size_function__ManusRawNode__raw_nodes,  // size() function pointer
    ManusGlove__rosidl_typesupport_introspection_c__get_const_function__ManusRawNode__raw_nodes,  // get_const(index) function pointer
    ManusGlove__rosidl_typesupport_introspection_c__get_function__ManusRawNode__raw_nodes,  // get(index) function pointer
    ManusGlove__rosidl_typesupport_introspection_c__resize_function__ManusRawNode__raw_nodes  // resize(index) function pointer
  },
  {
    "ergonomics_count",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(manus_ros2_msgs__msg__ManusGlove, ergonomics_count),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "ergonomics",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(manus_ros2_msgs__msg__ManusGlove, ergonomics),  // bytes offset in struct
    NULL,  // default value
    ManusGlove__rosidl_typesupport_introspection_c__size_function__ManusErgonomics__ergonomics,  // size() function pointer
    ManusGlove__rosidl_typesupport_introspection_c__get_const_function__ManusErgonomics__ergonomics,  // get_const(index) function pointer
    ManusGlove__rosidl_typesupport_introspection_c__get_function__ManusErgonomics__ergonomics,  // get(index) function pointer
    ManusGlove__rosidl_typesupport_introspection_c__resize_function__ManusErgonomics__ergonomics  // resize(index) function pointer
  },
  {
    "raw_sensor_orientation",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(manus_ros2_msgs__msg__ManusGlove, raw_sensor_orientation),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "raw_sensor_count",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(manus_ros2_msgs__msg__ManusGlove, raw_sensor_count),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "raw_sensor",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(manus_ros2_msgs__msg__ManusGlove, raw_sensor),  // bytes offset in struct
    NULL,  // default value
    ManusGlove__rosidl_typesupport_introspection_c__size_function__Pose__raw_sensor,  // size() function pointer
    ManusGlove__rosidl_typesupport_introspection_c__get_const_function__Pose__raw_sensor,  // get_const(index) function pointer
    ManusGlove__rosidl_typesupport_introspection_c__get_function__Pose__raw_sensor,  // get(index) function pointer
    ManusGlove__rosidl_typesupport_introspection_c__resize_function__Pose__raw_sensor  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers ManusGlove__rosidl_typesupport_introspection_c__ManusGlove_message_members = {
  "manus_ros2_msgs__msg",  // message namespace
  "ManusGlove",  // message name
  9,  // number of fields
  sizeof(manus_ros2_msgs__msg__ManusGlove),
  ManusGlove__rosidl_typesupport_introspection_c__ManusGlove_message_member_array,  // message members
  ManusGlove__rosidl_typesupport_introspection_c__ManusGlove_init_function,  // function to initialize message memory (memory has to be allocated)
  ManusGlove__rosidl_typesupport_introspection_c__ManusGlove_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t ManusGlove__rosidl_typesupport_introspection_c__ManusGlove_message_type_support_handle = {
  0,
  &ManusGlove__rosidl_typesupport_introspection_c__ManusGlove_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_manus_ros2_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, manus_ros2_msgs, msg, ManusGlove)() {
  ManusGlove__rosidl_typesupport_introspection_c__ManusGlove_message_member_array[3].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, manus_ros2_msgs, msg, ManusRawNode)();
  ManusGlove__rosidl_typesupport_introspection_c__ManusGlove_message_member_array[5].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, manus_ros2_msgs, msg, ManusErgonomics)();
  ManusGlove__rosidl_typesupport_introspection_c__ManusGlove_message_member_array[6].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Quaternion)();
  ManusGlove__rosidl_typesupport_introspection_c__ManusGlove_message_member_array[8].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Pose)();
  if (!ManusGlove__rosidl_typesupport_introspection_c__ManusGlove_message_type_support_handle.typesupport_identifier) {
    ManusGlove__rosidl_typesupport_introspection_c__ManusGlove_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &ManusGlove__rosidl_typesupport_introspection_c__ManusGlove_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
