// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from manus_ros2_msgs:msg/ManusVibrationCommand.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "manus_ros2_msgs/msg/detail/manus_vibration_command__rosidl_typesupport_introspection_c.h"
#include "manus_ros2_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "manus_ros2_msgs/msg/detail/manus_vibration_command__functions.h"
#include "manus_ros2_msgs/msg/detail/manus_vibration_command__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void ManusVibrationCommand__rosidl_typesupport_introspection_c__ManusVibrationCommand_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  manus_ros2_msgs__msg__ManusVibrationCommand__init(message_memory);
}

void ManusVibrationCommand__rosidl_typesupport_introspection_c__ManusVibrationCommand_fini_function(void * message_memory)
{
  manus_ros2_msgs__msg__ManusVibrationCommand__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember ManusVibrationCommand__rosidl_typesupport_introspection_c__ManusVibrationCommand_message_member_array[1] = {
  {
    "intensities",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    5,  // array size
    false,  // is upper bound
    offsetof(manus_ros2_msgs__msg__ManusVibrationCommand, intensities),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers ManusVibrationCommand__rosidl_typesupport_introspection_c__ManusVibrationCommand_message_members = {
  "manus_ros2_msgs__msg",  // message namespace
  "ManusVibrationCommand",  // message name
  1,  // number of fields
  sizeof(manus_ros2_msgs__msg__ManusVibrationCommand),
  ManusVibrationCommand__rosidl_typesupport_introspection_c__ManusVibrationCommand_message_member_array,  // message members
  ManusVibrationCommand__rosidl_typesupport_introspection_c__ManusVibrationCommand_init_function,  // function to initialize message memory (memory has to be allocated)
  ManusVibrationCommand__rosidl_typesupport_introspection_c__ManusVibrationCommand_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t ManusVibrationCommand__rosidl_typesupport_introspection_c__ManusVibrationCommand_message_type_support_handle = {
  0,
  &ManusVibrationCommand__rosidl_typesupport_introspection_c__ManusVibrationCommand_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_manus_ros2_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, manus_ros2_msgs, msg, ManusVibrationCommand)() {
  if (!ManusVibrationCommand__rosidl_typesupport_introspection_c__ManusVibrationCommand_message_type_support_handle.typesupport_identifier) {
    ManusVibrationCommand__rosidl_typesupport_introspection_c__ManusVibrationCommand_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &ManusVibrationCommand__rosidl_typesupport_introspection_c__ManusVibrationCommand_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
