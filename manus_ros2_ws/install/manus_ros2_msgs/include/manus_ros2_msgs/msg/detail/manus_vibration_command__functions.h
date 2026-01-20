// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from manus_ros2_msgs:msg/ManusVibrationCommand.idl
// generated code does not contain a copyright notice

#ifndef MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_VIBRATION_COMMAND__FUNCTIONS_H_
#define MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_VIBRATION_COMMAND__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "manus_ros2_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "manus_ros2_msgs/msg/detail/manus_vibration_command__struct.h"

/// Initialize msg/ManusVibrationCommand message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * manus_ros2_msgs__msg__ManusVibrationCommand
 * )) before or use
 * manus_ros2_msgs__msg__ManusVibrationCommand__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
bool
manus_ros2_msgs__msg__ManusVibrationCommand__init(manus_ros2_msgs__msg__ManusVibrationCommand * msg);

/// Finalize msg/ManusVibrationCommand message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
void
manus_ros2_msgs__msg__ManusVibrationCommand__fini(manus_ros2_msgs__msg__ManusVibrationCommand * msg);

/// Create msg/ManusVibrationCommand message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * manus_ros2_msgs__msg__ManusVibrationCommand__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
manus_ros2_msgs__msg__ManusVibrationCommand *
manus_ros2_msgs__msg__ManusVibrationCommand__create();

/// Destroy msg/ManusVibrationCommand message.
/**
 * It calls
 * manus_ros2_msgs__msg__ManusVibrationCommand__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
void
manus_ros2_msgs__msg__ManusVibrationCommand__destroy(manus_ros2_msgs__msg__ManusVibrationCommand * msg);

/// Check for msg/ManusVibrationCommand message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
bool
manus_ros2_msgs__msg__ManusVibrationCommand__are_equal(const manus_ros2_msgs__msg__ManusVibrationCommand * lhs, const manus_ros2_msgs__msg__ManusVibrationCommand * rhs);

/// Copy a msg/ManusVibrationCommand message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
bool
manus_ros2_msgs__msg__ManusVibrationCommand__copy(
  const manus_ros2_msgs__msg__ManusVibrationCommand * input,
  manus_ros2_msgs__msg__ManusVibrationCommand * output);

/// Initialize array of msg/ManusVibrationCommand messages.
/**
 * It allocates the memory for the number of elements and calls
 * manus_ros2_msgs__msg__ManusVibrationCommand__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
bool
manus_ros2_msgs__msg__ManusVibrationCommand__Sequence__init(manus_ros2_msgs__msg__ManusVibrationCommand__Sequence * array, size_t size);

/// Finalize array of msg/ManusVibrationCommand messages.
/**
 * It calls
 * manus_ros2_msgs__msg__ManusVibrationCommand__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
void
manus_ros2_msgs__msg__ManusVibrationCommand__Sequence__fini(manus_ros2_msgs__msg__ManusVibrationCommand__Sequence * array);

/// Create array of msg/ManusVibrationCommand messages.
/**
 * It allocates the memory for the array and calls
 * manus_ros2_msgs__msg__ManusVibrationCommand__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
manus_ros2_msgs__msg__ManusVibrationCommand__Sequence *
manus_ros2_msgs__msg__ManusVibrationCommand__Sequence__create(size_t size);

/// Destroy array of msg/ManusVibrationCommand messages.
/**
 * It calls
 * manus_ros2_msgs__msg__ManusVibrationCommand__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
void
manus_ros2_msgs__msg__ManusVibrationCommand__Sequence__destroy(manus_ros2_msgs__msg__ManusVibrationCommand__Sequence * array);

/// Check for msg/ManusVibrationCommand message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
bool
manus_ros2_msgs__msg__ManusVibrationCommand__Sequence__are_equal(const manus_ros2_msgs__msg__ManusVibrationCommand__Sequence * lhs, const manus_ros2_msgs__msg__ManusVibrationCommand__Sequence * rhs);

/// Copy an array of msg/ManusVibrationCommand messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
bool
manus_ros2_msgs__msg__ManusVibrationCommand__Sequence__copy(
  const manus_ros2_msgs__msg__ManusVibrationCommand__Sequence * input,
  manus_ros2_msgs__msg__ManusVibrationCommand__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_VIBRATION_COMMAND__FUNCTIONS_H_
