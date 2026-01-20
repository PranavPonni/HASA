// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from manus_ros2_msgs:msg/ManusGlove.idl
// generated code does not contain a copyright notice

#ifndef MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_GLOVE__FUNCTIONS_H_
#define MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_GLOVE__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "manus_ros2_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "manus_ros2_msgs/msg/detail/manus_glove__struct.h"

/// Initialize msg/ManusGlove message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * manus_ros2_msgs__msg__ManusGlove
 * )) before or use
 * manus_ros2_msgs__msg__ManusGlove__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
bool
manus_ros2_msgs__msg__ManusGlove__init(manus_ros2_msgs__msg__ManusGlove * msg);

/// Finalize msg/ManusGlove message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
void
manus_ros2_msgs__msg__ManusGlove__fini(manus_ros2_msgs__msg__ManusGlove * msg);

/// Create msg/ManusGlove message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * manus_ros2_msgs__msg__ManusGlove__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
manus_ros2_msgs__msg__ManusGlove *
manus_ros2_msgs__msg__ManusGlove__create();

/// Destroy msg/ManusGlove message.
/**
 * It calls
 * manus_ros2_msgs__msg__ManusGlove__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
void
manus_ros2_msgs__msg__ManusGlove__destroy(manus_ros2_msgs__msg__ManusGlove * msg);

/// Check for msg/ManusGlove message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
bool
manus_ros2_msgs__msg__ManusGlove__are_equal(const manus_ros2_msgs__msg__ManusGlove * lhs, const manus_ros2_msgs__msg__ManusGlove * rhs);

/// Copy a msg/ManusGlove message.
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
manus_ros2_msgs__msg__ManusGlove__copy(
  const manus_ros2_msgs__msg__ManusGlove * input,
  manus_ros2_msgs__msg__ManusGlove * output);

/// Initialize array of msg/ManusGlove messages.
/**
 * It allocates the memory for the number of elements and calls
 * manus_ros2_msgs__msg__ManusGlove__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
bool
manus_ros2_msgs__msg__ManusGlove__Sequence__init(manus_ros2_msgs__msg__ManusGlove__Sequence * array, size_t size);

/// Finalize array of msg/ManusGlove messages.
/**
 * It calls
 * manus_ros2_msgs__msg__ManusGlove__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
void
manus_ros2_msgs__msg__ManusGlove__Sequence__fini(manus_ros2_msgs__msg__ManusGlove__Sequence * array);

/// Create array of msg/ManusGlove messages.
/**
 * It allocates the memory for the array and calls
 * manus_ros2_msgs__msg__ManusGlove__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
manus_ros2_msgs__msg__ManusGlove__Sequence *
manus_ros2_msgs__msg__ManusGlove__Sequence__create(size_t size);

/// Destroy array of msg/ManusGlove messages.
/**
 * It calls
 * manus_ros2_msgs__msg__ManusGlove__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
void
manus_ros2_msgs__msg__ManusGlove__Sequence__destroy(manus_ros2_msgs__msg__ManusGlove__Sequence * array);

/// Check for msg/ManusGlove message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
bool
manus_ros2_msgs__msg__ManusGlove__Sequence__are_equal(const manus_ros2_msgs__msg__ManusGlove__Sequence * lhs, const manus_ros2_msgs__msg__ManusGlove__Sequence * rhs);

/// Copy an array of msg/ManusGlove messages.
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
manus_ros2_msgs__msg__ManusGlove__Sequence__copy(
  const manus_ros2_msgs__msg__ManusGlove__Sequence * input,
  manus_ros2_msgs__msg__ManusGlove__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_GLOVE__FUNCTIONS_H_
