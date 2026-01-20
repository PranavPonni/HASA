// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from manus_ros2_msgs:msg/ManusRawNode.idl
// generated code does not contain a copyright notice

#ifndef MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_RAW_NODE__FUNCTIONS_H_
#define MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_RAW_NODE__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "manus_ros2_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "manus_ros2_msgs/msg/detail/manus_raw_node__struct.h"

/// Initialize msg/ManusRawNode message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * manus_ros2_msgs__msg__ManusRawNode
 * )) before or use
 * manus_ros2_msgs__msg__ManusRawNode__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
bool
manus_ros2_msgs__msg__ManusRawNode__init(manus_ros2_msgs__msg__ManusRawNode * msg);

/// Finalize msg/ManusRawNode message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
void
manus_ros2_msgs__msg__ManusRawNode__fini(manus_ros2_msgs__msg__ManusRawNode * msg);

/// Create msg/ManusRawNode message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * manus_ros2_msgs__msg__ManusRawNode__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
manus_ros2_msgs__msg__ManusRawNode *
manus_ros2_msgs__msg__ManusRawNode__create();

/// Destroy msg/ManusRawNode message.
/**
 * It calls
 * manus_ros2_msgs__msg__ManusRawNode__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
void
manus_ros2_msgs__msg__ManusRawNode__destroy(manus_ros2_msgs__msg__ManusRawNode * msg);

/// Check for msg/ManusRawNode message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
bool
manus_ros2_msgs__msg__ManusRawNode__are_equal(const manus_ros2_msgs__msg__ManusRawNode * lhs, const manus_ros2_msgs__msg__ManusRawNode * rhs);

/// Copy a msg/ManusRawNode message.
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
manus_ros2_msgs__msg__ManusRawNode__copy(
  const manus_ros2_msgs__msg__ManusRawNode * input,
  manus_ros2_msgs__msg__ManusRawNode * output);

/// Initialize array of msg/ManusRawNode messages.
/**
 * It allocates the memory for the number of elements and calls
 * manus_ros2_msgs__msg__ManusRawNode__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
bool
manus_ros2_msgs__msg__ManusRawNode__Sequence__init(manus_ros2_msgs__msg__ManusRawNode__Sequence * array, size_t size);

/// Finalize array of msg/ManusRawNode messages.
/**
 * It calls
 * manus_ros2_msgs__msg__ManusRawNode__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
void
manus_ros2_msgs__msg__ManusRawNode__Sequence__fini(manus_ros2_msgs__msg__ManusRawNode__Sequence * array);

/// Create array of msg/ManusRawNode messages.
/**
 * It allocates the memory for the array and calls
 * manus_ros2_msgs__msg__ManusRawNode__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
manus_ros2_msgs__msg__ManusRawNode__Sequence *
manus_ros2_msgs__msg__ManusRawNode__Sequence__create(size_t size);

/// Destroy array of msg/ManusRawNode messages.
/**
 * It calls
 * manus_ros2_msgs__msg__ManusRawNode__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
void
manus_ros2_msgs__msg__ManusRawNode__Sequence__destroy(manus_ros2_msgs__msg__ManusRawNode__Sequence * array);

/// Check for msg/ManusRawNode message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
bool
manus_ros2_msgs__msg__ManusRawNode__Sequence__are_equal(const manus_ros2_msgs__msg__ManusRawNode__Sequence * lhs, const manus_ros2_msgs__msg__ManusRawNode__Sequence * rhs);

/// Copy an array of msg/ManusRawNode messages.
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
manus_ros2_msgs__msg__ManusRawNode__Sequence__copy(
  const manus_ros2_msgs__msg__ManusRawNode__Sequence * input,
  manus_ros2_msgs__msg__ManusRawNode__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_RAW_NODE__FUNCTIONS_H_
