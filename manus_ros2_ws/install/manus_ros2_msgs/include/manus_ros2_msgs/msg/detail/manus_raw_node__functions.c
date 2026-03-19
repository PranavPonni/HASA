// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from manus_ros2_msgs:msg/ManusRawNode.idl
// generated code does not contain a copyright notice
#include "manus_ros2_msgs/msg/detail/manus_raw_node__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `joint_type`
// Member `chain_type`
#include "rosidl_runtime_c/string_functions.h"
// Member `pose`
#include "geometry_msgs/msg/detail/pose__functions.h"

bool
manus_ros2_msgs__msg__ManusRawNode__init(manus_ros2_msgs__msg__ManusRawNode * msg)
{
  if (!msg) {
    return false;
  }
  // node_id
  // parent_node_id
  // joint_type
  if (!rosidl_runtime_c__String__init(&msg->joint_type)) {
    manus_ros2_msgs__msg__ManusRawNode__fini(msg);
    return false;
  }
  // chain_type
  if (!rosidl_runtime_c__String__init(&msg->chain_type)) {
    manus_ros2_msgs__msg__ManusRawNode__fini(msg);
    return false;
  }
  // pose
  if (!geometry_msgs__msg__Pose__init(&msg->pose)) {
    manus_ros2_msgs__msg__ManusRawNode__fini(msg);
    return false;
  }
  return true;
}

void
manus_ros2_msgs__msg__ManusRawNode__fini(manus_ros2_msgs__msg__ManusRawNode * msg)
{
  if (!msg) {
    return;
  }
  // node_id
  // parent_node_id
  // joint_type
  rosidl_runtime_c__String__fini(&msg->joint_type);
  // chain_type
  rosidl_runtime_c__String__fini(&msg->chain_type);
  // pose
  geometry_msgs__msg__Pose__fini(&msg->pose);
}

bool
manus_ros2_msgs__msg__ManusRawNode__are_equal(const manus_ros2_msgs__msg__ManusRawNode * lhs, const manus_ros2_msgs__msg__ManusRawNode * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // node_id
  if (lhs->node_id != rhs->node_id) {
    return false;
  }
  // parent_node_id
  if (lhs->parent_node_id != rhs->parent_node_id) {
    return false;
  }
  // joint_type
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->joint_type), &(rhs->joint_type)))
  {
    return false;
  }
  // chain_type
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->chain_type), &(rhs->chain_type)))
  {
    return false;
  }
  // pose
  if (!geometry_msgs__msg__Pose__are_equal(
      &(lhs->pose), &(rhs->pose)))
  {
    return false;
  }
  return true;
}

bool
manus_ros2_msgs__msg__ManusRawNode__copy(
  const manus_ros2_msgs__msg__ManusRawNode * input,
  manus_ros2_msgs__msg__ManusRawNode * output)
{
  if (!input || !output) {
    return false;
  }
  // node_id
  output->node_id = input->node_id;
  // parent_node_id
  output->parent_node_id = input->parent_node_id;
  // joint_type
  if (!rosidl_runtime_c__String__copy(
      &(input->joint_type), &(output->joint_type)))
  {
    return false;
  }
  // chain_type
  if (!rosidl_runtime_c__String__copy(
      &(input->chain_type), &(output->chain_type)))
  {
    return false;
  }
  // pose
  if (!geometry_msgs__msg__Pose__copy(
      &(input->pose), &(output->pose)))
  {
    return false;
  }
  return true;
}

manus_ros2_msgs__msg__ManusRawNode *
manus_ros2_msgs__msg__ManusRawNode__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  manus_ros2_msgs__msg__ManusRawNode * msg = (manus_ros2_msgs__msg__ManusRawNode *)allocator.allocate(sizeof(manus_ros2_msgs__msg__ManusRawNode), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(manus_ros2_msgs__msg__ManusRawNode));
  bool success = manus_ros2_msgs__msg__ManusRawNode__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
manus_ros2_msgs__msg__ManusRawNode__destroy(manus_ros2_msgs__msg__ManusRawNode * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    manus_ros2_msgs__msg__ManusRawNode__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
manus_ros2_msgs__msg__ManusRawNode__Sequence__init(manus_ros2_msgs__msg__ManusRawNode__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  manus_ros2_msgs__msg__ManusRawNode * data = NULL;

  if (size) {
    data = (manus_ros2_msgs__msg__ManusRawNode *)allocator.zero_allocate(size, sizeof(manus_ros2_msgs__msg__ManusRawNode), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = manus_ros2_msgs__msg__ManusRawNode__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        manus_ros2_msgs__msg__ManusRawNode__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
manus_ros2_msgs__msg__ManusRawNode__Sequence__fini(manus_ros2_msgs__msg__ManusRawNode__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      manus_ros2_msgs__msg__ManusRawNode__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

manus_ros2_msgs__msg__ManusRawNode__Sequence *
manus_ros2_msgs__msg__ManusRawNode__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  manus_ros2_msgs__msg__ManusRawNode__Sequence * array = (manus_ros2_msgs__msg__ManusRawNode__Sequence *)allocator.allocate(sizeof(manus_ros2_msgs__msg__ManusRawNode__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = manus_ros2_msgs__msg__ManusRawNode__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
manus_ros2_msgs__msg__ManusRawNode__Sequence__destroy(manus_ros2_msgs__msg__ManusRawNode__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    manus_ros2_msgs__msg__ManusRawNode__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
manus_ros2_msgs__msg__ManusRawNode__Sequence__are_equal(const manus_ros2_msgs__msg__ManusRawNode__Sequence * lhs, const manus_ros2_msgs__msg__ManusRawNode__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!manus_ros2_msgs__msg__ManusRawNode__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
manus_ros2_msgs__msg__ManusRawNode__Sequence__copy(
  const manus_ros2_msgs__msg__ManusRawNode__Sequence * input,
  manus_ros2_msgs__msg__ManusRawNode__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(manus_ros2_msgs__msg__ManusRawNode);
    manus_ros2_msgs__msg__ManusRawNode * data =
      (manus_ros2_msgs__msg__ManusRawNode *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!manus_ros2_msgs__msg__ManusRawNode__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          manus_ros2_msgs__msg__ManusRawNode__fini(&data[i]);
        }
        free(data);
        return false;
      }
    }
    output->data = data;
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!manus_ros2_msgs__msg__ManusRawNode__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
