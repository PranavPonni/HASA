// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from manus_ros2_msgs:msg/ManusGlove.idl
// generated code does not contain a copyright notice
#include "manus_ros2_msgs/msg/detail/manus_glove__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `side`
#include "rosidl_runtime_c/string_functions.h"
// Member `raw_nodes`
#include "manus_ros2_msgs/msg/detail/manus_raw_node__functions.h"
// Member `ergonomics`
#include "manus_ros2_msgs/msg/detail/manus_ergonomics__functions.h"
// Member `raw_sensor_orientation`
#include "geometry_msgs/msg/detail/quaternion__functions.h"
// Member `raw_sensor`
#include "geometry_msgs/msg/detail/pose__functions.h"

bool
manus_ros2_msgs__msg__ManusGlove__init(manus_ros2_msgs__msg__ManusGlove * msg)
{
  if (!msg) {
    return false;
  }
  // glove_id
  // side
  if (!rosidl_runtime_c__String__init(&msg->side)) {
    manus_ros2_msgs__msg__ManusGlove__fini(msg);
    return false;
  }
  // raw_node_count
  // raw_nodes
  if (!manus_ros2_msgs__msg__ManusRawNode__Sequence__init(&msg->raw_nodes, 0)) {
    manus_ros2_msgs__msg__ManusGlove__fini(msg);
    return false;
  }
  // ergonomics_count
  // ergonomics
  if (!manus_ros2_msgs__msg__ManusErgonomics__Sequence__init(&msg->ergonomics, 0)) {
    manus_ros2_msgs__msg__ManusGlove__fini(msg);
    return false;
  }
  // raw_sensor_orientation
  if (!geometry_msgs__msg__Quaternion__init(&msg->raw_sensor_orientation)) {
    manus_ros2_msgs__msg__ManusGlove__fini(msg);
    return false;
  }
  // raw_sensor_count
  // raw_sensor
  if (!geometry_msgs__msg__Pose__Sequence__init(&msg->raw_sensor, 0)) {
    manus_ros2_msgs__msg__ManusGlove__fini(msg);
    return false;
  }
  return true;
}

void
manus_ros2_msgs__msg__ManusGlove__fini(manus_ros2_msgs__msg__ManusGlove * msg)
{
  if (!msg) {
    return;
  }
  // glove_id
  // side
  rosidl_runtime_c__String__fini(&msg->side);
  // raw_node_count
  // raw_nodes
  manus_ros2_msgs__msg__ManusRawNode__Sequence__fini(&msg->raw_nodes);
  // ergonomics_count
  // ergonomics
  manus_ros2_msgs__msg__ManusErgonomics__Sequence__fini(&msg->ergonomics);
  // raw_sensor_orientation
  geometry_msgs__msg__Quaternion__fini(&msg->raw_sensor_orientation);
  // raw_sensor_count
  // raw_sensor
  geometry_msgs__msg__Pose__Sequence__fini(&msg->raw_sensor);
}

bool
manus_ros2_msgs__msg__ManusGlove__are_equal(const manus_ros2_msgs__msg__ManusGlove * lhs, const manus_ros2_msgs__msg__ManusGlove * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // glove_id
  if (lhs->glove_id != rhs->glove_id) {
    return false;
  }
  // side
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->side), &(rhs->side)))
  {
    return false;
  }
  // raw_node_count
  if (lhs->raw_node_count != rhs->raw_node_count) {
    return false;
  }
  // raw_nodes
  if (!manus_ros2_msgs__msg__ManusRawNode__Sequence__are_equal(
      &(lhs->raw_nodes), &(rhs->raw_nodes)))
  {
    return false;
  }
  // ergonomics_count
  if (lhs->ergonomics_count != rhs->ergonomics_count) {
    return false;
  }
  // ergonomics
  if (!manus_ros2_msgs__msg__ManusErgonomics__Sequence__are_equal(
      &(lhs->ergonomics), &(rhs->ergonomics)))
  {
    return false;
  }
  // raw_sensor_orientation
  if (!geometry_msgs__msg__Quaternion__are_equal(
      &(lhs->raw_sensor_orientation), &(rhs->raw_sensor_orientation)))
  {
    return false;
  }
  // raw_sensor_count
  if (lhs->raw_sensor_count != rhs->raw_sensor_count) {
    return false;
  }
  // raw_sensor
  if (!geometry_msgs__msg__Pose__Sequence__are_equal(
      &(lhs->raw_sensor), &(rhs->raw_sensor)))
  {
    return false;
  }
  return true;
}

bool
manus_ros2_msgs__msg__ManusGlove__copy(
  const manus_ros2_msgs__msg__ManusGlove * input,
  manus_ros2_msgs__msg__ManusGlove * output)
{
  if (!input || !output) {
    return false;
  }
  // glove_id
  output->glove_id = input->glove_id;
  // side
  if (!rosidl_runtime_c__String__copy(
      &(input->side), &(output->side)))
  {
    return false;
  }
  // raw_node_count
  output->raw_node_count = input->raw_node_count;
  // raw_nodes
  if (!manus_ros2_msgs__msg__ManusRawNode__Sequence__copy(
      &(input->raw_nodes), &(output->raw_nodes)))
  {
    return false;
  }
  // ergonomics_count
  output->ergonomics_count = input->ergonomics_count;
  // ergonomics
  if (!manus_ros2_msgs__msg__ManusErgonomics__Sequence__copy(
      &(input->ergonomics), &(output->ergonomics)))
  {
    return false;
  }
  // raw_sensor_orientation
  if (!geometry_msgs__msg__Quaternion__copy(
      &(input->raw_sensor_orientation), &(output->raw_sensor_orientation)))
  {
    return false;
  }
  // raw_sensor_count
  output->raw_sensor_count = input->raw_sensor_count;
  // raw_sensor
  if (!geometry_msgs__msg__Pose__Sequence__copy(
      &(input->raw_sensor), &(output->raw_sensor)))
  {
    return false;
  }
  return true;
}

manus_ros2_msgs__msg__ManusGlove *
manus_ros2_msgs__msg__ManusGlove__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  manus_ros2_msgs__msg__ManusGlove * msg = (manus_ros2_msgs__msg__ManusGlove *)allocator.allocate(sizeof(manus_ros2_msgs__msg__ManusGlove), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(manus_ros2_msgs__msg__ManusGlove));
  bool success = manus_ros2_msgs__msg__ManusGlove__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
manus_ros2_msgs__msg__ManusGlove__destroy(manus_ros2_msgs__msg__ManusGlove * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    manus_ros2_msgs__msg__ManusGlove__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
manus_ros2_msgs__msg__ManusGlove__Sequence__init(manus_ros2_msgs__msg__ManusGlove__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  manus_ros2_msgs__msg__ManusGlove * data = NULL;

  if (size) {
    data = (manus_ros2_msgs__msg__ManusGlove *)allocator.zero_allocate(size, sizeof(manus_ros2_msgs__msg__ManusGlove), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = manus_ros2_msgs__msg__ManusGlove__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        manus_ros2_msgs__msg__ManusGlove__fini(&data[i - 1]);
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
manus_ros2_msgs__msg__ManusGlove__Sequence__fini(manus_ros2_msgs__msg__ManusGlove__Sequence * array)
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
      manus_ros2_msgs__msg__ManusGlove__fini(&array->data[i]);
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

manus_ros2_msgs__msg__ManusGlove__Sequence *
manus_ros2_msgs__msg__ManusGlove__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  manus_ros2_msgs__msg__ManusGlove__Sequence * array = (manus_ros2_msgs__msg__ManusGlove__Sequence *)allocator.allocate(sizeof(manus_ros2_msgs__msg__ManusGlove__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = manus_ros2_msgs__msg__ManusGlove__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
manus_ros2_msgs__msg__ManusGlove__Sequence__destroy(manus_ros2_msgs__msg__ManusGlove__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    manus_ros2_msgs__msg__ManusGlove__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
manus_ros2_msgs__msg__ManusGlove__Sequence__are_equal(const manus_ros2_msgs__msg__ManusGlove__Sequence * lhs, const manus_ros2_msgs__msg__ManusGlove__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!manus_ros2_msgs__msg__ManusGlove__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
manus_ros2_msgs__msg__ManusGlove__Sequence__copy(
  const manus_ros2_msgs__msg__ManusGlove__Sequence * input,
  manus_ros2_msgs__msg__ManusGlove__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(manus_ros2_msgs__msg__ManusGlove);
    manus_ros2_msgs__msg__ManusGlove * data =
      (manus_ros2_msgs__msg__ManusGlove *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!manus_ros2_msgs__msg__ManusGlove__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          manus_ros2_msgs__msg__ManusGlove__fini(&data[i]);
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
    if (!manus_ros2_msgs__msg__ManusGlove__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
