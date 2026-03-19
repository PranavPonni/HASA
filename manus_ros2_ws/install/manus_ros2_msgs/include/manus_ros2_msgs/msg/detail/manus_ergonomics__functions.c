// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from manus_ros2_msgs:msg/ManusErgonomics.idl
// generated code does not contain a copyright notice
#include "manus_ros2_msgs/msg/detail/manus_ergonomics__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `type`
#include "rosidl_runtime_c/string_functions.h"

bool
manus_ros2_msgs__msg__ManusErgonomics__init(manus_ros2_msgs__msg__ManusErgonomics * msg)
{
  if (!msg) {
    return false;
  }
  // type
  if (!rosidl_runtime_c__String__init(&msg->type)) {
    manus_ros2_msgs__msg__ManusErgonomics__fini(msg);
    return false;
  }
  // value
  return true;
}

void
manus_ros2_msgs__msg__ManusErgonomics__fini(manus_ros2_msgs__msg__ManusErgonomics * msg)
{
  if (!msg) {
    return;
  }
  // type
  rosidl_runtime_c__String__fini(&msg->type);
  // value
}

bool
manus_ros2_msgs__msg__ManusErgonomics__are_equal(const manus_ros2_msgs__msg__ManusErgonomics * lhs, const manus_ros2_msgs__msg__ManusErgonomics * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // type
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->type), &(rhs->type)))
  {
    return false;
  }
  // value
  if (lhs->value != rhs->value) {
    return false;
  }
  return true;
}

bool
manus_ros2_msgs__msg__ManusErgonomics__copy(
  const manus_ros2_msgs__msg__ManusErgonomics * input,
  manus_ros2_msgs__msg__ManusErgonomics * output)
{
  if (!input || !output) {
    return false;
  }
  // type
  if (!rosidl_runtime_c__String__copy(
      &(input->type), &(output->type)))
  {
    return false;
  }
  // value
  output->value = input->value;
  return true;
}

manus_ros2_msgs__msg__ManusErgonomics *
manus_ros2_msgs__msg__ManusErgonomics__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  manus_ros2_msgs__msg__ManusErgonomics * msg = (manus_ros2_msgs__msg__ManusErgonomics *)allocator.allocate(sizeof(manus_ros2_msgs__msg__ManusErgonomics), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(manus_ros2_msgs__msg__ManusErgonomics));
  bool success = manus_ros2_msgs__msg__ManusErgonomics__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
manus_ros2_msgs__msg__ManusErgonomics__destroy(manus_ros2_msgs__msg__ManusErgonomics * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    manus_ros2_msgs__msg__ManusErgonomics__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
manus_ros2_msgs__msg__ManusErgonomics__Sequence__init(manus_ros2_msgs__msg__ManusErgonomics__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  manus_ros2_msgs__msg__ManusErgonomics * data = NULL;

  if (size) {
    data = (manus_ros2_msgs__msg__ManusErgonomics *)allocator.zero_allocate(size, sizeof(manus_ros2_msgs__msg__ManusErgonomics), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = manus_ros2_msgs__msg__ManusErgonomics__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        manus_ros2_msgs__msg__ManusErgonomics__fini(&data[i - 1]);
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
manus_ros2_msgs__msg__ManusErgonomics__Sequence__fini(manus_ros2_msgs__msg__ManusErgonomics__Sequence * array)
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
      manus_ros2_msgs__msg__ManusErgonomics__fini(&array->data[i]);
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

manus_ros2_msgs__msg__ManusErgonomics__Sequence *
manus_ros2_msgs__msg__ManusErgonomics__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  manus_ros2_msgs__msg__ManusErgonomics__Sequence * array = (manus_ros2_msgs__msg__ManusErgonomics__Sequence *)allocator.allocate(sizeof(manus_ros2_msgs__msg__ManusErgonomics__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = manus_ros2_msgs__msg__ManusErgonomics__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
manus_ros2_msgs__msg__ManusErgonomics__Sequence__destroy(manus_ros2_msgs__msg__ManusErgonomics__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    manus_ros2_msgs__msg__ManusErgonomics__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
manus_ros2_msgs__msg__ManusErgonomics__Sequence__are_equal(const manus_ros2_msgs__msg__ManusErgonomics__Sequence * lhs, const manus_ros2_msgs__msg__ManusErgonomics__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!manus_ros2_msgs__msg__ManusErgonomics__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
manus_ros2_msgs__msg__ManusErgonomics__Sequence__copy(
  const manus_ros2_msgs__msg__ManusErgonomics__Sequence * input,
  manus_ros2_msgs__msg__ManusErgonomics__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(manus_ros2_msgs__msg__ManusErgonomics);
    manus_ros2_msgs__msg__ManusErgonomics * data =
      (manus_ros2_msgs__msg__ManusErgonomics *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!manus_ros2_msgs__msg__ManusErgonomics__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          manus_ros2_msgs__msg__ManusErgonomics__fini(&data[i]);
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
    if (!manus_ros2_msgs__msg__ManusErgonomics__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
