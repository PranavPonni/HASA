// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from manus_ros2_msgs:msg/ManusVibrationCommand.idl
// generated code does not contain a copyright notice
#include "manus_ros2_msgs/msg/detail/manus_vibration_command__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
manus_ros2_msgs__msg__ManusVibrationCommand__init(manus_ros2_msgs__msg__ManusVibrationCommand * msg)
{
  if (!msg) {
    return false;
  }
  // intensities
  return true;
}

void
manus_ros2_msgs__msg__ManusVibrationCommand__fini(manus_ros2_msgs__msg__ManusVibrationCommand * msg)
{
  if (!msg) {
    return;
  }
  // intensities
}

bool
manus_ros2_msgs__msg__ManusVibrationCommand__are_equal(const manus_ros2_msgs__msg__ManusVibrationCommand * lhs, const manus_ros2_msgs__msg__ManusVibrationCommand * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // intensities
  for (size_t i = 0; i < 5; ++i) {
    if (lhs->intensities[i] != rhs->intensities[i]) {
      return false;
    }
  }
  return true;
}

bool
manus_ros2_msgs__msg__ManusVibrationCommand__copy(
  const manus_ros2_msgs__msg__ManusVibrationCommand * input,
  manus_ros2_msgs__msg__ManusVibrationCommand * output)
{
  if (!input || !output) {
    return false;
  }
  // intensities
  for (size_t i = 0; i < 5; ++i) {
    output->intensities[i] = input->intensities[i];
  }
  return true;
}

manus_ros2_msgs__msg__ManusVibrationCommand *
manus_ros2_msgs__msg__ManusVibrationCommand__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  manus_ros2_msgs__msg__ManusVibrationCommand * msg = (manus_ros2_msgs__msg__ManusVibrationCommand *)allocator.allocate(sizeof(manus_ros2_msgs__msg__ManusVibrationCommand), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(manus_ros2_msgs__msg__ManusVibrationCommand));
  bool success = manus_ros2_msgs__msg__ManusVibrationCommand__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
manus_ros2_msgs__msg__ManusVibrationCommand__destroy(manus_ros2_msgs__msg__ManusVibrationCommand * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    manus_ros2_msgs__msg__ManusVibrationCommand__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
manus_ros2_msgs__msg__ManusVibrationCommand__Sequence__init(manus_ros2_msgs__msg__ManusVibrationCommand__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  manus_ros2_msgs__msg__ManusVibrationCommand * data = NULL;

  if (size) {
    data = (manus_ros2_msgs__msg__ManusVibrationCommand *)allocator.zero_allocate(size, sizeof(manus_ros2_msgs__msg__ManusVibrationCommand), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = manus_ros2_msgs__msg__ManusVibrationCommand__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        manus_ros2_msgs__msg__ManusVibrationCommand__fini(&data[i - 1]);
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
manus_ros2_msgs__msg__ManusVibrationCommand__Sequence__fini(manus_ros2_msgs__msg__ManusVibrationCommand__Sequence * array)
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
      manus_ros2_msgs__msg__ManusVibrationCommand__fini(&array->data[i]);
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

manus_ros2_msgs__msg__ManusVibrationCommand__Sequence *
manus_ros2_msgs__msg__ManusVibrationCommand__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  manus_ros2_msgs__msg__ManusVibrationCommand__Sequence * array = (manus_ros2_msgs__msg__ManusVibrationCommand__Sequence *)allocator.allocate(sizeof(manus_ros2_msgs__msg__ManusVibrationCommand__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = manus_ros2_msgs__msg__ManusVibrationCommand__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
manus_ros2_msgs__msg__ManusVibrationCommand__Sequence__destroy(manus_ros2_msgs__msg__ManusVibrationCommand__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    manus_ros2_msgs__msg__ManusVibrationCommand__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
manus_ros2_msgs__msg__ManusVibrationCommand__Sequence__are_equal(const manus_ros2_msgs__msg__ManusVibrationCommand__Sequence * lhs, const manus_ros2_msgs__msg__ManusVibrationCommand__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!manus_ros2_msgs__msg__ManusVibrationCommand__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
manus_ros2_msgs__msg__ManusVibrationCommand__Sequence__copy(
  const manus_ros2_msgs__msg__ManusVibrationCommand__Sequence * input,
  manus_ros2_msgs__msg__ManusVibrationCommand__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(manus_ros2_msgs__msg__ManusVibrationCommand);
    manus_ros2_msgs__msg__ManusVibrationCommand * data =
      (manus_ros2_msgs__msg__ManusVibrationCommand *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!manus_ros2_msgs__msg__ManusVibrationCommand__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          manus_ros2_msgs__msg__ManusVibrationCommand__fini(&data[i]);
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
    if (!manus_ros2_msgs__msg__ManusVibrationCommand__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
