// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from manus_ros2_msgs:msg/ManusGlove.idl
// generated code does not contain a copyright notice
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdbool.h>
#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "numpy/ndarrayobject.h"
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
#include "rosidl_runtime_c/visibility_control.h"
#include "manus_ros2_msgs/msg/detail/manus_glove__struct.h"
#include "manus_ros2_msgs/msg/detail/manus_glove__functions.h"

#include "rosidl_runtime_c/string.h"
#include "rosidl_runtime_c/string_functions.h"

#include "rosidl_runtime_c/primitives_sequence.h"
#include "rosidl_runtime_c/primitives_sequence_functions.h"

// Nested array functions includes
#include "geometry_msgs/msg/detail/pose__functions.h"
#include "manus_ros2_msgs/msg/detail/manus_ergonomics__functions.h"
#include "manus_ros2_msgs/msg/detail/manus_raw_node__functions.h"
// end nested array functions include
bool manus_ros2_msgs__msg__manus_raw_node__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * manus_ros2_msgs__msg__manus_raw_node__convert_to_py(void * raw_ros_message);
bool manus_ros2_msgs__msg__manus_ergonomics__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * manus_ros2_msgs__msg__manus_ergonomics__convert_to_py(void * raw_ros_message);
ROSIDL_GENERATOR_C_IMPORT
bool geometry_msgs__msg__quaternion__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * geometry_msgs__msg__quaternion__convert_to_py(void * raw_ros_message);
ROSIDL_GENERATOR_C_IMPORT
bool geometry_msgs__msg__pose__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * geometry_msgs__msg__pose__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool manus_ros2_msgs__msg__manus_glove__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[44];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("manus_ros2_msgs.msg._manus_glove.ManusGlove", full_classname_dest, 43) == 0);
  }
  manus_ros2_msgs__msg__ManusGlove * ros_message = _ros_message;
  {  // glove_id
    PyObject * field = PyObject_GetAttrString(_pymsg, "glove_id");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->glove_id = (int32_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // side
    PyObject * field = PyObject_GetAttrString(_pymsg, "side");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->side, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }
  {  // raw_node_count
    PyObject * field = PyObject_GetAttrString(_pymsg, "raw_node_count");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->raw_node_count = (int32_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // raw_nodes
    PyObject * field = PyObject_GetAttrString(_pymsg, "raw_nodes");
    if (!field) {
      return false;
    }
    PyObject * seq_field = PySequence_Fast(field, "expected a sequence in 'raw_nodes'");
    if (!seq_field) {
      Py_DECREF(field);
      return false;
    }
    Py_ssize_t size = PySequence_Size(field);
    if (-1 == size) {
      Py_DECREF(seq_field);
      Py_DECREF(field);
      return false;
    }
    if (!manus_ros2_msgs__msg__ManusRawNode__Sequence__init(&(ros_message->raw_nodes), size)) {
      PyErr_SetString(PyExc_RuntimeError, "unable to create manus_ros2_msgs__msg__ManusRawNode__Sequence ros_message");
      Py_DECREF(seq_field);
      Py_DECREF(field);
      return false;
    }
    manus_ros2_msgs__msg__ManusRawNode * dest = ros_message->raw_nodes.data;
    for (Py_ssize_t i = 0; i < size; ++i) {
      if (!manus_ros2_msgs__msg__manus_raw_node__convert_from_py(PySequence_Fast_GET_ITEM(seq_field, i), &dest[i])) {
        Py_DECREF(seq_field);
        Py_DECREF(field);
        return false;
      }
    }
    Py_DECREF(seq_field);
    Py_DECREF(field);
  }
  {  // ergonomics_count
    PyObject * field = PyObject_GetAttrString(_pymsg, "ergonomics_count");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->ergonomics_count = (int32_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // ergonomics
    PyObject * field = PyObject_GetAttrString(_pymsg, "ergonomics");
    if (!field) {
      return false;
    }
    PyObject * seq_field = PySequence_Fast(field, "expected a sequence in 'ergonomics'");
    if (!seq_field) {
      Py_DECREF(field);
      return false;
    }
    Py_ssize_t size = PySequence_Size(field);
    if (-1 == size) {
      Py_DECREF(seq_field);
      Py_DECREF(field);
      return false;
    }
    if (!manus_ros2_msgs__msg__ManusErgonomics__Sequence__init(&(ros_message->ergonomics), size)) {
      PyErr_SetString(PyExc_RuntimeError, "unable to create manus_ros2_msgs__msg__ManusErgonomics__Sequence ros_message");
      Py_DECREF(seq_field);
      Py_DECREF(field);
      return false;
    }
    manus_ros2_msgs__msg__ManusErgonomics * dest = ros_message->ergonomics.data;
    for (Py_ssize_t i = 0; i < size; ++i) {
      if (!manus_ros2_msgs__msg__manus_ergonomics__convert_from_py(PySequence_Fast_GET_ITEM(seq_field, i), &dest[i])) {
        Py_DECREF(seq_field);
        Py_DECREF(field);
        return false;
      }
    }
    Py_DECREF(seq_field);
    Py_DECREF(field);
  }
  {  // raw_sensor_orientation
    PyObject * field = PyObject_GetAttrString(_pymsg, "raw_sensor_orientation");
    if (!field) {
      return false;
    }
    if (!geometry_msgs__msg__quaternion__convert_from_py(field, &ros_message->raw_sensor_orientation)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // raw_sensor_count
    PyObject * field = PyObject_GetAttrString(_pymsg, "raw_sensor_count");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->raw_sensor_count = (int32_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // raw_sensor
    PyObject * field = PyObject_GetAttrString(_pymsg, "raw_sensor");
    if (!field) {
      return false;
    }
    PyObject * seq_field = PySequence_Fast(field, "expected a sequence in 'raw_sensor'");
    if (!seq_field) {
      Py_DECREF(field);
      return false;
    }
    Py_ssize_t size = PySequence_Size(field);
    if (-1 == size) {
      Py_DECREF(seq_field);
      Py_DECREF(field);
      return false;
    }
    if (!geometry_msgs__msg__Pose__Sequence__init(&(ros_message->raw_sensor), size)) {
      PyErr_SetString(PyExc_RuntimeError, "unable to create geometry_msgs__msg__Pose__Sequence ros_message");
      Py_DECREF(seq_field);
      Py_DECREF(field);
      return false;
    }
    geometry_msgs__msg__Pose * dest = ros_message->raw_sensor.data;
    for (Py_ssize_t i = 0; i < size; ++i) {
      if (!geometry_msgs__msg__pose__convert_from_py(PySequence_Fast_GET_ITEM(seq_field, i), &dest[i])) {
        Py_DECREF(seq_field);
        Py_DECREF(field);
        return false;
      }
    }
    Py_DECREF(seq_field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * manus_ros2_msgs__msg__manus_glove__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of ManusGlove */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("manus_ros2_msgs.msg._manus_glove");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "ManusGlove");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  manus_ros2_msgs__msg__ManusGlove * ros_message = (manus_ros2_msgs__msg__ManusGlove *)raw_ros_message;
  {  // glove_id
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->glove_id);
    {
      int rc = PyObject_SetAttrString(_pymessage, "glove_id", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // side
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->side.data,
      strlen(ros_message->side.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "side", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // raw_node_count
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->raw_node_count);
    {
      int rc = PyObject_SetAttrString(_pymessage, "raw_node_count", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // raw_nodes
    PyObject * field = NULL;
    size_t size = ros_message->raw_nodes.size;
    field = PyList_New(size);
    if (!field) {
      return NULL;
    }
    manus_ros2_msgs__msg__ManusRawNode * item;
    for (size_t i = 0; i < size; ++i) {
      item = &(ros_message->raw_nodes.data[i]);
      PyObject * pyitem = manus_ros2_msgs__msg__manus_raw_node__convert_to_py(item);
      if (!pyitem) {
        Py_DECREF(field);
        return NULL;
      }
      int rc = PyList_SetItem(field, i, pyitem);
      (void)rc;
      assert(rc == 0);
    }
    assert(PySequence_Check(field));
    {
      int rc = PyObject_SetAttrString(_pymessage, "raw_nodes", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // ergonomics_count
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->ergonomics_count);
    {
      int rc = PyObject_SetAttrString(_pymessage, "ergonomics_count", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // ergonomics
    PyObject * field = NULL;
    size_t size = ros_message->ergonomics.size;
    field = PyList_New(size);
    if (!field) {
      return NULL;
    }
    manus_ros2_msgs__msg__ManusErgonomics * item;
    for (size_t i = 0; i < size; ++i) {
      item = &(ros_message->ergonomics.data[i]);
      PyObject * pyitem = manus_ros2_msgs__msg__manus_ergonomics__convert_to_py(item);
      if (!pyitem) {
        Py_DECREF(field);
        return NULL;
      }
      int rc = PyList_SetItem(field, i, pyitem);
      (void)rc;
      assert(rc == 0);
    }
    assert(PySequence_Check(field));
    {
      int rc = PyObject_SetAttrString(_pymessage, "ergonomics", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // raw_sensor_orientation
    PyObject * field = NULL;
    field = geometry_msgs__msg__quaternion__convert_to_py(&ros_message->raw_sensor_orientation);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "raw_sensor_orientation", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // raw_sensor_count
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->raw_sensor_count);
    {
      int rc = PyObject_SetAttrString(_pymessage, "raw_sensor_count", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // raw_sensor
    PyObject * field = NULL;
    size_t size = ros_message->raw_sensor.size;
    field = PyList_New(size);
    if (!field) {
      return NULL;
    }
    geometry_msgs__msg__Pose * item;
    for (size_t i = 0; i < size; ++i) {
      item = &(ros_message->raw_sensor.data[i]);
      PyObject * pyitem = geometry_msgs__msg__pose__convert_to_py(item);
      if (!pyitem) {
        Py_DECREF(field);
        return NULL;
      }
      int rc = PyList_SetItem(field, i, pyitem);
      (void)rc;
      assert(rc == 0);
    }
    assert(PySequence_Check(field));
    {
      int rc = PyObject_SetAttrString(_pymessage, "raw_sensor", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
