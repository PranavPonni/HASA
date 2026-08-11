execute_process(COMMAND "/home/handlingteam2/HASA/ros_ws/build-vscode/allegro_hand_packages/allegro_hand_python/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/handlingteam2/HASA/ros_ws/build-vscode/allegro_hand_packages/allegro_hand_python/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
