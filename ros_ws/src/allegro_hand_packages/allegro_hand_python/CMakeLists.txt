cmake_minimum_required(VERSION 3.0.2)

project(allegro_hand_python)
find_package(catkin REQUIRED COMPONENTS rostest)

catkin_python_setup()

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES handover_viz
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS python-scipy python-matplotlib
)

## Testing
## Add folders to be run by python nosetests
if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(scripts/allegro_hand)
  add_rostest(launch_test/allegro_launch.test) # catkin_make run_tests
endif()
