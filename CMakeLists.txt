cmake_minimum_required(VERSION 3.8)
project(fast_calib)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2_storage REQUIRED)
find_package(livox_ros_driver REQUIRED)

# Find PCL
find_package(PCL REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(dependencies
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  cv_bridge
  pcl_conversions
  pcl_ros
  rosbag2_cpp
  rosbag2_storage
  livox_ros_driver
)

# fast_calib executable
add_executable(fast_calib src/main.cpp)
ament_target_dependencies(fast_calib ${dependencies})
target_link_libraries(fast_calib ${PCL_LIBRARIES} ${OpenCV_LIBS})

# multi_fast_calib executable
add_executable(multi_fast_calib src/multi_scene.cpp)
ament_target_dependencies(multi_fast_calib ${dependencies})
target_link_libraries(multi_fast_calib ${PCL_LIBRARIES} ${OpenCV_LIBS})

install(TARGETS
  fast_calib
  multi_fast_calib
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  rviz_cfg
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()