# 注释契约：增删源文件或改目标名时必须同步改本文件注释。
# video_publisher 功能包：读本地 avi/mp4，按文件帧率发 /image_raw。
# 本目录是 colcon 功能包；工作空间根是上两级 video_publisher/，不要从 detector2 根 CMakeLists add_subdirectory。

cmake_minimum_required(VERSION 3.10)
project(video_publisher)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# 作用：编出发布节点。输入：video_publisher.cpp。输出：可执行文件 video_publisher。
add_executable(video_publisher src/video_publisher.cpp)
ament_target_dependencies(video_publisher
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
)
target_link_libraries(video_publisher ${OpenCV_LIBS})

install(TARGETS video_publisher DESTINATION lib/${PROJECT_NAME})

ament_package()
