# 注释契约：增删源文件或改目标名时必须同步改本文件注释。
# armor_subscriber 功能包：订阅 /image_raw，回调里编进 detector2 的灯条/装甲板源码做检测。
# 检测 .cpp 只链接、不修改；工作空间根是上两级 armor_subscriber/，不要从 detector2 根 CMakeLists add_subdirectory。

cmake_minimum_required(VERSION 3.10)
project(armor_subscriber)

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

# 灯条/装甲板源码在 detector2 工程里；改仓库位置时只改这一处。
set(DETECTOR2_ROOT "/home/zzy/Clion_project/TDT/detector2" CACHE PATH "detector2 源码树")

# 作用：编出订阅+检测节点。输入：本包 cpp + detector2 检测源文件。输出：可执行文件 armor_subscriber。
add_executable(armor_subscriber
  src/armor_subscriber.cpp
  ${DETECTOR2_ROOT}/Detector/armor_detector/ArmorBoard.cpp
  ${DETECTOR2_ROOT}/Detector/armor_detector/armor_detector.cpp
  ${DETECTOR2_ROOT}/Detector/armor_detector/number_classifier.cpp
  ${DETECTOR2_ROOT}/Detector/light_detector_gray/Light.cpp
  ${DETECTOR2_ROOT}/Detector/light_detector_gray/light_detector.cpp
)
ament_target_dependencies(armor_subscriber
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
)
target_include_directories(armor_subscriber PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${DETECTOR2_ROOT}/Detector/light_detector_gray
  ${DETECTOR2_ROOT}/Detector/armor_detector
)
target_link_libraries(armor_subscriber ${OpenCV_LIBS})

install(TARGETS armor_subscriber DESTINATION lib/${PROJECT_NAME})

ament_package()
