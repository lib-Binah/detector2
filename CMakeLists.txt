# 注释契约：禁止删除本文件注释；增删源文件或改目标名时必须同步改注释。
# 检测筛选与 overlay 显示必须读 Light 的同一属性或函数（计算方式相同）。
#
# detector2：灰度阈值灯条检测工程（OpenCV + C++20）。
# 可执行文件 light_test = 几何 Light + LightDetector + 双窗口播放调试。

cmake_minimum_required(VERSION 3.10)
project(detector2)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "OpenCV Include: ${OpenCV_INCLUDE_DIRS}")

# 三个源文件分工：Light 几何、检测器、播放 UI。改路径时三处 include 也要改。
add_executable(light_test
        Detector/light_detector_gray/light_test.cpp
        Detector/light_detector_gray/Light.cpp
        Detector/light_detector_gray/light_detector.cpp
)

# 头文件就地 #include "Light.h"；同时允许从工程根引用。
target_include_directories(light_test PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/Detector/light_detector_gray
)

target_link_libraries(light_test PRIVATE ${OpenCV_LIBS})
