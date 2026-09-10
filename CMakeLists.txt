# 注释契约：禁止删除本文件注释；增删源文件或改目标名时必须同步改注释。
# 检测筛选与 overlay 显示必须读 Light 的同一属性或函数（计算方式相同）。
# 每个函数须写清作用/输入/输出；调用处也要注明调谁、送什么、得到什么。
#
# detector2：灰度阈值灯条检测 + 装甲板配对 + 弹道预瞄（OpenCV + C++20）。
# 可执行文件 light_test = 几何 Light + LightDetector + 配置加载 + 双窗口播放调试。
# 可执行文件 armor_test = 复用灯条检测（不改其源码）+ ArmorBoard + ArmorDetector + 数字 CNN + PnP。
# 可执行文件 aim_test = 检测 + 同济整车 EKF + 弹道预瞄（只含可视化开关，调检测请用 armor_test）。
# detector_config.cpp：查找/打开 config/detector_params.yaml。改路径时四处目标都要加这一份。

cmake_minimum_required(VERSION 3.10)
project(detector2)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "OpenCV Include: ${OpenCV_INCLUDE_DIRS}")

# 三个源文件分工：Light 几何、检测器、播放 UI。改路径时三处 include 也要改。
add_executable(light_test
        Detector/light_detector_gray/light_test.cpp
        Detector/light_detector_gray/Light.cpp
        Detector/light_detector_gray/light_detector.cpp
        Detector/detector_config.cpp
)

# 头文件就地 #include "Light.h"；同时允许从工程根引用。
target_include_directories(light_test PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/Detector/light_detector_gray
)

target_link_libraries(light_test PRIVATE ${OpenCV_LIBS})

# 装甲板调试：灯条几何/检测源文件只编译进本目标，不修改 light_detector_gray 逻辑。
add_executable(armor_test
        Detector/armor_detector/armor_test.cpp
        Detector/armor_detector/ArmorBoard.cpp
        Detector/armor_detector/armor_detector.cpp
        Detector/armor_detector/number_classifier.cpp
        Detector/light_detector_gray/Light.cpp
        Detector/light_detector_gray/light_detector.cpp
        Detector/detector_config.cpp
)

target_include_directories(armor_test PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/Detector/light_detector_gray
        ${CMAKE_CURRENT_SOURCE_DIR}/Detector/armor_detector
)

target_link_libraries(armor_test PRIVATE ${OpenCV_LIBS})

# 预瞄调试：同济 11 维整车 EKF + 弹道。检测源文件只编译进本目标，不改 armor_test。
add_executable(aim_test
        Aim/aim_test.cpp
        Aim/aimer.cpp
        Aim/ballistic.cpp
        Aim/ekf.cpp
        Aim/vehicle_ekf.cpp
        Detector/armor_detector/ArmorBoard.cpp
        Detector/armor_detector/armor_detector.cpp
        Detector/armor_detector/number_classifier.cpp
        Detector/light_detector_gray/Light.cpp
        Detector/light_detector_gray/light_detector.cpp
        Detector/detector_config.cpp
)

target_include_directories(aim_test PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/Detector/light_detector_gray
        ${CMAKE_CURRENT_SOURCE_DIR}/Detector/armor_detector
        ${CMAKE_CURRENT_SOURCE_DIR}/Aim
)

target_link_libraries(aim_test PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)
