#ifndef DETECTOR_CONFIG_H
#define DETECTOR_CONFIG_H

#include <opencv2/opencv.hpp>
#include <string>

/**
 * 注释契约：改配置文件名、查找路径或 FileStorage 读法时必须同步改本文件注释。
 * 每个函数须写清作用、输入、输出。
 *
 * 检测门槛的磁盘入口：默认读 config/detector_params.yaml（新 HSV 双段红）。
 * 旧门槛在 config/detector_params_original.yaml，启动时把路径指过去即可恢复。
 * OpenCV FileStorage 读 YAML；缺键时调用方保持结构体默认值。
 */

/**
 * 作用：在常见工作目录里找 detector_params.yaml（相对 cmake-build-debug 时 ../../config）。
 * 输入：preferred 非空则优先用该路径（命令行 argv / ROS 参数）。
 * 输出：能打开的路径；都找不到返回空串。
 */
std::string findDetectorConfigPath(const std::string& preferred = {});

/**
 * 作用：打开一份检测 YAML（只读）。
 * 输入：path 为文件路径。输出：true 且 fs 已打开；失败时 fs 不保证可用。
 */
bool openDetectorYaml(const std::string& path, cv::FileStorage& fs);

/**
 * 作用：从 FileNode 读一个键，键不存在则不改 dst。
 * 输入：n 为父节点；key 为键名；dst 为要写入的变量。
 * 输出：无；键存在时 OpenCV >> 写入 dst。
 */
template <typename T>
void readYaml(const cv::FileNode& n, const char* key, T& dst) {
    if (n.empty() || n[key].empty())
        return;
    n[key] >> dst;
}

#endif  // DETECTOR_CONFIG_H
