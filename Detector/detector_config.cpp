/**
 * @file detector_config.cpp
 * @brief 查找并打开检测 YAML。改文件名或搜索前缀时必须改本文件注释。
 *
 * 作用/输入/输出见 detector_config.h。调用处：LightDetector::loadConfig 等打开同一份文件。
 */
#include "Detector/detector_config.h"

#include <fstream>

/**
 * 作用：在若干候选路径里找默认配置。
 * 输入：preferred 非空时先探测它。输出：能 ifstream 打开的路径，否则空。
 */
std::string findDetectorConfigPath(const std::string& preferred) {
    if (!preferred.empty()) {
        std::ifstream in(preferred);
        if (in.good())
            return preferred;
    }
    const std::string names[] = {"detector_params.yaml"};
    const std::string prefixes[] = {
        "config/",
        "../config/",
        "../../config/",
        "/home/zzy/Clion_project/TDT/detector2/config/",
    };
    for (const std::string& pre : prefixes) {
        for (const std::string& name : names) {
            const std::string cand = pre + name;
            std::ifstream in(cand);
            if (in.good())
                return cand;
        }
    }
    return {};
}

/**
 * 作用：FileStorage 只读打开 YAML。
 * 输入：path。输出：true 表示可以按节点读键。
 */
bool openDetectorYaml(const std::string& path, cv::FileStorage& fs) {
    if (path.empty())
        return false;
    fs.open(path, cv::FileStorage::READ);  // YAML → FileStorage
    return fs.isOpened();
}
