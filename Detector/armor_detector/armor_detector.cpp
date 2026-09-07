/**
 * @file armor_detector.cpp
 * @brief 灯条配对装甲板（逻辑对齐 clone_detector Detector::matchLights）
 *
 * 注释契约：禁止随意删除本文件注释；改门槛、夹灯判定或流水线顺序时必须改注释。
 * 检测用的长度比/中心距/倾角必须来自 ArmorBoard 的同一属性，与 overlay 显示
 * 计算方式相同；禁止在此另算 height 比或中心距。
 * 每个函数须写清作用/输入/输出；调用处也要注明调谁、送什么、得到什么。
 * get_armors 每一步必须逐步注释。
 *
 * 原理概要：
 * 同色灯条全组合。先丢掉两灯包围盒里还夹着第三根灯的对（跨过了另一块板）。
 * 再读 ArmorBoard 构造时写入的 light_ratio / center_distance / angle / light_angle_diff：
 *   长度比 > min_light_ratio；
 *   归一化中心距落入小板或大板区间；
 *   中心连线倾角 < max_angle；
 *   |Light::angle 左 - Light::angle 右| 不大于 max_light_angle_diff（配置默认 10°）。
 * 过关后透视数字、CNN 分类。一根灯条最多属于一块板：按 confidence 从高到低贪心占灯，
 * 共用灯条时留下模型置信度更高的那块。最后 IPPE PnP。
 */
#include "armor_detector.h"
#include "Detector/detector_config.h"

#include <algorithm>
#include <cmath>

/**
 * 作用：加载数字 CNN。
 * 输入：ONNX 路径。输出：true 表示 net 非空。
 */
bool ArmorDetector::loadClassifier(const std::string& model_path) {
    return classifier_.loadModel(model_path);  // 路径 → NumberClassifier 内部 Net
}

/**
 * 作用：YAML armor 节点 → 配对门槛，并加载 number 透视尺寸 / use_binary。
 * 输入：path。输出：true 表示文件打开成功。
 */
bool ArmorDetector::loadConfig(const std::string& path) {
    cv::FileStorage fs;
    if (!openDetectorYaml(path, fs))
        return false;
    const cv::FileNode n = fs["armor"];
    readYaml(n, "min_light_ratio", params_.min_light_ratio);
    readYaml(n, "min_small_center_distance", params_.min_small_center_distance);
    readYaml(n, "max_small_center_distance", params_.max_small_center_distance);
    readYaml(n, "min_large_center_distance", params_.min_large_center_distance);
    readYaml(n, "max_large_center_distance", params_.max_large_center_distance);
    readYaml(n, "max_angle", params_.max_angle);
    readYaml(n, "max_light_angle_diff", params_.max_light_angle_diff);
    readYaml(n, "detect_color", params_.detect_color);
    readYaml(n, "classifier_threshold", params_.classifier_threshold);
    float digit6 = NumberClassifier::kMinConfidenceDigit6;
    readYaml(n, "classifier_threshold_digit6", digit6);
    classifier_.setDigit6Threshold(digit6);  // 数字 6 单独门槛，conf 进度条不改这项
    classifier_.loadConfig(path);  // number 透视条带尺寸 + use_binary（Otsu 是否写入 number_img）
    return true;
}

/**
 * 作用：两灯四端点包围盒里是否夹着第三根灯。
 * 输入：一对灯、本帧全部灯。输出：true=夹了。
 */
bool ArmorDetector::containLight(const Light& a, const Light& b,
                                 const std::vector<Light>& lights) const {
    // 包围盒只用四端点，不用整根旋转矩形，避免把斜向邻近灯误判成“夹在中间”。
    const std::vector<cv::Point2f> pts = {
        ArmorBoard::topOf(a), ArmorBoard::bottomOf(a),
        ArmorBoard::topOf(b), ArmorBoard::bottomOf(b)};
    const cv::Rect box = cv::boundingRect(pts);  // 四端点 → 轴对齐盒

    for (const auto& test : lights) {
        if (test.center() == a.center() || test.center() == b.center())
            continue;
        const cv::Point2f top = ArmorBoard::topOf(test);
        const cv::Point2f bottom = ArmorBoard::bottomOf(test);
        const cv::Point2f c = test.center();
        // Rect::contains 只收整数点；端点是 float，先四舍五入再测是否在盒内。
        auto inside = [&](const cv::Point2f& p) {
            return box.contains(cv::Point(cvRound(p.x), cvRound(p.y)));
        };
        if (inside(top) || inside(bottom) || inside(c))
            return true;
    }
    return false;
}

/**
 * 作用：读 ArmorBoard 已写入的四个量判是否像装甲，并写入 type。
 * 输入：构造好的 armor。输出：true 表示通过门槛。
 */
bool ArmorDetector::isArmor(ArmorBoard& armor) const {
    const bool ratio_ok = armor.light_ratio > params_.min_light_ratio;

    const double d = armor.center_distance;
    const bool small_ok =
        (params_.min_small_center_distance < d && d < params_.max_small_center_distance);
    const bool large_ok =
        (params_.min_large_center_distance < d && d < params_.max_large_center_distance);
    const bool distance_ok = small_ok || large_ok;

    const bool angle_ok = armor.angle < params_.max_angle;
    // light_angle_diff 已在构造时用 Light::angle 算好；大于 max_light_angle_diff 说明两灯条倾角差太大，不像一块板。
    const bool light_angle_ok = armor.light_angle_diff <= params_.max_light_angle_diff;

    // 分界用大板区间下限：比它大的当大板。overlay 的 type 必须读这个字段。
    armor.type = (d > params_.min_large_center_distance) ? ArmorType::LARGE : ArmorType::SMALL;

    return ratio_ok && distance_ok && angle_ok && light_angle_ok;
}

/**
 * 作用：同色两两配对，中间不夹灯且几何像装甲。
 * 输入：本帧灯条。输出：未做数字/PnP 的板。
 */
std::vector<ArmorBoard> ArmorDetector::matchLights(const std::vector<Light>& lights) const {
    std::vector<ArmorBoard> armors;
    const int n = static_cast<int>(lights.size());
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const Light& a = lights[static_cast<size_t>(i)];
            const Light& b = lights[static_cast<size_t>(j)];

            // 颜色：detect_color==0 时只要同色；否则两根都得是指定色。
            if (a.color != b.color)
                continue;
            if (a.color != 1 && a.color != 2)
                continue;
            if (params_.detect_color == 1 || params_.detect_color == 2) {
                if (a.color != params_.detect_color)
                    continue;
            }

            if (containLight(a, b, lights))
                continue;  // 四端点盒里夹了第三根灯 → 跨板错配

            ArmorBoard armor(a, b);  // 两灯 → 左右排序 + 四个几何量
            if (isArmor(armor))
                armors.push_back(std::move(armor));
        }
    }
    return armors;
}

/**
 * 作用：按 confidence 从高到低贪心占用灯条，一根灯只留给一块板。
 * 输入：已分类的 armors。输出：就地删掉与更高置信度板共用左或右灯的板。
 */
void ArmorDetector::greedyResolveSharedLights(std::vector<ArmorBoard>& armors) const {
    if (armors.size() < 2)
        return;

    // 稳定排序：置信度高的先占灯；相同则 light_ratio 更大的更像一块板；再相同保持配对顺序。
    std::stable_sort(armors.begin(), armors.end(),
                     [](const ArmorBoard& a, const ArmorBoard& b) {
                         if (a.confidence != b.confidence)
                             return a.confidence > b.confidence;
                         return a.light_ratio > b.light_ratio;
                     });

    auto sameCenter = [](const cv::Point2f& p, const cv::Point2f& q) { return p == q; };
    std::vector<cv::Point2f> used_centers;
    auto taken = [&](const cv::Point2f& c) {
        for (const auto& u : used_centers) {
            if (sameCenter(u, c))
                return true;
        }
        return false;
    };

    std::vector<ArmorBoard> kept;
    kept.reserve(armors.size());
    for (auto& armor : armors) {
        const cv::Point2f lc = armor.left.center();   // 与 containLight 同一套中心相等判定
        const cv::Point2f rc = armor.right.center();
        if (taken(lc) || taken(rc))
            continue;
        used_centers.push_back(lc);
        used_centers.push_back(rc);
        kept.push_back(std::move(armor));
    }
    armors.swap(kept);
}

/**
 * 作用：灯条 → 配对 → 数字 → 贪心消共用灯 → PnP。
 * 输入：get_lights 的灯条 + 同一帧 BGR。输出：通过筛选的 ArmorBoard。
 */
std::vector<ArmorBoard> ArmorDetector::get_armors(const std::vector<Light>& lights,
                                                  const cv::Mat& frame) {
    // 1) matchLights：同色全组合 → 不夹第三灯 → 长度比/中心距/倾角/灯条倾角差。
    std::vector<ArmorBoard> armors = matchLights(lights);

    if (armors.empty())
        return armors;

    // 2) extractNumbers：四端点透视 → 大板 20×28 / 小板宽 +30% → 可选 Otsu，写入 number_img。
    classifier_.extractNumbers(frame, armors);

    // 3) classify：CNN softmax；other / 置信度未大于门槛（数字 6 用固定 60%）/ 大板非 1 或小板为 1 的板删掉。没模型则跳过。
    if (classifier_.loaded())
        classifier_.classify(armors, params_.classifier_threshold);

    // 4) 贪心消共用灯：按 ArmorBoard::confidence 高→低占灯，一根灯条只留一块板。
    greedyResolveSharedLights(armors);

    // 5) solvePnP：每块板 IPPE；失败只把 pnp_ok=false，几何板仍留下给 overlay。
    for (auto& armor : armors)
        armor.solvePnP();

    return armors;
}
