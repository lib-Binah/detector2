/**
 * 注释契约：禁止随意删除本文件注释；改 warp 尺寸、归一化、类别映射或大小板矛盾规则时必须改注释。
 * 每个函数须写清作用/输入/输出；OpenCV DNN / 透视调用处也要注明。
 *
 * 数字：灯条端点透视 → 20×28 → 可选 Otsu 二值 → (x/255-0.5)/0.5 → ONNX Softmax → 1~8 / other。
 * number_img 与 CNN 输入必须是同一张；改二值开关时 overlay 左上角跟着变。
 */
#include "number_classifier.h"
#include "Detector/detector_config.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <vector>

/**
 * 作用：读 ONNX CNN（导出时包了 Softmax，输出已是概率）。
 * 输入：model_path。输出：true 表示 net 非空。
 */
bool NumberClassifier::loadModel(const std::string& model_path) {
    loaded_ = false;
    try {
        net_ = cv::dnn::readNetFromONNX(model_path);  // ONNX 文件 → cv::dnn::Net
    } catch (const cv::Exception&) {
        return false;
    }
    if (net_.empty())
        return false;
    loaded_ = true;
    return true;
}

/**
 * 作用：写入透视条带尺寸。
 * 输入：四项像素尺寸。输出：无。
 */
void NumberClassifier::setWarpParams(int light_length, int warp_height, int small_w, int large_w) {
    light_length_ = std::max(1, light_length);
    warp_height_ = std::max(kInputH, warp_height);
    small_armor_width_ = std::max(kInputW, small_w);
    large_armor_width_ = std::max(kInputW, large_w);
}

/**
 * 作用：YAML number 节点 → 透视尺寸与是否 Otsu 二值。
 * 输入：path。输出：true 表示打开成功。
 */
bool NumberClassifier::loadConfig(const std::string& path) {
    cv::FileStorage fs;
    if (!openDetectorYaml(path, fs))
        return false;
    int light_length = light_length_;
    int warp_height = warp_height_;
    int small_w = small_armor_width_;
    int large_w = large_armor_width_;
    const cv::FileNode n = fs["number"];
    readYaml(n, "light_length", light_length);
    readYaml(n, "warp_height", warp_height);
    readYaml(n, "small_armor_width", small_w);
    readYaml(n, "large_armor_width", large_w);
    int use_binary = use_binary_ ? 1 : 0;
    readYaml(n, "use_binary", use_binary);
    use_binary_ = use_binary != 0;
    setWarpParams(light_length, warp_height, small_w, large_w);
    return true;
}

/**
 * 作用：number_img → blob。二值 0/255 与灰度都走 (x/255-0.5)/0.5。
 * 输入：单通道图（通常已是 20×28）。输出：[1,1,28,20] float。
 */
cv::Mat NumberClassifier::preprocess(const cv::Mat& gray) const {
    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(kInputW, kInputH), 0, 0, cv::INTER_NEAREST);  // 二值用最近邻，避免插出灰边

    cv::Mat float_img;
    resized.convertTo(float_img, CV_32F, 1.0 / 255.0);  // [0,255] → [0,1]
    // Normalize((0.5,),(0.5,))：(x-0.5)/0.5；二值会落到约 -1 和 +1。
    const cv::Mat normalized = (float_img - 0.5f) / 0.5f;
    return cv::dnn::blobFromImage(normalized);  // HWC → NCHW [1,1,28,20]
}

/**
 * 作用：每块板透视出 20×28 数字图，可选 Otsu 二值，写入 number_img（CNN 与左上角缩略图同源）。
 * 输入：BGR 帧；armors 就地改。输出：无。use_binary_ 为 true 时 number_img 为 0/255。
 */
void NumberClassifier::extractNumbers(const cv::Mat& bgr, std::vector<ArmorBoard>& armors) const {
    if (bgr.empty())
        return;

    for (auto& armor : armors) {
        armor.number_img.release();
        const int warp_width =
            (armor.type == ArmorType::SMALL) ? small_armor_width_ : large_armor_width_;
        // 灯条端点映到条带左右边：下、上、上、下，与 clone_detector extractNumbers 同序。
        const cv::Point2f src[4] = {
            armor.left_bottom, armor.left_top, armor.right_top, armor.right_bottom};
        const int top_y = (warp_height_ - light_length_) / 2 - 1;
        const int bottom_y = top_y + light_length_;
        const cv::Point2f dst[4] = {
            {0.f, static_cast<float>(bottom_y)},
            {0.f, static_cast<float>(top_y)},
            {static_cast<float>(warp_width - 1), static_cast<float>(top_y)},
            {static_cast<float>(warp_width - 1), static_cast<float>(bottom_y)},
        };

        const cv::Mat M = cv::getPerspectiveTransform(src, dst);  // 四端点 → 条带
        cv::Mat warped;
        cv::warpPerspective(bgr, warped, M, cv::Size(warp_width, warp_height_));

        const int x0 = (warp_width - kInputW) / 2;
        if (x0 < 0 || warped.cols < x0 + kInputW || warped.rows < kInputH)
            continue;
        const cv::Mat roi = warped(cv::Rect(x0, 0, kInputW, kInputH));  // 中间 20×28

        cv::Mat gray;
        if (roi.channels() == 3)
            cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);  // BGR ROI → 灰度，再按 use_binary_ 决定是否 Otsu
        else
            gray = roi;
        if (use_binary_) {
            // Otsu：按本块 ROI 直方图自选门槛写成 0/255；THRESH_BINARY 亮区=255。CNN 与 overlay 都读这张。
            cv::threshold(gray, armor.number_img, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        } else {
            gray.copyTo(armor.number_img);  // 灰度通路：与训练照片分布一致
        }
    }
}

/**
 * 作用：CNN 分类；other / 置信度未严格大于门槛 / 大板数字不是 1 或小板数字是 1 则删除该板。
 * 输入：已 extract 的 armors；threshold 为概率门槛（须 confidence > threshold，默认 0.80）。
 * 输出：就地留下通过的板。
 */
void NumberClassifier::classify(std::vector<ArmorBoard>& armors, float threshold) {
    if (!loaded_)
        return;  // 没模型时不筛，几何配对和 PnP 仍可调

    std::vector<ArmorBoard> kept;
    kept.reserve(armors.size());

    for (auto& armor : armors) {
        if (armor.number_img.empty())
            continue;

        const cv::Mat blob = preprocess(armor.number_img);  // number_img（二值或灰度）→ NCHW blob
        net_.setInput(blob);
        cv::Mat output = net_.forward();  // 9 维 softmax
        output = output.reshape(1, 1);

        double min_val = 0.0;
        double max_val = 0.0;
        cv::Point min_loc;
        cv::Point max_loc;
        cv::minMaxLoc(output, &min_val, &max_val, &min_loc, &max_loc);
        const int cls = max_loc.x;
        armor.confidence = static_cast<float>(max_val);

        // 必须严格大于门槛才当装甲板；等于 80% 也丢掉。overlay 读同一份 confidence。
        if (cls == kOtherClass || armor.confidence <= threshold) {
            armor.number = 0;
            armor.classification_result = "other";
            continue;
        }

        armor.number = cls + 1;  // 索引 0~7 → 标签 1~8
        std::ostringstream ss;
        ss << armor.number << ":_" << std::fixed;
        ss.precision(1);
        ss << (armor.confidence * 100.f) << "%";
        armor.classification_result = ss.str();

        // 大小板与兵种数字：大板只认英雄 1（2~8 都丢掉）；小板不该是英雄 1。
        const bool large = (armor.type == ArmorType::LARGE);
        const bool mismatch = large ? (armor.number != 1) : (armor.number == 1);
        if (mismatch)
            continue;

        kept.push_back(std::move(armor));
    }
    armors.swap(kept);
}
