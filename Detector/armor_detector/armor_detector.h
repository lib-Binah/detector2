#ifndef ARMOR_DETECTOR_H
#define ARMOR_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "ArmorBoard.h"
#include "Light.h"
#include "number_classifier.h"

/**
 * 注释契约（给后续修改者 / 智能体）：
 * 本模块注释说明装甲板配对各段的作用与原理。禁止为“精简”删除注释；
 * 改颜色过滤、夹灯判定、长度比/中心距/倾角门槛时，必须同步修改对应注释。
 * 几何筛选必须读 ArmorBoard 的属性（light_ratio、center_distance、angle、light_angle_diff），与 overlay
 * 显示同源、计算方式相同；禁止在本文件里另写一套中心距或长度比。
 * 每个函数须写清作用、输入、输出；调用处也要注明调谁、送什么、得到什么。
 * get_armors 每一步必须逐步注释（同色配对 / containLight / isArmor / 数字 / PnP）。
 *
 * 装甲板检测：已检出的灯条 → 同色两两配对 → 包围盒不夹第三灯 → 几何像一块板
 * → CNN 认数字 → IPPE PnP。灯条本身仍由 LightDetector 提供，本类不改灯条源码。
 */
struct ArmorParams {
    // 左右灯条 Light::height 之比（短/长）；过小说明两边不是同一块装甲。
    double min_light_ratio = 0.7;
    // 两灯中心距 / 平均 Light::height：小板、大板各一段区间。
    double min_small_center_distance = 0.8;
    double max_small_center_distance = 3.2;
    double min_large_center_distance = 3.2;
    double max_large_center_distance = 5.5;
    // 两灯中心连线相对水平的最大倾角（度）。
    double max_angle = 35.0;
    // 左右灯条 Light::angle 差的绝对值上限（度）；大于该值不当装甲板。
    double max_light_angle_diff = 10.0;
    // 0=红蓝都配（同色对），1=只红，2=只蓝。clone_detector 默认锁一色；调试默认都要。
    int detect_color = 0;
    // CNN softmax 门槛：confidence 必须严格大于该值才保留（默认 kMinConfidence=0.80）。播放进度条会改它。
    float classifier_threshold = NumberClassifier::kMinConfidence;
};

class ArmorDetector {
public:
    /**
     * 作用：空检测器，配对参数用 ArmorParams 默认值，CNN 尚未加载。
     * 输入：无。输出：可 matchLights；classify 在 loadClassifier 成功前不会删板。
     */
    ArmorDetector() = default;

    /**
     * 作用：只读查看配对参数。
     * 输入：无。输出：ArmorParams 常引用。
     */
    const ArmorParams& params() const { return params_; }

    /**
     * 作用：写入配对参数（播放器进度条改门槛时用）。
     * 输入：p 为完整参数。输出：无。
     */
    void setParams(const ArmorParams& p) { params_ = p; }

    /**
     * 作用：加载数字 CNN。
     * 输入：model_path 为 ONNX。输出：true 表示加载成功。
     */
    bool loadClassifier(const std::string& model_path);

    bool classifierLoaded() const { return classifier_.loaded(); }

    /**
     * 作用：一帧里从灯条配出装甲板：配对 → 数字 → PnP。
     * 输入：lights 为 LightDetector::get_lights 的结果；frame 为同一帧 BGR。
     * 输出：通过几何（以及若已加载模型则通过数字）的 ArmorBoard；PnP 失败仍保留几何板。
     */
    std::vector<ArmorBoard> get_armors(const std::vector<Light>& lights, const cv::Mat& frame);

private:
    /**
     * 作用：同色灯条两两配对，中间不夹灯且几何像装甲。
     * 输入：本帧全部合法灯条。输出：未做数字/PnP 的 ArmorBoard 列表。
     */
    std::vector<ArmorBoard> matchLights(const std::vector<Light>& lights) const;

    /**
     * 作用：两灯四端点包围盒里是否夹着第三根灯（跨装甲错配）。
     * 输入：一对灯、本帧全部灯。输出：true 表示夹了，这对丢掉。
     */
    bool containLight(const Light& a, const Light& b, const std::vector<Light>& lights) const;

    /**
     * 作用：用已写入的 light_ratio/center_distance/angle/light_angle_diff 判是否像装甲，并写入 type。
     * 输入：armor 构造时已算好四个量。输出：true 表示通过门槛。
     */
    bool isArmor(ArmorBoard& armor) const;

    ArmorParams params_;
    NumberClassifier classifier_;
};

#endif  // ARMOR_DETECTOR_H
