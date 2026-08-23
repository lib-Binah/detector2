/**
 * @file light_detector.cpp
 * @brief 灰度阈值灯条检测
 *
 * 注释契约：禁止随意删除本文件注释；改阈值、掩码或筛选条件时必须改注释。
 * 检测用的高/宽/细长比/倾角/面积/HSV占比必须来自 Light 的同一属性或函数，与 overlay 显示
 * 计算方式相同；禁止在此另算 height/width 当细长比，也禁止另算 HSV 占比。
 * 每个函数须写清作用/输入/输出；调用处也要注明调谁、送什么、得到什么。
 * get_lights 每一步必须逐步注释。
 *
 * 原理概要：
 * 灯条是高亮、近竖直的细长色块。先把 BGR 转灰度，大于 gray_thres 的像素为前景，
 * 闭运算补小缺口，再找外轮廓。高/宽/细长比/倾角/面积一律用 Light 构造时写入的字段
 * 或由其算出的 area()：筛选和 overlay 同一份数据。
 * 细长比读 Light::ratio（height/width），倾角读 |Light::angle|（度）。
 * 颜色先看 RGB：在旋转矩形内、且落在二值前景上的像素，R>B 计红面积、B>R 计蓝面积，
 * 红面积更大判红，蓝面积更大判蓝。
 * 再按已判颜色做 HSV 二次确认：旋转矩形内 inRange(红=red_params_ / 蓝=blue_params_)
 * 的像素占矩形栅格面积的比例写入 Light::hsv_ratio，须大于 kHsvRatioMin 才保留。
 */
#include "light_detector.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

constexpr int kMinArea = 40;     // 旋转矩形面积下限，滤掉碎点
constexpr int kMaxArea = 3500;   // 面积上限，滤掉大块误分割
const double kMaxAbsAngle = 25.0;  // 右上→右下相对竖直的倾角门限（度）
constexpr double kHsvRatioMin = 0.4;  // 旋转矩形内对应色 HSV 像素占比下限，须大于此值才是灯条

}  // namespace

/**
 * 作用：空检测器，灰度用 LightParams 默认值，并写入红/蓝 HSV 默认区间。
 * 输入：无。输出：可直接 get_lights 的实例。
 */
LightDetector::LightDetector() {
    // OpenCV HSV：H∈[0,180]。红在 0 附近（此处只用低 H 段，未包 170–180 环绕）。
    red_params_.h_min = 0;
    red_params_.h_max = 35;
    red_params_.s_min = 60;
    red_params_.s_max = 255;
    red_params_.v_min = 120;
    red_params_.v_max = 255;

    // 蓝在 H≈80–105（偏青蓝），S 下限较低以兼容过曝发白的蓝灯。
    blue_params_.h_min = 80;
    blue_params_.h_max = 105;
    blue_params_.s_min = 20;
    blue_params_.s_max = 255;
    blue_params_.v_min = 200;
    blue_params_.v_max = 255;
}

/**
 * 作用：把 gray 进度条的值写进检测器。
 * 输入：thres 任意 int。输出：无；params_.gray_thres 被夹到 [0,255]。
 */
void LightDetector::setGrayThres(int thres) {
    // clamp：任意 int → [0,255]，写入 params_.gray_thres，供 get_bin 使用。
    params_.gray_thres = std::clamp(thres, 0, 255);
}

/**
 * 作用：灰度阈值 + 闭运算得到灯条高亮前景（找轮廓和调试窗必须用同一张）。
 * 输入：frame 为 BGR 或单通道灰度。
 * 输出：CV_8UC1，> gray_thres 为 255。
 */
cv::Mat LightDetector::get_bin(const cv::Mat& frame) const {
    cv::Mat gray;
    if (frame.channels() == 3)
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);  // BGR 帧 → 单通道灰度
    else
        gray = frame;

    // threshold：灰度图 + gray_thres → 二值图（>阈值 255，否则 0）。
    cv::Mat bin;
    cv::threshold(gray, bin, params_.gray_thres, 255, cv::THRESH_BINARY);

    // morphologyEx CLOSE：二值图 → 补上灯芯小黑洞；核太大会把邻近灯粘在一起。
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, kernel);
    return bin;  // CV_8UC1 前景 255
}

/**
 * 作用：在框 ∩ 灰度前景上用 R>B / B>R 像素数投票判红蓝（不用 HSV）。
 * 输入：整帧 r_gt_b/b_gt_r、get_bin 的 bin、Light 四顶点（左上/右上/左下/右下）。
 * 输出：1 红 / 2 蓝 / -1 丢掉。
 */
int LightDetector::classifyColorRgb(const cv::Mat& r_gt_b, const cv::Mat& b_gt_r,
                                    const cv::Mat& bin, const cv::Point2f pts[4]) const {
    // 判色范围 = 旋转矩形 ∩ 灰度二值前景。
    // 只比框内灯芯像素的 R/B，避免框里装甲板、数字、背景把颜色带偏。
    // pts 存的是 左上/右上/左下/右下（不是绕序）；fillConvexPoly 必须按凸包绕序
    // 喂点，否则对角线交叉、填出来的不是灯条矩形。
    const cv::Point ipts[4] = {pts[0], pts[1], pts[3], pts[2]};  // 左上→右上→右下→左下

    // zeros+fillConvexPoly：四顶点 → 整图大小的凸包掩码（框内 255）。
    cv::Mat poly = cv::Mat::zeros(bin.size(), CV_8UC1);
    cv::fillConvexPoly(poly, ipts, 4, 255);

    // bitwise_and：poly ∩ bin → 既在框内、又过灰度阈值的像素。
    cv::Mat inside;
    cv::bitwise_and(poly, bin, inside);

    // r_gt_b / b_gt_r 是整帧「R>B / B>R」；与 inside 相与后只剩下灯芯上的偏红/偏蓝点。
    cv::Mat red_mask;
    cv::Mat blue_mask;
    cv::bitwise_and(r_gt_b, inside, red_mask);
    cv::bitwise_and(b_gt_r, inside, blue_mask);

    // 过曝灯芯不用 HSV（H 乱、S 低），用偏红/偏蓝像素个数投票。
    const int red_area = cv::countNonZero(red_mask);   // 偏红像素数
    const int blue_area = cv::countNonZero(blue_mask);  // 偏蓝像素数
    if (red_area > blue_area)
        return 1;  // 红灯条
    if (blue_area > red_area)
        return 2;  // 蓝灯条
    return -1;     // 一样多，或框内没有前景，调用处不当灯条
}

/**
 * 作用：整帧 HSV 按 H/S/V 上下限做 inRange。
 * 输入：hsv 为 BGR2HSV 图；p 为红或蓝参数。
 * 输出：落在区间内的像素为 255。
 */
cv::Mat LightDetector::hsvMask(const cv::Mat& hsv, const HsvParams& p) const {
    cv::Mat mask;
    // inRange：HSV 图 + 上下限 → 该色二值掩码。
    cv::inRange(hsv,
                cv::Scalar(p.h_min, p.s_min, p.v_min),
                cv::Scalar(p.h_max, p.s_max, p.v_max),
                mask);
    return mask;
}

/**
 * 作用：旋转矩形内符合对应色 HSV 的像素占矩形栅格面积的比例。
 * 输入：整帧 color_mask、Light 四顶点（左上/右上/左下/右下）。
 * 输出：[0,1]，供写入 Light::hsv_ratio；框栅格面积为 0 时返回 0。
 */
double LightDetector::hsvRatioInRotatedRect(const cv::Mat& color_mask,
                                            const cv::Point2f pts[4]) const {
    // pts 不是绕序；fillConvexPoly 必须按凸包绕序，否则对角线交叉。
    const cv::Point ipts[4] = {pts[0], pts[1], pts[3], pts[2]};  // 左上→右上→右下→左下

    // zeros+fillConvexPoly：四顶点 → 整图大小的旋转矩形栅格掩码（框内 255）。
    cv::Mat poly = cv::Mat::zeros(color_mask.size(), CV_8UC1);
    cv::fillConvexPoly(poly, ipts, 4, 255);

    // 分母用栅格像素数，保证占比落在 [0,1]，与 overlay 读的 hsv_ratio 同一套算法。
    const int poly_area = cv::countNonZero(poly);
    if (poly_area <= 0)
        return 0.0;

    // bitwise_and：框 ∩ 该色 HSV → 命中像素。
    cv::Mat inside;
    cv::bitwise_and(color_mask, poly, inside);
    return static_cast<double>(cv::countNonZero(inside)) / static_cast<double>(poly_area);
}

/**
 * 作用：一帧检出合法灯条：灰度分割 → 几何筛选 → RGB 判色 → HSV 面积比。
 * 输入：frame 必须 BGR 三通道；bin_out 非空则写入与检测同一张二值图。
 * 输出：通过门限的 Light 列表；空帧或非三通道返回空。
 */
std::vector<Light> LightDetector::get_lights(const cv::Mat& frame, cv::Mat* bin_out) {
    std::vector<Light> valid_lights;
    // 后面 classifyColorRgb 要拆 B/R，灰度单通道不够。
    if (frame.empty() || frame.channels() != 3)
        return valid_lights;

    // get_bin：BGR 帧 → 二值前景（灰度阈值 + 闭运算）。调试窗和找轮廓必须用这一张。
    const cv::Mat bin = get_bin(frame);
    if (bin_out)
        *bin_out = bin;

    // split+compare：BGR 帧 → 整帧 r_gt_b(R>B)、b_gt_r(B>R)。每个轮廓只做与运算。
    std::vector<cv::Mat> ch;
    cv::split(frame, ch);  // 0=B, 1=G, 2=R
    cv::Mat r_gt_b;
    cv::Mat b_gt_r;
    cv::compare(ch[2], ch[0], r_gt_b, cv::CMP_GT);
    cv::compare(ch[0], ch[2], b_gt_r, cv::CMP_GT);

    // cvtColor+hsvMask：BGR → 整帧红/蓝 HSV 掩码，每个轮廓只做框内占比，避免重复 inRange。
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);  // BGR 帧 → HSV，H∈[0,180]
    const cv::Mat red_hsv = hsvMask(hsv, red_params_);    // 红 inRange
    const cv::Mat blue_hsv = hsvMask(hsv, blue_params_);  // 蓝 inRange

    // clone：复制 bin，避免 findContours 改脏调试用的原图。
    // findContours：二值图 → 外轮廓列表（CHAIN_APPROX_SIMPLE 矩形可能只剩 4 点）。
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat bin_find = bin.clone();
    cv::findContours(bin_find, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 细长比门限来自参数，比较对象必须是 Light::ratio（与 overlay 同源）。
    const double min_ratio = params_.min_height_width;
    const double max_ratio = params_.max_height_width;

    for (const auto& contour : contours) {
        // 至少 3 点才能构成面；不能要求 ≥5。
        if (contour.size() < 3)
            continue;

        // Light(contour)：轮廓 → 旋转矩形顶点 + height/width/ratio/angle/area。几何只在这里算一次。
        Light light(contour);
        if (light.height <= 1e-6)
            continue;

        // 同源契约：细长比/面积/倾角只读 Light 字段或 area()；HSV 占比只读 hsv_ratio。
        if (light.width <= 1e-6)
            continue;
        if (light.ratio < min_ratio || light.ratio > max_ratio)
            continue;  // 太胖（噪声）或太瘦（线噪声）

        // area()：读 rotatedRect 宽×高，与 overlay 若显示面积必须同一函数。
        if (light.area() < kMinArea || light.area() > kMaxArea)
            continue;  // 碎点 / 大块误分割

        if (std::fabs(light.angle) > kMaxAbsAngle)
            continue;  // 近水平，不像竖直灯条

        // classifyColorRgb：框 ∩ 灰度前景 + 整帧 R>B/B>R → 1 红 / 2 蓝 / -1 丢掉。
        light.color = classifyColorRgb(r_gt_b, b_gt_r, bin, light.pts);
        if (light.color != 1 && light.color != 2)
            continue;

        // hsvRatioInRotatedRect：旋转矩形 ∩ 该色 HSV → 占比，写入 Light::hsv_ratio（overlay 同源）。
        const cv::Mat& color_hsv = (light.color == 1) ? red_hsv : blue_hsv;
        light.hsv_ratio = hsvRatioInRotatedRect(color_hsv, light.pts);
        if (light.hsv_ratio <= kHsvRatioMin)
            continue;  // HSV 命中面积不够，RGB 投票过了也不当灯条

        valid_lights.push_back(light);
    }
    return valid_lights;  // 通过几何+RGB+HSV 占比门限的灯条
}
