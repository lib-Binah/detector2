/**
 * @file light_detector.cpp
 * @brief 灰度阈值灯条检测
 *
 * 注释契约：禁止随意删除本文件注释；改阈值、掩码或筛选条件时必须改注释。
 *
 * 原理概要：
 * 灯条是高亮、近竖直的细长色块。先把 BGR 转灰度，大于 gray_thres 的像素为前景，
 * 闭运算补小缺口，再找外轮廓。用旋转矩形四顶点计算高（左上→左下）、
 * 宽（右上→右下）。细长比用高/顶边（左上→右上），因为右上→右下与高是对边。
 * 颜色不看 HSV：在旋转矩形内、且落在二值前景上的像素，R>B 计红面积、B>R 计蓝面积，
 * 红面积更大判红，蓝面积更大判蓝。
 */
#include "light_detector.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

constexpr int kMinArea = 40;     // 旋转矩形面积下限，滤掉碎点
constexpr int kMaxArea = 3500;   // 面积上限，滤掉大块误分割
const double kMaxAbsAngle = 25.0;  // 高边相对竖直的倾角门限（度）

}  // namespace

LightDetector::LightDetector() = default;

void LightDetector::setGrayThres(int thres) {
    params_.gray_thres = std::clamp(thres, 0, 255);
}

cv::Mat LightDetector::get_bin(const cv::Mat& frame) const {
    cv::Mat gray;
    if (frame.channels() == 3)
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    else
        gray = frame;

    // THRESH_BINARY：灰度 > gray_thres → 255，否则 0。
    cv::Mat bin;
    cv::threshold(gray, bin, params_.gray_thres, 255, cv::THRESH_BINARY);

    // 闭运算把灯芯小黑洞补上，核太大会把邻近灯粘在一起。
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, kernel);
    return bin;
}

int LightDetector::classifyColorRgb(const cv::Mat& r_gt_b, const cv::Mat& b_gt_r,
                                    const cv::Mat& bin, const cv::Point2f pts[4]) const {
    // 只在「框内且过灰度阈值」的像素上比 R/B，避免框里背景把颜色带偏。
    cv::Point ipts[4];
    for (int i = 0; i < 4; ++i)
        ipts[i] = pts[i];

    cv::Mat poly = cv::Mat::zeros(bin.size(), CV_8UC1);
    cv::fillConvexPoly(poly, ipts, 4, 255);

    cv::Mat inside;
    cv::bitwise_and(poly, bin, inside);

    cv::Mat red_mask;
    cv::Mat blue_mask;
    cv::bitwise_and(r_gt_b, inside, red_mask);
    cv::bitwise_and(b_gt_r, inside, blue_mask);

    const int red_area = cv::countNonZero(red_mask);
    const int blue_area = cv::countNonZero(blue_mask);
    if (red_area > blue_area)
        return 1;  // 红
    if (blue_area > red_area)
        return 2;  // 蓝
    return -1;     // 一样多或框内无前景，不当灯条
}

std::vector<Light> LightDetector::get_lights(const cv::Mat& frame, cv::Mat* bin_out) {
    std::vector<Light> valid_lights;
    if (frame.empty() || frame.channels() != 3)
        return valid_lights;  // RGB 判色需要三通道

    const cv::Mat bin = get_bin(frame);
    if (bin_out)
        *bin_out = bin;

    // 整帧算一次 R>B / B>R，后面每个轮廓只做与运算，避免重复 split。
    std::vector<cv::Mat> ch;
    cv::split(frame, ch);  // 0=B, 1=G, 2=R
    cv::Mat r_gt_b;
    cv::Mat b_gt_r;
    cv::compare(ch[2], ch[0], r_gt_b, cv::CMP_GT);
    cv::compare(ch[0], ch[2], b_gt_r, cv::CMP_GT);

    // clone 再 findContours：部分 OpenCV 会改传入图，不能弄脏调试用 bin。
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat bin_find = bin.clone();
    cv::findContours(bin_find, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    const double min_ratio = params_.min_height_width;
    const double max_ratio = params_.max_height_width;

    for (const auto& contour : contours) {
        // CHAIN_APPROX_SIMPLE 下矩形轮廓可能只有 4 点，不能要求 ≥5。
        if (contour.size() < 3)
            continue;

        Light light(contour);
        if (light.height <= 1e-6)
            continue;

        // 细长比 = 高 / 顶边。右上→右下与高是对边，不能用来筛。
        const double short_side = light.top_width();
        if (short_side <= 1e-6)
            continue;
        const double ratio = light.height / short_side;
        if (ratio < min_ratio || ratio > max_ratio)
            continue;

        const float area = light.rotatedRect.size.width * light.rotatedRect.size.height;
        if (area < kMinArea || area > kMaxArea)
            continue;

        if (std::fabs(light.angle) > kMaxAbsAngle)
            continue;

        light.color = classifyColorRgb(r_gt_b, b_gt_r, bin, light.pts);
        if (light.color != 1 && light.color != 2)
            continue;

        valid_lights.push_back(light);
    }
    return valid_lights;
}
