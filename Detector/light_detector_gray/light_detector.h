#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "Light.h"

/**
 * 注释契约（给后续修改者 / 智能体）：
 * 本模块注释说明灯条识别各段的作用与原理。禁止为“精简”删除注释；
 * 改灰度阈值、形态学、几何门限、RGB 判色或 HSV 面积比时，必须同步修改对应注释。
 * 几何筛选必须读 Light 的属性或函数（ratio、angle、area()、hsv_ratio 等），与 overlay
 * 显示同源、计算方式相同；禁止在本文件里另写一套高宽比或 HSV 占比。
 * 每个函数须写清作用、输入、输出；调用处也要注明调谁、送什么、得到什么。
 * get_lights 每一步必须逐步注释（get_bin / R>B / findContours / 几何 / RGB / HSV 占比）。
 *
 * 装甲板灯条检测：BGR→灰度阈值分割 + 轮廓几何筛选 + 旋转矩形内 RGB 判色
 * + 对应色 HSV 在旋转矩形内的面积比门限（红/蓝下限分开）。过曝灯芯在灰度上仍是高亮，单阈值即可接住；
 * HSV 只做二次确认，不代替灰度找轮廓。
 */

/** 单色 HSV 上下限。OpenCV H∈[0,180]；红/蓝各一份，筛占比和 inRange 都读同一组。 */
struct HsvParams {
    int h_min = 0, h_max = 180;
    int s_min = 0, s_max = 255;
    int v_min = 0, v_max = 255;
};

struct LightParams {
    // 默认灰度阈值只在这里定一份。成员 gray_thres 不能当 LightParams::gray_thres 用
    //（那是实例字段）；进度条初值请读 kGrayThresDefault。
    static constexpr int kGrayThresDefault = 145;
    int gray_thres = kGrayThresDefault;  // 灰度下限，像素 > 该值视为前景；播放时可调
    double min_height_width = 1.5;    // 细长比下限：Light::ratio = height/width
    double max_height_width = 15.0;   // 细长比上限，过瘦噪声丢掉
};

class LightDetector {
public:
    /**
     * 作用：空检测器，灰度参数用 LightParams 默认值，并写入红/蓝 HSV 默认区间。
     * 输入：无。输出：可直接 get_lights 的实例。
     */
    LightDetector();

    /**
     * 作用：把播放窗口 gray 进度条的值写进检测器。
     * 输入：thres 任意 int，内部夹到 [0,255]。输出：无；之后 get_bin 用新阈值。
     */
    void setGrayThres(int thres);

    /**
     * 作用：读出当前灰度阈值。
     * 输入：无。输出：[0,255]，与进度条/get_bin 用的是同一份 params_.gray_thres。
     */
    int grayThres() const { return params_.gray_thres; }

    /**
     * 作用：只读查看全部检测参数。
     * 输入：无。输出：LightParams 常引用，不要改它来调阈值（走 setGrayThres）。
     */
    const LightParams& params() const { return params_; }

    /**
     * 作用：一帧里检出合法灯条（灰度分割 → 几何筛选 → RGB 判红/蓝 → HSV 面积比）。
     * 输入：frame 必须是 BGR 三通道；bin_out 非空时写入与检测同一张已闭运算二值图。
     * 输出：通过几何+RGB+HSV 占比门限的 Light 列表；空帧或非三通道返回空。
     */
    std::vector<Light> get_lights(const cv::Mat& frame, cv::Mat* bin_out = nullptr);

    /**
     * 作用：得到灯条高亮前景，给找轮廓和调试窗用（必须同一张图）。
     * 输入：frame 为 BGR 三通道或单通道灰度。
     * 输出：CV_8UC1，灰度 > gray_thres 为 255，再 3x3 椭圆闭运算补灯芯黑洞。
     */
    cv::Mat get_bin(const cv::Mat& frame) const;

private:
    /**
     * 作用：在灯条框 ∩ 灰度前景上，用 R>B / B>R 像素数投票判红蓝（不用 HSV）。
     * 输入：r_gt_b、b_gt_r 为整帧比较掩码；bin 为 get_bin 结果；pts 为 Light 四顶点
     *      （左上/右上/左下/右下，函数内会改成绕序再填凸包）。
     * 输出：1 红 / 2 蓝 / -1 票数相等或框内无前景（调用处丢掉）。
     */
    int classifyColorRgb(const cv::Mat& r_gt_b, const cv::Mat& b_gt_r,
                         const cv::Mat& bin, const cv::Point2f pts[4]) const;

    /**
     * 作用：整帧 HSV 按一组 H/S/V 上下限做 inRange，得到该色前景。
     * 输入：hsv 为 COLOR_BGR2HSV 结果；p 为红或蓝 HsvParams。
     * 输出：CV_8UC1，落在区间内为 255。
     */
    cv::Mat hsvMask(const cv::Mat& hsv, const HsvParams& p) const;

    /**
     * 作用：旋转矩形内、符合对应色 HSV 的像素占矩形栅格面积的比例，写入 Light::hsv_ratio。
     * 输入：color_mask 为整帧红或蓝 hsvMask；pts 为 Light 四顶点（左上/右上/左下/右下，
     *      函数内改成绕序再填凸包）。
     * 输出：[0,1]；分母是 fillConvexPoly 非零像素，框面积为 0 时返回 0。
     */
    double hsvRatioInRotatedRect(const cv::Mat& color_mask, const cv::Point2f pts[4]) const;

    LightParams params_;
    HsvParams red_params_;   // 红灯 HSV：H 低段，筛占比与 inRange 同源
    HsvParams blue_params_;  // 蓝灯 HSV：H≈80–105，筛占比与 inRange 同源
};
