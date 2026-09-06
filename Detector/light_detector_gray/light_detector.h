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
 * + 可选的对应色 HSV 面积比门限（红/蓝下限分开，use_hsv_filter 关掉则不算过滤）。
 * 红灯 H 为两段 OR：low 与 high（配置 red_h_*_low / red_h_*_high），因为 OpenCV H 在 0/180 环绕。
 * 过曝灯芯在灰度上仍是高亮，单阈值即可接住；HSV 只做二次确认，不代替灰度找轮廓。
 * RGB 投票和 HSV 占比在灯条四顶点的轴对齐包围盒 ROI 上计算：像素判定与整图填凸包相同，
 * 避免每帧整图 BGR2HSV/inRange、以及每根灯 zeros(整图)。
 * hsv_ratio 始终写入 Light，筛选是否用它读 params_.use_hsv_filter；overlay 仍读同一字段。
 * 几何/HSV 数值默认来自 config/detector_params.yaml；缺文件时用下面结构体初值（与新配置一致）。
 */

/**
 * 单色 HSV 上下限。OpenCV H∈[0,180]。
 * 红：h_min_low–h_max_low 或（若启用）h_min_high–h_max_high 任一命中，再乘 S/V。
 * 蓝：只用 low 段（use_h_high=false）。筛占比和 inRange 都读同一组。
 */
struct HsvParams {
    int h_min_low = 0;
    int h_max_low = 180;
    int h_min_high = 0;
    int h_max_high = 0;
    bool use_h_high = false;  // true：高段与低段 OR，用于红的 156–180
    int s_min = 0, s_max = 255;
    int v_min = 0, v_max = 255;
};

struct LightParams {
    // 缺配置文件时的灰度初值；有 yaml 时以 light.gray_thres 为准。进度条也读这份。
    static constexpr int kGrayThresDefault = 145;
    int gray_thres = kGrayThresDefault;  // 灰度下限，像素 > 该值视为前景；播放时可调
    int close_ksize = 3;              // 闭运算核边长，须为奇数
    double min_height_width = 1.5;    // 细长比下限：Light::ratio = height/width
    double max_height_width = 15.0;   // 细长比上限，过瘦噪声丢掉
    int min_area = 20;                // Light::area() 下限
    int max_area = 3500;              // Light::area() 上限
    double max_abs_angle = 25.0;      // |Light::angle| 上限（度）
    // true：hsv_ratio 须大于红/蓝下限才保留；false：仍写入 hsv_ratio，但不按占比丢灯。
    bool use_hsv_filter = true;
    double hsv_ratio_min_red = 0.6;   // 红灯 hsv_ratio 须严格大于该值
    double hsv_ratio_min_blue = 0.4;  // 蓝灯 hsv_ratio 须严格大于该值
};

class LightDetector {
public:
    /**
     * 作用：空检测器，灰度/几何用 LightParams 默认值，红蓝 HSV 用新配置（红双段 H）。
     * 输入：无。输出：可直接 get_lights；有 yaml 时再 loadConfig 覆盖。
     */
    LightDetector();

    /**
     * 作用：从 YAML 读 light + hsv 节点，覆盖 params_ 与红/蓝 HsvParams。
     * 输入：path 为 detector_params.yaml 一类文件。输出：true 表示打开成功（缺键保持原值）。
     */
    bool loadConfig(const std::string& path);

    /**
     * 作用：整份写入灯条几何/灰度门槛（配置加载或测试用）。
     * 输入：p 为完整 LightParams。输出：无。
     */
    void setParams(const LightParams& p) { params_ = p; }

    /**
     * 作用：写入红/蓝 HSV 区间（与 hsvMask / hsv_ratio 同源）。
     * 输入：red/blue 为 HsvParams。输出：无。
     */
    void setHsvParams(const HsvParams& red, const HsvParams& blue);

    /**
     * 作用：把播放窗口 gray 进度条的值写进检测器。
     * 输入：thres 任意 int，内部夹到 [0,255]。输出：无；之后 get_bin 用新阈值。
     */
    void setGrayThres(int thres);

    /**
     * 作用：打开或关闭 HSV 面积比过滤（不改变 hsv_ratio 的计算方式）。
     * 输入：on 为 true 时，红/蓝占比须大于 params_.hsv_ratio_min_red / hsv_ratio_min_blue。
     * 输出：无；写入 params_.use_hsv_filter，get_lights 与进度条读同一份。
     */
    void setUseHsvFilter(bool on);

    /**
     * 作用：HSV 占比过滤当前是否启用。
     * 输入：无。输出：与 params_.use_hsv_filter / 进度条 hsv_f 同一份。
     */
    bool useHsvFilter() const { return params_.use_hsv_filter; }

    /**
     * 作用：读出当前灰度阈值。
     * 输入：无。输出：[0,255]，与进度条/get_bin 用的是同一份 params_.gray_thres。
     */
    int grayThres() const { return params_.gray_thres; }

    /**
     * 作用：只读查看全部检测参数。
     * 输入：无。输出：LightParams 常引用，不要改它来调阈值（走 setGrayThres / setUseHsvFilter / loadConfig）。
     */
    const LightParams& params() const { return params_; }

    /**
     * 作用：只读红灯 HSV 区间。
     * 输入：无。输出：与 inRange / hsv_ratio 同一份 red_params_。
     */
    const HsvParams& redHsv() const { return red_params_; }

    /**
     * 作用：只读蓝灯 HSV 区间。
     * 输入：无。输出：与 inRange / hsv_ratio 同一份 blue_params_。
     */
    const HsvParams& blueHsv() const { return blue_params_; }

    /**
     * 作用：一帧里检出合法灯条（灰度分割 → 几何筛选 → RGB 判红/蓝 → 可选 HSV 面积比）。
     * 输入：frame 必须是 BGR 三通道；bin_out 非空时写入与检测同一张已闭运算二值图。
     * 输出：通过几何+RGB（以及若 use_hsv_filter 则再过 HSV 占比）的 Light 列表；空帧或非三通道返回空。
     */
    std::vector<Light> get_lights(const cv::Mat& frame, cv::Mat* bin_out = nullptr);

    /**
     * 作用：得到灯条高亮前景，给找轮廓和调试窗用（必须同一张图）。
     * 输入：frame 为 BGR 三通道或单通道灰度。
     * 输出：CV_8UC1，灰度 > gray_thres 为 255，再 close_ksize 椭圆闭运算补灯芯黑洞。
     */
    cv::Mat get_bin(const cv::Mat& frame) const;

private:
    /**
     * 作用：在灯条框 ∩ 灰度前景上，用 R>B / B>R 像素数投票判红蓝（不用 HSV）。
     * 输入：bgr 为当前帧；bin 为 get_bin 结果；pts 为 Light 四顶点
     *      （左上/右上/左下/右下，函数内会改成绕序再在包围盒 ROI 上填凸包）。
     * 输出：1 红 / 2 蓝 / -1 票数相等或框内无前景（调用处丢掉）。
     */
    int classifyColorRgb(const cv::Mat& bgr, const cv::Mat& bin, const cv::Point2f pts[4]) const;

    /**
     * 作用：HSV 图按 H/S/V 做 inRange。红可两段 H OR，蓝单段。逐像素，整帧或 ROI 结果一致。
     * 输入：hsv 为 COLOR_BGR2HSV 结果（可仅为灯条包围盒）；p 为红或蓝 HsvParams。
     * 输出：CV_8UC1，落在区间内为 255。
     */
    cv::Mat hsvMask(const cv::Mat& hsv, const HsvParams& p) const;

    /**
     * 作用：旋转矩形内、符合对应色 HSV 的像素占矩形栅格面积的比例，写入 Light::hsv_ratio。
     * 输入：bgr 为当前帧；p 为已判定颜色的 HsvParams；pts 为 Light 四顶点（左上/右上/左下/右下，
     *      函数内改成绕序，在包围盒 ROI 上填凸包；像素集合与整图填凸包相同）。
     * 输出：[0,1]；分母是 fillConvexPoly 非零像素，框面积为 0 时返回 0。
     */
    double hsvRatioInRotatedRect(const cv::Mat& bgr, const HsvParams& p,
                                 const cv::Point2f pts[4]) const;

    LightParams params_;
    HsvParams red_params_;   // 红灯 HSV：H 低段或高段 OR，筛占比与 inRange 同源
    HsvParams blue_params_;  // 蓝灯 HSV：单段 H，筛占比与 inRange 同源
    // get_bin 为 const，核缓存用 mutable：close_ksize 不变则不重建椭圆核。
    mutable cv::Mat close_kernel_;
    mutable int close_kernel_k_ = -1;
};


