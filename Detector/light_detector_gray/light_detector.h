#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "Light.h"

/**
 * 注释契约（给后续修改者 / 智能体）：
 * 本模块注释说明灯条识别各段的作用与原理。禁止为“精简”删除注释；
 * 改灰度阈值、形态学、几何门限或 RGB 判色时，必须同步修改对应注释。
 *
 * 装甲板灯条检测：BGR→灰度阈值分割 + 轮廓几何筛选 + 旋转矩形内 RGB 判色。
 * 不用 HSV 取色。过曝灯芯在灰度上仍是高亮，单阈值即可接住。
 */
struct LightParams {
    int gray_thres = 180;           // 灰度下限，像素 > 该值视为前景；播放时可调
    int min_height_width = 3;       // 细长比下限：高 / 宽（左上→左下 / 左上→右上）
    int max_height_width = 15;      // 细长比上限，过瘦噪声丢掉
};

class LightDetector {
public:
    LightDetector();

    /** 夹紧到 [0,255] 后写入；播放窗口 gray 进度条走这里 */
    void setGrayThres(int thres);
    int grayThres() const { return params_.gray_thres; }
    const LightParams& params() const { return params_; }

    /**
     * 灰度阈值 → 闭运算 → 外轮廓 → 几何筛选 → RGB 判红/蓝。
     * @param bin_out 非空时写出与检测一致的二值图（已闭运算），给调试窗用
     */
    std::vector<Light> get_lights(const cv::Mat& frame, cv::Mat* bin_out = nullptr);

    /** 调试用二值：灰度阈值 + 3x3 椭圆闭运算，与 get_lights 同一套掩码 */
    cv::Mat get_bin(const cv::Mat& frame) const;

private:
    /**
     * 旋转矩形 ∩ 二值前景内，比较 R>B 与 B>R 的像素面积。
     * @return 1 红 / 2 蓝 / -1 相等或没有前景
     */
    int classifyColorRgb(const cv::Mat& r_gt_b, const cv::Mat& b_gt_r,
                         const cv::Mat& bin, const cv::Point2f pts[4]) const;

    LightParams params_;
};
