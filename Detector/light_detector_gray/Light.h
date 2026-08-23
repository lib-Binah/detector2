#ifndef LIGHT_H
#define LIGHT_H

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * 注释契约（给后续修改者 / 智能体）：
 * 本文件注释说明灯条几何表示的作用与原理。禁止为“精简代码”删除注释；
 * 改字段含义、顶点顺序、高宽公式或 tan_xy 定义时，必须同步改注释。
 *
 * 单根装甲板侧边灯条的几何结果。
 * 原理：轮廓 → 轴对齐外接框 + 最小外接旋转矩形 →
 * 四顶点排成 左上 / 右上 / 左下 / 右下（不是绕序）。
 *
 * 高：左上 → 左下 的距离（灯条长边）。
 * 宽：左上 → 右上 的距离（灯条短边）。
 * 细长比 = height / width。
 * tan_xy：右上→右下边的 Δx/Δy，竖直时接近 0。
 */
class Light {
public:
    cv::RotatedRect rotatedRect;  // 最小外接旋转矩形：中心、朝向、OpenCV 宽高
    cv::Rect rect;                // 轴对齐外接矩形：给后续装甲板配对用
    double height = 0.0;          // 左上角到左下角的距离
    double width = 0.0;           // 左上角到右上角的距离
    float tan_xy = 0.0f;          // 右上→右下的 Δx/Δy，近水平时绝对值会被放大
    int color = -1;               // 1=red, 2=blue, 其他=未知（由检测器 RGB 写入）
    cv::Point2f pts[4]{};         // [0]左上 [1]右上 [2]左下 [3]右下

    Light() = default;

    /**
     * 由轮廓构造几何（颜色可稍后写入）。
     * @param contour 输入轮廓，至少 3 点
     * @param color   颜色 ID（1 红 / 2 蓝），检测阶段常先传 -1
     */
    explicit Light(const std::vector<cv::Point>& contour, int color = -1);

    /** 旋转矩形中心，画标注时当锚点 */
    cv::Point2f center() const { return rotatedRect.center; }

    /** 高/宽 = 长边/短边，筛灯条用这个 */
    double hw_ratio() const { return width > 1e-6 ? height / width : 0.0; }

    /** 按绕序画闭合四边形；红→亮黄，蓝→亮橙，避免和灯体抢色 */
    void draw_light(cv::Mat& img) const;
};

#endif  // LIGHT_H
