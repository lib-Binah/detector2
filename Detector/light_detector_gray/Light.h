#ifndef LIGHT_H
#define LIGHT_H

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * 注释契约（给后续修改者 / 智能体）：
 * 本文件注释说明灯条几何表示的作用与原理。禁止为“精简代码”删除注释；
 * 改字段含义、顶点顺序、高宽公式、angle 或 hsv_ratio 定义时，必须同步改注释。
 * 检测筛选和 overlay 显示必须读同一个属性或同一个函数（计算方式相同）；
 * 禁止检测用 ratio、显示另算 height/width。改公式只改本类，两边一起变。
 * 每个成员函数须写清作用、输入、输出；调用处也要注明。
 *
 * 单根装甲板侧边灯条的几何结果。
 * 原理：轮廓 → 轴对齐外接框 + 最小外接旋转矩形 →
 * 四顶点排成 左上 / 右上 / 左下 / 右下（不是绕序）。
 *
 * 高：左上 → 左下 的距离（灯条长边）。
 * 宽：左上 → 右上 的距离（灯条短边）。
 * 细长比 ratio：构造时写入 height/width，检测筛选和 overlay 都读这个字段。
 * 夹角：右上→右下边相对竖直的角度（度），atan2(Δx, Δy)，竖直时接近 0。
 * 面积：rotatedRect 的 OpenCV 宽×高，筛面积用 area()，不要在检测器里再乘一遍。
 * hsv_ratio：旋转矩形栅格内符合该色 HSV 的像素占比；筛灯条和 overlay 都读这个字段。
 */
class Light {
public:
    cv::RotatedRect rotatedRect;  // 最小外接旋转矩形：中心、朝向、OpenCV 宽高
    cv::Rect rect;                // 轴对齐外接矩形：给后续装甲板配对用
    double height = 0.0;          // 左上角到左下角的距离
    double width = 0.0;           // 左上角到右上角的距离
    double angle = 0.0;           // 右上→右下相对竖直的夹角（度），近水平时接近 ±90
    double ratio = 0.0;           // height/width，构造时写入；宽过小则为 0
    int color = -1;               // 1=red, 2=blue, 其他=未知（由检测器 RGB 写入）
    double hsv_ratio = 0.0;       // 框内该色 HSV 像素 / 旋转矩形栅格像素；检测与 overlay 同源
    cv::Point2f pts[4]{};         // [0]左上 [1]右上 [2]左下 [3]右下

    /**
     * 作用：空灯条，字段保持默认 0 / -1。
     * 输入：无。输出：未填充的 Light，检测路径不会用这个。
     */
    Light() = default;

    /**
     * 作用：由轮廓算出旋转矩形、四顶点、高/宽/ratio/angle（颜色与 hsv_ratio 可稍后写入）。
     * 输入：contour 至少 3 点；color 为 1 红 / 2 蓝，检测阶段常先传 -1。
     * 输出：写入 rect、rotatedRect、pts、height、width、ratio、angle；color/hsv_ratio 由检测器后写。
     */
    explicit Light(const std::vector<cv::Point>& contour, int color = -1);

    /**
     * 作用：给 overlay 标注当锚点。
     * 输入：无（读 rotatedRect）。输出：旋转矩形中心。
     */
    cv::Point2f center() const { return rotatedRect.center; }

    /**
     * 作用：筛面积、调试用的灯条面积。必须走这里，禁止在检测器里再乘宽高。
     * 输入：无（读 rotatedRect.size）。输出：minAreaRect 的宽×高（像素²）。
     */
    double area() const {
        return static_cast<double>(rotatedRect.size.width) * rotatedRect.size.height;
    }

    /**
     * 作用：按绕序把灯条框画到图上；红→亮黄，蓝→亮橙，避免和灯体抢色。
     * 输入：img 为要画的 BGR 图（就地修改）。输出：无返回值，四条边画在 img 上。
     */
    void draw_light(cv::Mat& img) const;
};

#endif  // LIGHT_H
