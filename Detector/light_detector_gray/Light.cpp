/**
 * 注释契约：禁止随意删除本文件注释；改顶点排序、高宽公式、夹角或绘制颜色时必须改注释。
 * 高/宽/ratio/angle 只在本构造函数算一次并写入成员；检测筛选和 overlay 必须读这些
 * 成员（或 area()），禁止各自再算一遍，保证计算方式相同。
 * 每个函数须写清作用/输入/输出；本文件内的 OpenCV 调用处也要注明。
 *
 * 灯条几何构造与可视化：minAreaRect → 顶点重排 → 高/宽/倾角 → 画框。
 */
#include "Light.h"

#include <algorithm>
#include <cmath>

/**
 * 作用：由轮廓填充灯条几何（颜色与 hsv_ratio 可稍后写）。
 * 输入：contour 至少 3 点；color 1 红 / 2 蓝，检测时常传 -1。
 * 输出：写入 rect、rotatedRect、pts、height、width、ratio、angle；hsv_ratio 保持 0。
 */
Light::Light(const std::vector<cv::Point>& contour, int color)
    : color(color) {
    // boundingRect：contour → 轴对齐框，给后续装甲板配对看位置。
    rect = cv::boundingRect(contour);
    // minAreaRect：contour → 最小外接旋转矩形（中心/角/OpenCV 宽高）。
    rotatedRect = cv::minAreaRect(contour);
    // points：旋转矩形 → 四个顶点，绕序不固定，下面重排。
    rotatedRect.points(pts);

    // minAreaRect 顶点顺序不固定。先按 y 分成上下两组，再各自按 x 排成左右，
    // 得到 pts: 0 左上、1 右上、2 左下、3 右下，供配对和画框使用。
    std::sort(pts, pts + 4, [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.y < b.y;
    });
    if (pts[0].x > pts[1].x)
        std::swap(pts[0], pts[1]);
    if (pts[2].x > pts[3].x)
        std::swap(pts[2], pts[3]);

    // 高：左上→左下；宽：左上→右上（短边）。ratio 当场写入，检测和 overlay 都读它。
    height = cv::norm(pts[0] - pts[2]);
    width = cv::norm(pts[0] - pts[1]);
    ratio = (width > 1e-6) ? height / width : 0.0;

    // 右上→右下边相对竖直的夹角。竖直时 dy 大、angle≈0；
    // atan2(dx, dy) 在 dy≈0（近水平）时得到 ±90°，检测器用 |angle| 门限丢掉。
    const float dx = pts[1].x - pts[3].x;
    const float dy = pts[3].y - pts[1].y;
    angle = std::atan2(dx, dy) * 180.0 / CV_PI;
}

/**
 * 作用：把灯条闭合框画到图上（红→亮黄，蓝→亮橙）。
 * 输入：img 为 BGR 图，就地画线。输出：无返回值。
 */
void Light::draw_light(cv::Mat& img) const {
    constexpr int kThickness = 1;  // 细线，能看见轮廓即可
    // pts 不是绕序（2=左下、3=右下）。画框按 左上→右上→右下→左下 连线。
    constexpr int kOrder[] = {0, 1, 3, 2};

    // 边框不直接画红/蓝，避免和灯条本体抢色：红灯条亮黄，蓝灯条亮橙（BGR）。
    cv::Scalar edge_color(0, 0, 0);
    if (color == 1)
        edge_color = {0, 255, 255};
    else if (color == 2)
        edge_color = {0, 165, 255};

    for (int i = 0; i < 4; ++i) {
        const cv::Point2f& p1 = pts[kOrder[i]];
        const cv::Point2f& p2 = pts[kOrder[(i + 1) % 4]];
        // line：两点 → 画在 img 上的一条边。
        cv::line(img, p1, p2, edge_color, kThickness);
    }
}
