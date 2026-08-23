/**
 * 注释契约：禁止随意删除本文件注释；改顶点排序、高宽公式、夹角或绘制颜色时必须改注释。
 *
 * 灯条几何构造与可视化：minAreaRect → 顶点重排 → 高/宽/倾角 → 画框。
 */
#include "Light.h"

#include <algorithm>
#include <cmath>

Light::Light(const std::vector<cv::Point>& contour, int color)
    : color(color) {
    rect = cv::boundingRect(contour);      // 轴对齐框，配对时看位置范围
    rotatedRect = cv::minAreaRect(contour);
    rotatedRect.points(pts);               // 四顶点，绕序不固定，下面重排

    // minAreaRect 顶点顺序不固定。先按 y 分成上下两组，再各自按 x 排成左右，
    // 得到临时：0 左上、1 右上、2 左下、3 右下；再交换底边两点，
    // 最终 pts: 0 左上、1 右上、2 右下、3 左下（顺时针）。
    std::sort(pts, pts + 4, [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.y < b.y;
    });
    if (pts[0].x > pts[1].x)
        std::swap(pts[0], pts[1]);
    if (pts[2].x > pts[3].x)
        std::swap(pts[2], pts[3]);
    std::swap(pts[2], pts[3]);  // 底边 左下/右下 → 右下/左下

    // 高：左上 → 左下；宽：右上 → 右下（对边，≈高）。
    height = cv::norm(pts[0] - pts[3]);
    width = cv::norm(pts[1] - pts[2]);

    // 高边向量取左上→左下 (Δx, Δy)。竖直向下时 Δx=0、Δy>0，angle=0。
    // atan2(Δx, Δy)：灯条底部向右偏 → 正角。改“竖直”定义时改这里。
    const float dx = pts[3].x - pts[0].x;
    const float dy = pts[3].y - pts[0].y;
    angle = std::atan2(dx, dy) * 180.0 / CV_PI;
}

void Light::draw_light(cv::Mat& img) const {
    constexpr int kThickness = 1;  // 细线，能看见轮廓即可
    // 已按 左上→右上→右下→左下 排列，依次连线闭合即可。
    constexpr int kOrder[] = {0, 1, 2, 3};

    // 边框不直接画红/蓝，避免和灯条本体抢色：红灯条亮黄，蓝灯条亮橙（BGR）。
    cv::Scalar edge_color(0, 0, 0);
    if (color == 1)
        edge_color = {0, 255, 255};
    else if (color == 2)
        edge_color = {0, 165, 255};

    for (int i = 0; i < 4; ++i) {
        const cv::Point2f& p1 = pts[kOrder[i]];
        const cv::Point2f& p2 = pts[kOrder[(i + 1) % 4]];
        cv::line(img, p1, p2, edge_color, kThickness);
    }
}
