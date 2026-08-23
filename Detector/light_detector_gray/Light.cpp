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
    // 得到 pts: 0 左上、1 右上、2 左下、3 右下，供配对和画框使用。
    std::sort(pts, pts + 4, [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.y < b.y;
    });
    if (pts[0].x > pts[1].x)
        std::swap(pts[0], pts[1]);
    if (pts[2].x > pts[3].x)
        std::swap(pts[2], pts[3]);

    // 高：左上→左下；宽：左上→右上（短边）。overlay 的 H/W 就显示这两项。
    height = cv::norm(pts[0] - pts[2]);
    width = cv::norm(pts[0] - pts[1]);

    // 右上→右下边相对竖直的夹角。竖直时 dy 大、angle≈0；
    // atan2(dx, dy) 在 dy≈0（近水平）时得到 ±90°，检测器用 |angle| 门限丢掉。
    const float dx = pts[1].x - pts[3].x;
    const float dy = pts[3].y - pts[1].y;
    angle = std::atan2(dx, dy) * 180.0 / CV_PI;
}

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
        cv::line(img, p1, p2, edge_color, kThickness);
    }
}
