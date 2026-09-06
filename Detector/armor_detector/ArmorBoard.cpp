/**
 * 注释契约：禁止随意删除本文件注释；改端点公式、物点尺寸、内参或绘制颜色时必须改注释。
 * light_ratio / center_distance / angle / light_angle_diff 只在构造函数算一次并写入成员；
 * 配对筛选和 overlay 必须读这些成员，禁止各自再算一遍。
 * 每个函数须写清作用/输入/输出；本文件内的 OpenCV 调用处也要注明。
 *
 * 装甲板几何：左右灯条端点 → 长度比/中心距/倾角/灯条倾角差 → 画框 → IPPE PnP。
 */
#include "ArmorBoard.h"
#include "Detector/detector_config.h"

#include <algorithm>
#include <cmath>
#include <vector>

// 物点单位 mm。这是两灯条构成的矩形（不是外扩后的数字面）。
// 默认与旧硬编码一致；loadPnpConfig 从 yaml 覆盖。改尺寸时 PnP 距离会一起变。
float ArmorBoard::small_width_mm_ = 135.f;
float ArmorBoard::small_height_mm_ = 55.f;
float ArmorBoard::large_width_mm_ = 225.f;
float ArmorBoard::large_height_mm_ = 55.f;

// 与 detector/ArmorBoard.cpp、clone_detector/tools/detect_video.py 同一份默认内参；畸变置零。
cv::Mat ArmorBoard::camera_matrix_ = (cv::Mat_<double>(3, 3) <<
    1.7774091341308808e+03, 0.0, 7.1075979428865026e+02,
    0.0, 1.7754170626354828e+03, 5.3472407285624729e+02,
    0.0, 0.0, 1.0);
cv::Mat ArmorBoard::dist_coeffs_ = (cv::Mat_<double>(5, 1) << 0, 0, 0, 0, 0);

/**
 * 作用：灯条上端点 = 左上与右上的中点。
 * 输入：读 Light::pts。输出：图像坐标。
 */
cv::Point2f ArmorBoard::topOf(const Light& light) {
    return (light.pts[0] + light.pts[1]) * 0.5f;  // 左上、右上中点
}

/**
 * 作用：灯条下端点 = 左下与右下的中点。
 * 输入：读 Light::pts。输出：图像坐标。
 */
cv::Point2f ArmorBoard::bottomOf(const Light& light) {
    return (light.pts[2] + light.pts[3]) * 0.5f;  // 左下、右下中点
}

/**
 * 作用：两灯条填几何；左右按中心 x 排序；四个配对量当场写入。
 * 输入：两根 Light。输出：left/right/端点/center/light_ratio/center_distance/angle/light_angle_diff。
 */
ArmorBoard::ArmorBoard(const Light& a, const Light& b) {
    // 左灯条 = 中心更靠左的那根；后续 PnP 点序依赖这个约定。
    if (a.center().x < b.center().x) {
        left = a;
        right = b;
    } else {
        left = b;
        right = a;
    }

    left_top = topOf(left);
    left_bottom = bottomOf(left);
    right_top = topOf(right);
    right_bottom = bottomOf(right);
    center = (left.center() + right.center()) * 0.5f;

    // 长度比：短 / 长，必须读 Light::height（与灯条检测同源）。
    const double h1 = left.height;
    const double h2 = right.height;
    const double h_max = std::max(h1, h2);
    light_ratio = (h_max > 1e-6) ? std::min(h1, h2) / h_max : 0.0;

    // 中心距用平均灯条高归一化，远近尺度近似不变。
    const double h_avg = 0.5 * (h1 + h2);
    const double pixel_dist = cv::norm(left.center() - right.center());
    center_distance = (h_avg > 1e-6) ? pixel_dist / h_avg : 0.0;

    // 两中心连线相对水平的倾角：|atan(dy/dx)|。dx≈0 时当成 90°。
    const cv::Point2f diff = left.center() - right.center();
    if (std::fabs(diff.x) < 1e-6f)
        angle = 90.0;
    else
        angle = std::fabs(std::atan(diff.y / diff.x)) * 180.0 / CV_PI;

    // 两灯条自身倾角差：必须读 Light::angle（与灯条 overlay 的 ang 同源），不要另算一遍。
    light_angle_diff = std::fabs(left.angle - right.angle);
}

/**
 * 作用：画装甲板四边形和交叉线；小板金黄，大板绿。
 * 输入：BGR 图，就地画线。输出：无返回值。
 */
void ArmorBoard::draw_armor(cv::Mat& img) const {
    constexpr int kThickness = 2;
    // 小板金黄、大板绿，和旧 detector 调试色一致，避免和红蓝灯条抢色。
    const cv::Scalar color = (type == ArmorType::SMALL)
                                 ? cv::Scalar(0, 215, 255)
                                 : cv::Scalar(0, 255, 0);

    const cv::Point lt(left_top);
    const cv::Point rt(right_top);
    const cv::Point rb(right_bottom);
    const cv::Point lb(left_bottom);
    cv::line(img, lt, rt, color, kThickness, cv::LINE_AA);  // 上边
    cv::line(img, rt, rb, color, kThickness, cv::LINE_AA);  // 右边
    cv::line(img, rb, lb, color, kThickness, cv::LINE_AA);  // 下边
    cv::line(img, lb, lt, color, kThickness, cv::LINE_AA);  // 左边
    // X：clone_detector 用对角确认配对，overlay 同源四个端点。
    cv::line(img, lt, rb, color, 1, cv::LINE_AA);
    cv::line(img, lb, rt, color, 1, cv::LINE_AA);
    cv::circle(img, center, 3, color, -1, cv::LINE_AA);
}

/**
 * 作用：四共面点 IPPE PnP；物点 mm，距离换成米。
 * 输入：无（读 type 与四端点）。输出：成功且 z>0 为 true。
 */
bool ArmorBoard::solvePnP() {
    pnp_ok = false;
    dist_m = -1.0;

    const bool large = (type == ArmorType::LARGE);
    const float half_x = (large ? large_width_mm_ : small_width_mm_) * 0.5f;
    const float half_y = (large ? large_height_mm_ : small_height_mm_) * 0.5f;

    // 物点：左下起顺时针。Y 正方向朝板下方，与 OpenCV 图像 y 向下一致。
    const std::vector<cv::Point3f> object_points = {
        {-half_x, half_y, 0.f},   // 左下
        {-half_x, -half_y, 0.f},  // 左上
        {half_x, -half_y, 0.f},   // 右上
        {half_x, half_y, 0.f},    // 右下
    };
    const std::vector<cv::Point2f> image_points = {
        left_bottom, left_top, right_top, right_bottom};

    cv::Mat r, t;
    // solvePnP IPPE：四共面点位姿；失败再试 ITERATIVE。
    bool ok = cv::solvePnP(object_points, image_points, camera_matrix_, dist_coeffs_,
                           r, t, false, cv::SOLVEPNP_IPPE);
    if (!ok)
        ok = cv::solvePnP(object_points, image_points, camera_matrix_, dist_coeffs_,
                          r, t, false, cv::SOLVEPNP_ITERATIVE);
    if (!ok)
        return false;

    const double z = t.at<double>(2);
    if (z <= 1e-3)
        return false;  // 相机后方，几何上不合理

    rvec = r;
    tvec = t;
    dist_m = cv::norm(t) / 1000.0;  // mm → m
    pnp_ok = true;
    return true;
}

/**
 * 作用：YAML pnp/camera → 静态物点尺寸与内参。
 * 输入：path。输出：true 表示打开成功。
 */
bool ArmorBoard::loadPnpConfig(const std::string& path) {
    cv::FileStorage fs;
    if (!openDetectorYaml(path, fs))
        return false;

    const cv::FileNode pnp = fs["pnp"];
    readYaml(pnp, "small_width_mm", small_width_mm_);
    readYaml(pnp, "small_height_mm", small_height_mm_);
    readYaml(pnp, "large_width_mm", large_width_mm_);
    readYaml(pnp, "large_height_mm", large_height_mm_);

    const cv::FileNode cam = fs["camera"];
    double fx = camera_matrix_.at<double>(0, 0);
    double fy = camera_matrix_.at<double>(1, 1);
    double cx = camera_matrix_.at<double>(0, 2);
    double cy = camera_matrix_.at<double>(1, 2);
    double k1 = dist_coeffs_.at<double>(0);
    double k2 = dist_coeffs_.at<double>(1);
    double p1 = dist_coeffs_.at<double>(2);
    double p2 = dist_coeffs_.at<double>(3);
    double k3 = dist_coeffs_.at<double>(4);
    readYaml(cam, "fx", fx);
    readYaml(cam, "fy", fy);
    readYaml(cam, "cx", cx);
    readYaml(cam, "cy", cy);
    readYaml(cam, "k1", k1);
    readYaml(cam, "k2", k2);
    readYaml(cam, "p1", p1);
    readYaml(cam, "p2", p2);
    readYaml(cam, "k3", k3);
    camera_matrix_ = (cv::Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
    dist_coeffs_ = (cv::Mat_<double>(5, 1) << k1, k2, p1, p2, k3);
    return true;
}
