/**
 * 注释契约：禁止随意删除本文件注释；改端点公式、物点尺寸、内参、PnP 点序/消歧或绘制颜色时必须改注释。
 * light_ratio / center_distance / angle / light_angle_diff 只在构造函数算一次并写入成员；
 * 配对筛选和 overlay 必须读这些成员，禁止各自再算一遍。
 * 法线 overlay 必须读 normal_cam，禁止显示时另从 rvec 算一遍。
 * 每个函数须写清作用/输入/输出；本文件内的 OpenCV 调用处也要注明。
 *
 * 装甲板几何：左右灯条端点 → 长度比/中心距/倾角/灯条倾角差 → 画框 → IPPE 双根 PnP。
 */
#include "ArmorBoard.h"
#include "Detector/detector_config.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <opencv2/calib3d.hpp>

namespace {

// 板心必须在相机前方这么远（mm）才算根有效；过近多半是歧义解。
constexpr double kMinDepthMm = 50.0;
// 超过这个深度（mm）当作飞点丢掉。
constexpr double kMaxDepthMm = 20000.0;
// 平均重投影超过这个像素数，这个根不可信。
constexpr double kMaxReprojPx = 8.0;

/**
 * 作用：把 3D 物点投回图像，算四点平均像素误差。
 * 输入：object/image 为 PnP 那四对点；r/t 为一个根；K/D 与求解同一套。
 * 输出：平均 |投影-观测|（像素）。
 */
double meanReprojPx(const std::vector<cv::Point3f>& object_points,
                    const std::vector<cv::Point2f>& image_points,
                    const cv::Mat& rvec,
                    const cv::Mat& tvec,
                    const cv::Mat& K,
                    const cv::Mat& D) {
    std::vector<cv::Point2f> projected;
    cv::projectPoints(object_points, rvec, tvec, K, D, projected);  // 物点+位姿 → 像素
    double sum = 0.0;
    for (size_t i = 0; i < image_points.size(); ++i)
        sum += cv::norm(image_points[i] - projected[i]);
    return sum / static_cast<double>(image_points.size());
}

/**
 * 作用：物系 +Z 在相机系里的方向（R 的第三列）。
 * 输入：rvec 为 Rodrigues 向量。输出：未归一化的相机系向量。
 */
cv::Vec3d objectZInCamera(const cv::Mat& rvec) {
    cv::Mat R;
    cv::Rodrigues(rvec, R);  // rvec → 3×3 旋转，列是物系轴在相机系的方向
    R.convertTo(R, CV_64F);
    return cv::Vec3d(R.at<double>(0, 2), R.at<double>(1, 2), R.at<double>(2, 2));
}

/**
 * 作用：物系 +Z 翻成「远离相机」的单位法线（n_z≥0）。
 * 输入：rvec。输出：相机系单位向量；退化时为零向量。
 * 原理：正面朝向我们时 +Z 指向相机，取反后指向车内；OpenCV y 向下，n_y>0 即略朝下。
 */
cv::Vec3d awayNormalCam(const cv::Mat& rvec) {
    cv::Vec3d n = objectZInCamera(rvec);
    if (n[2] < 0.0)
        n = -n;
    const double len = cv::norm(n);
    if (len < 1e-9)
        return cv::Vec3d(0, 0, 0);
    return n / len;
}

}  // namespace

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
 * 作用：把板法线画成箭头，方向远离相机。
 * 输入：img 为 BGR。输出：无。必须读 normal_cam，与 solvePnP 写入的是同一个向量。
 */
void ArmorBoard::draw_normal(cv::Mat& img) const {
    if (!pnp_ok || tvec.empty() || cv::norm(normal_cam) < 1e-6)
        return;

    cv::Mat t64;
    tvec.convertTo(t64, CV_64F);
    t64 = t64.reshape(1, 3);
    const cv::Point3d origin(t64.at<double>(0, 0), t64.at<double>(1, 0), t64.at<double>(2, 0));
    // 箭头加长：5 倍板高，远距离再用深度的 18%，否则俯仰几乎看不出来。
    const double plate_h = (type == ArmorType::LARGE) ? large_height_mm_ : small_height_mm_;
    const double len = std::max(5.0 * plate_h, 0.18 * origin.z);
    const cv::Point3d tip(origin.x + normal_cam[0] * len,
                          origin.y + normal_cam[1] * len,
                          origin.z + normal_cam[2] * len);
    if (origin.z <= 1e-3 || tip.z <= 1e-3)
        return;  // 有一端在相机后方，投影会翻到图像另一侧

    // 点已在相机系：rvec/tvec 取零，只用与 PnP 相同的 K/D 投到像素。
    const std::vector<cv::Point3f> cam_pts = {
        cv::Point3f(static_cast<float>(origin.x), static_cast<float>(origin.y),
                    static_cast<float>(origin.z)),
        cv::Point3f(static_cast<float>(tip.x), static_cast<float>(tip.y),
                    static_cast<float>(tip.z)),
    };
    const cv::Mat r0 = cv::Mat::zeros(3, 1, CV_64F);
    const cv::Mat t0 = cv::Mat::zeros(3, 1, CV_64F);
    std::vector<cv::Point2f> pix;
    cv::projectPoints(cam_pts, r0, t0, camera_matrix_, dist_coeffs_, pix);
    if (pix.size() != 2)
        return;

    const cv::Point p0(cvRound(pix[0].x), cvRound(pix[0].y));
    const cv::Point p1(cvRound(pix[1].x), cvRound(pix[1].y));
    // 品红：和金黄/绿板框、青字 overlay 分开，一眼能看法线。
    const cv::Scalar magenta(255, 0, 255);
    cv::arrowedLine(img, p0, p1, magenta, 2, cv::LINE_AA, 0, 0.25);
}

/**
 * 作用：四共面点 IPPE 双根 PnP；正面朝向相机，背法线锁定略朝下，避免俯仰跳变。
 * 输入：无（读 type 与四端点）。输出：成功且深度合法为 true。
 */
bool ArmorBoard::solvePnP() {
    pnp_ok = false;
    dist_m = -1.0;
    reproj_error_px = -1.0;
    normal_cam = cv::Vec3d(0, 0, 0);
    rvec.release();
    tvec.release();

    const bool large = (type == ArmorType::LARGE);
    const float half_x = (large ? large_width_mm_ : small_width_mm_) * 0.5f;
    const float half_y = (large ? large_height_mm_ : small_height_mm_) * 0.5f;

    // 物点：左下起顺时针。Y 正方向朝板下方，与 OpenCV 图像 y 向下一致。
    // +Z = X×Y：正面朝向相机时指向相机；画箭头时再翻成远离相机。
    const std::vector<cv::Point3f> object_points = {
        {-half_x, half_y, 0.f},   // 左下
        {-half_x, -half_y, 0.f},  // 左上
        {half_x, -half_y, 0.f},   // 右上
        {half_x, half_y, 0.f},    // 右下
    };
    const std::vector<cv::Point2f> image_points = {
        left_bottom, left_top, right_top, right_bottom};

    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    try {
        // IPPE 平面四对应最多两个根；两个都留下，后面用朝向+重投影选。
        cv::solvePnPGeneric(object_points, image_points, camera_matrix_, dist_coeffs_,
                            rvecs, tvecs, false, cv::SOLVEPNP_IPPE);
    } catch (const cv::Exception&) {
        rvecs.clear();
        tvecs.clear();
    }

    if (rvecs.empty() || tvecs.empty()) {
        cv::Mat r, t;
        // 双根失败再试迭代单根，避免偶发 IPPE 数值失败整帧没距离。
        if (!cv::solvePnP(object_points, image_points, camera_matrix_, dist_coeffs_,
                          r, t, false, cv::SOLVEPNP_ITERATIVE))
            return false;
        rvecs.push_back(r);
        tvecs.push_back(t);
    }

    const size_t nroots = std::min(rvecs.size(), tvecs.size());
    int best = -1;
    bool best_faces_cam = false;
    bool best_points_down = false;
    double best_err = std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < nroots; ++i) {
        cv::Mat r64, t64;
        rvecs[i].convertTo(r64, CV_64F);
        tvecs[i].convertTo(t64, CV_64F);
        t64 = t64.reshape(1, 3);
        const double z = t64.at<double>(2, 0);
        if (z < kMinDepthMm || z > kMaxDepthMm)
            continue;

        const double err = meanReprojPx(object_points, image_points, r64, t64,
                                        camera_matrix_, dist_coeffs_);
        if (err > kMaxReprojPx)
            continue;

        // 物系 +Z 的相机 z 分量：<0 表示板正面朝向我们（看得见灯条的那一面）。
        const cv::Vec3d n_obj = objectZInCamera(r64);
        const bool faces_cam = n_obj[2] < 0.0;
        const cv::Vec3d away = awayNormalCam(r64);
        // OpenCV 相机 y 向下：远离相机的法线 n_y>0 即图像里略朝下。
        const bool points_down = away[1] > 0.0;

        // 两根俯仰接近相反，重投影差往往小于角点抖动，只比误差会上下跳。
        // 朝下先验优先于更小重投影：兵种板上沿更近，背法线应指向车内略朝下。
        const bool better = (best < 0) ||
                            (faces_cam && !best_faces_cam) ||
                            (faces_cam == best_faces_cam && points_down && !best_points_down) ||
                            (faces_cam == best_faces_cam && points_down == best_points_down &&
                             err < best_err);
        if (!better)
            continue;
        best = static_cast<int>(i);
        best_faces_cam = faces_cam;
        best_points_down = points_down;
        best_err = err;
        rvec = r64;
        tvec = t64;
    }

    if (best < 0)
        return false;

    const cv::Vec3d n = awayNormalCam(rvec);
    if (cv::norm(n) < 1e-9)
        return false;
    normal_cam = n;

    dist_m = cv::norm(tvec) / 1000.0;  // mm → m
    reproj_error_px = best_err;
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
