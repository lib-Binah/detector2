/**
 * 注释契约：禁止删本文件注释；改重力轴、单位换算或迭代式时必须改注释。
 * 每个函数须写清作用/输入/输出；调用处注明送什么、得到什么。
 *
 * 在「Y 向上」的枪口系里做抛物线，再变回 OpenCV 相机系，避免 Y 向下把 g 的符号搞反。
 */
#include "ballistic.h"

#include <algorithm>
#include <cmath>

/**
 * 作用：重力抛物迭代：瞄准点抬高 ½gt²，用三维距离/弹速更新 t。
 * 输入：target_cam_mm 与 PnP 板心同一相机系；p.bullet_speed_mps>0。
 * 输出：补偿后的瞄准点、飞时、yaw/pitch。
 */
BallisticSolution solveBallistic(const cv::Point3d& target_cam_mm, const BallisticParams& p) {
    BallisticSolution s;
    if (p.bullet_speed_mps <= 1e-3 || target_cam_mm.z <= 1.0)
        return s;

    // 枪口系：X 右、Y 上、Z 前。相机 Y 下 → 取反。
    const double x = target_cam_mm.x / 1000.0;
    const double y_up = -target_cam_mm.y / 1000.0;
    const double z = target_cam_mm.z / 1000.0;
    const double d = std::hypot(x, z);  // 水平距离（XZ）
    if (d < 1e-4)
        return s;

    double h_aim = y_up;
    double t = std::hypot(d, h_aim) / p.bullet_speed_mps;
    const int n = std::max(1, p.iterations);
    for (int i = 0; i < n; ++i) {
        // 子弹下落 ½gt²，要对着比目标更高的点打。
        h_aim = y_up + 0.5 * p.gravity_mps2 * t * t;
        t = std::hypot(d, h_aim) / p.bullet_speed_mps;
    }

    s.ok = true;
    s.fly_time_s = t;
    s.yaw_rad = std::atan2(x, z);
    s.pitch_rad = -std::atan2(h_aim, d);  // 写回相机：上仰使 Y 减小，角度取负方便对照图像
    s.aim_cam_mm = cv::Point3d(target_cam_mm.x, -h_aim * 1000.0, target_cam_mm.z);
    return s;
}
