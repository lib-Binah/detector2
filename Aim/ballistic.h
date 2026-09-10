#ifndef AIM_BALLISTIC_H
#define AIM_BALLISTIC_H

#include <opencv2/core.hpp>

/**
 * 注释契约：本文件说明弹道为什么这么算。禁止删注释；改重力方向、迭代量或单位时必须改注释。
 * 每个函数须写清作用、输入、输出。
 *
 * 弹道解算（重力抛物，无空气阻力）。
 * 相机系与 PnP 相同：X 右、Y 下、Z 前。枪口近似在光心。
 * 重力沿 +Y（相机水平朝前时 Y 朝地）。要打到目标，枪口要瞄得比目标更高，
 * 即补偿后的瞄准点 Y 更小（图像上方）。
 * 飞行时间迭代：用当前瞄准点距离 / 弹速估 t，再把 ½gt² 加到目标高度上。
 * 预瞄点必须读本模块写出的 aim_cam_mm，禁止显示时另写一套落差公式。
 */
struct BallisticParams {
    double bullet_speed_mps = 25.0;  // 出膛速度；步兵常见 15–30
    double gravity_mps2 = 9.81;      // 重力加速度
    int iterations = 8;              // t 与落差互相依赖，迭代几次就够
};

struct BallisticSolution {
    bool ok = false;
    double fly_time_s = 0.0;     // 弹丸飞到瞄准点的时间
    double yaw_rad = 0.0;        // 水平角：atan2(X, Z)，右偏为正
    double pitch_rad = 0.0;      // 补偿后的高低角：相对光轴，上仰为负（Y 向下）
    cv::Point3d aim_cam_mm;      // 枪口应对准的相机系点（已抬高）
};

/**
 * 作用：由目标相机系位置迭代出带重力补偿的瞄准点和飞行时间。
 * 输入：target_cam_mm 为要打的点（mm，与 ArmorBoard::tvec 同源）；p 为弹速/重力。
 * 输出：ok=false 表示弹速非法或目标在相机后方；成功时 aim_cam_mm 比目标更靠图像上方。
 */
BallisticSolution solveBallistic(const cv::Point3d& target_cam_mm, const BallisticParams& p);

#endif  // AIM_BALLISTIC_H
