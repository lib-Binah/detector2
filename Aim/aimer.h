#ifndef AIM_AIMER_H
#define AIM_AIMER_H

#include "ArmorBoard.h"
#include "ballistic.h"
#include "vehicle_ekf.h"

#include <map>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

/**
 * 注释契约：本文件说明怎么用 lovod 整车 EKF 推预瞄点。禁止删注释；改选板、飞时迭代或绘制时必须改注释。
 * 检测板心/法线读 ArmorBoard；车心/四板读 VehicleTracker 的同一状态；弹道读 solveBallistic。
 * 禁止显示时另写落差或圆周公式。每个函数须写清作用、输入、输出。
 *
 * 多车：按 ArmorBoard::number 分桶，每种号各一套 VehicleTracker / EKF，禁止把 3 号和 4 号并进同一滤波器。
 * 同号最多两块侧面仍进该号的 11 维状态（primary+support）。单车内部仍是 lovod 圆周。
 * 飞时内用 kv·v 把车心推走、用 kw·ω 把当前相位的板转过去（同一块板超前，小陀螺时准星左右摆），
 * 转出可打锥面再沿缩放后的 ω 切下一块；最后再做重力抬高（弹道在 Aim，lovod 跟踪包里没有）。
 * kv/kw 只乘飞时用的速度拷贝，不改 EKF 状态。
 * overlay：每辆车各画一套；车心青圈连到该车本帧最正对预测板（facing_id），不连飞时相位。
 * 正对板心画切向箭头：v=ω×(板心−车心)，ω 读该车相机系 vyaw。
 */
struct AimParams {
    BallisticParams ballistic;
    // 飞时 yaw += kw·ω·t。1=全额自旋提前；<1 减小左右幅度。不进入滤波器。
    double w_lead_scale = 1.0;
    // 飞时 车心 += kv·v·t。1=全额平移提前；只乘 vx/vy/vz，不改 EKF。
    double v_lead_scale = 1.0;
};

struct AimDrawFlags {
    bool hub = true;     // 青圈：EKF 车心；车心→正对板连线；正对板心 ω 切向箭头
    bool plates = true;  // 半透明色块：EKF 预测的 n 块板面（与 PnP 同尺寸）
    bool aim = true;     // 橙十字：弹道瞄准点
    bool hud = true;     // 文字：id/状态/飞时
    bool vel = true;     // 车心旁：|v|/ω 以及 r1、r2，读相机系速度与 Tracker 当前/另一对半径
};

struct AimTarget {
    bool ok = false;
    VehicleTrackState track_state = VehicleTrackState::Lost;
    int vehicle_id = 0;
    double dist_m = -1.0;
    double fly_time_s = 0.0;
    double nis = 0.0;
    double r_m = 0.0;          // 当前相位半径 r1 或 r2（米），与圆周同一字段
    double rl_m = 0.0;         // 另一对半径（米），禁止另算
    double vyaw_rad_s = 0.0;  // 相机系自旋 ω（绕竖直，rad/s），由跟踪系 vyaw 取反
    double vx_mps = 0.0;      // 相机 X（右）
    double vy_mps = 0.0;      // 相机 Y（下）
    double vz_mps = 0.0;      // 相机 Z（前）
    int phase_id = 0;      // 锁定超前的相位（飞时后，给橙十字）
    int facing_id = 0;     // 本帧最正对相机的预测板下标（min |板yaw−视线yaw|），给车心连线
    cv::Point3d hub_mm;
    cv::Point3d aim_cam_mm;
    cv::Point2f hub_px{-1.f, -1.f};
    cv::Point2f aim_px{-1.f, -1.f};
    cv::Point2f omega_tip_px{-1.f, -1.f};  // 正对板心 ω 箭头尖；ω≈0 或投影失败为 -1
    std::vector<cv::Point2f> plate_px;
    // 每块预测板四角像素（左下、左上、右上、右下），与 ArmorBoard::projectArmorQuad / PnP 物点同源。
    std::vector<std::vector<cv::Point2f>> plate_quads;
    BallisticSolution ballistic;
};

class Aimer {
public:
    /**
     * 作用：空瞄准器。输入：无。输出：未跟踪。
     */
    Aimer() = default;

    /**
     * 作用：读 yaml 的弹速/重力/v·ω 提前系数与 EKF 门槛。
     * 输入：detector_params.yaml。输出：true 表示文件打开成功。
     */
    bool loadConfig(const std::string& path);

    const AimParams& params() const { return params_; }

    /**
     * 作用：丢掉所有车的 EKF。输入：无。输出：下一帧按本帧出现的 number 重新建跟踪。
     */
    void reset();

    /**
     * 作用：按 number 分桶，每号各更新自己的 EKF，再各自算飞时弹道。
     * 输入：armors 已 PnP；dt 为视频帧间隔，≤0 只重算瞄准不推进时间。
     * 输出：当前未 Lost 的每辆车一条 AimTarget；没有车时为空。像心近的排前面。
     */
    std::vector<AimTarget> update(const std::vector<ArmorBoard>& armors, double dt);

    /**
     * 作用：把一辆车的车心/预测板/瞄准点/速度半径画到图上，读该车 AimTarget，不另算 EKF。
     * 预测板四角读 plate_quads（与 PnP 同物点）；v/ω 旁的 kv、kw 读 params_；r1/r2 读 r_m/rl_m。
     * hub 开时再画车心→plate_px[facing_id] 连线，以及该板心→omega_tip_px 的 ω 切向箭头。
     * 输入：img 为 BGR；tgt 为该车 update 结果；flags 为各开关。Lost 且未瞄准则不画。
     * 输出：无。
     */
    void draw(cv::Mat& img, const AimTarget& tgt, const AimDrawFlags& flags) const;

    /**
     * 作用：每辆车各画一套 overlay，读各自 AimTarget，不把异号板画进同一刚体。
     * 输入：img、update 返回的列表、开关。输出：无。
     */
    void draw(cv::Mat& img, const std::vector<AimTarget>& tgts, const AimDrawFlags& flags) const;

private:
    /**
     * 作用：从已更新的单车 EKF 写出一套 overlay/弹道。输入：该号 tracker。输出：AimTarget。
     */
    AimTarget makeTarget(VehicleTracker& tr) const;

    AimParams params_;
    std::string config_path_;  // 新建某号 EKF 时复用同一份 yaml
    std::map<int, std::unique_ptr<VehicleTracker>> trackers_;
};

#endif  // AIM_AIMER_H
