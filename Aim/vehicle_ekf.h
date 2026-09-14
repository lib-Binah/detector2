#ifndef AIM_VEHICLE_EKF_H
#define AIM_VEHICLE_EKF_H

#include "ArmorBoard.h"
#include "ekf.h"

#include <Eigen/Dense>
#include <string>
#include <vector>

/**
 * 注释契约：本文件说明同济整车状态为什么是 11 维、量测为什么是 ypd+装甲 yaw。
 * 禁止删注释；改状态顺序、圆周公式、相位或 Q/R 时必须同步改注释。
 * 板心/法线必须读 ArmorBoard 的 tvec 与 normal_cam，禁止在此另做 PnP。
 * 每个函数须写清作用、输入、输出。
 *
 * 同济 Target 状态（相机系当世界系，视频无 IMU）：
 *   [xc, vxc, yc, vyc, zc, vzc, yaw, vyaw, r, l, h]
 * 水平面是相机 XZ（X 右、Z 前），Y 向下。
 * 装甲 yaw = atan2(normal_cam.x, normal_cam.z)，与 overlay 法线同一向量。
 * 圆周：xa = xc - r sin(yaw+id·2π/n)，za = zc - r cos(yaw+id·2π/n)；
 * 与 init 的 xc = xa + r sinθ 互为逆。四装甲时 id=1/3 用 r+l 和 yc+h。
 * 量测 z = (LOS yaw, LOS pitch, 距离, 装甲 yaw)，H = H_ypd · H_xyza。
 */
struct VehicleEkfParams {
    // 单板锁车：沿背法线推车心的 r 初值（米）。yaml aim.default_radius_m 覆盖此项。
    // 同帧有两块同号板时改由背法线交点估 r，不用这个数当圆半径。
    double default_radius_m = 0.25;
    // 四装甲侧板（id=1/3）半径 = r+l。yaml aim.init_l_m。正值侧板更靠外。
    double init_l_m = 0.0;
    // 侧板高度 y = yc+h（相机 Y 向下）。yaml aim.init_h_m。只动 1/3 相位，不抬车心。
    double init_h_m = 0.0;
    // 初始化时把车心沿相机 -Y 抬高这么多米（画面向上、离开地面）。车心偏低、整圈扎地时加大。
    double init_hub_up_m = 0.0;
    double q_acc = 100.0;            // 平移加速度方差（同济 piecewise white noise v1）
    double q_alpha = 400.0;          // 角加速度方差（同济 v2）
    double max_dt_s = 0.10;          // 帧间隔超过则当相机掉线，重置
    int min_detect_count = 3;        // 连续匹配这么多帧才从 detecting → tracking
    int max_temp_lost_count = 15;    // 短暂丢失超过这么多帧 → lost
    double nis_fail_rate = 0.40;     // 近窗 NIS 失败比例超此值视为未收敛
};

enum class VehicleTrackState {
    Lost = 0,
    Detecting = 1,
    Tracking = 2,
    TempLost = 3,
};

class VehicleTracker {
public:
    VehicleTracker() = default;

    /**
     * 作用：从 YAML 的 aim 节点读半径/侧板初值、Q、状态机门槛。
     * 输入：path 为 detector_params.yaml。输出：true 表示文件打开成功；缺键保持默认。
     */
    bool loadConfig(const std::string& path);

    const VehicleEkfParams& params() const { return params_; }
    VehicleTrackState state() const { return state_; }
    int trackedId() const { return tracked_id_; }
    int armorNum() const { return armor_num_; }
    /**
     * 作用：当前锁的大小板类型，给预测板矩形用同一套物点宽高。
     * 输入：无。输出：setTarget 写入的 SMALL/LARGE；无目标时为 SMALL。
     */
    ArmorType trackedType() const { return tracked_type_; }
    /**
     * 作用：本帧匹配到的装甲相位（0..n-1），给飞时超前锁定同一块板。
     * 输入：无。输出：无目标时为 0。
     */
    int lastPhaseId() const { return last_phase_id_; }
    bool hasTarget() const { return state_ != VehicleTrackState::Lost; }
    const ExtendedKalmanFilter& ekf() const { return ekf_; }

    /**
     * 作用：丢掉当前车，下一帧重新选最近车辆。
     * 输入：无。输出：state=Lost。
     */
    void reset();

    /**
     * 作用：本帧装甲 → 预测/匹配/更新。优先锁定距离最近的那辆车。
     * 输入：armors 为已 PnP 的板；dt 为视频帧间隔秒（暂停/seek 应传 0 并先 reset）。
     * 输出：无；成功时 ekf_.x 为后验整车状态。
     */
    void step(const std::vector<ArmorBoard>& armors, double dt);

    /**
     * 作用：按当前（或给定）状态推出 n 块板的 (x,y,z,yaw)，单位 m / rad。
     * 输入：x 为 11 维状态，空则用 ekf_.x。输出：长度 armor_num_；无目标时为空。
     */
    std::vector<Eigen::Vector4d> armorXyza(const Eigen::VectorXd* x = nullptr) const;

    /**
     * 作用：只把状态按匀速+匀角速推 dt，不改滤波器（给弹道飞时用）。
     * 输入：x 为当前状态；dt 秒。输出：外推后的 11 维；dt≤0 原样返回。
     */
    Eigen::VectorXd extrapolate(const Eigen::VectorXd& x, double dt) const;

    /**
     * 作用：车心相机系 mm，与 PnP 同一原点。
     * 输入：无。输出：无目标时 (0,0,0)。
     */
    cv::Point3d hubMm() const;

private:
    VehicleEkfParams params_;
    VehicleTrackState state_ = VehicleTrackState::Lost;
    ExtendedKalmanFilter ekf_;
    bool ekf_ready_ = false;
    int tracked_id_ = 0;
    int armor_num_ = 4;
    ArmorType tracked_type_ = ArmorType::SMALL;
    int detect_count_ = 0;
    int temp_lost_count_ = 0;
    int last_phase_id_ = 0;

    bool setTarget(const std::vector<ArmorBoard>& armors);
    bool updateTarget(const std::vector<ArmorBoard>& armors, double dt);
    void stateMachine(bool found);
    void initEkf(const ArmorBoard& armor, const std::vector<ArmorBoard>& armors);
    void updateYpda(const ArmorBoard& armor, int phase_id);
    Eigen::Vector3d armorXyz(const Eigen::VectorXd& x, int id) const;
    Eigen::MatrixXd hJacobian(const Eigen::VectorXd& x, int id) const;
    static int armorCountFor(const ArmorBoard& armor);
};

#endif  // AIM_VEHICLE_EKF_H
