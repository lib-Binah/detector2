#ifndef AIM_VEHICLE_EKF_H
#define AIM_VEHICLE_EKF_H

#include "ArmorBoard.h"
#include "ekf.h"

#include <Eigen/Dense>
#include <map>
#include <utility>

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
 * 同济长短轴：前后板半径 r、高度 yc；左右板半径 r+l、高度 yc+h。对侧共用，没有四套半径。
 * l = r_左右 − r_前后，h = y_左右 − y_前后（本相机系 Y 向下）。
 * 圆周（相机系，对应同济世界系的 cos/sin 互换）：xa = xc - r sin(yaw+id·2π/n)，
 * za = zc - r cos(yaw+id·2π/n)；与 init 的 xc = xa + r sinθ 互为逆。
 * 四装甲 id=1/3 用 r+l 和 yc+h。量测 z = (LOS yaw, pitch, 距离, 装甲 yaw)，H = H_ypd · H_xyza。
 * 同济每块可见板都做 ypda 更新，不冻 r；l/h 只在 id=1/3 的雅可比里非零。
 * 锁车板固定相位 0。转到相位 1/3（差 90°）时，用观测 Y 对 yc+h 与 yc−h 比误差，
 * 决定这块板比锁车板更高还是更低（只改 h 符号/初值），并冻 yc，禁止用车心高度去贴 90° 板。
 * 90° 邻板常是另一种大小，匹配只要求同号同色，不要求 type。
 * r 与 r+l 有 yaml 上限（max_r_m / max_r_l_m），更新后钳位，避免半径被量测拉飞；小于 0.05 仍丢目标。
 * 锁车只按最近一块板 + 给定半径 init，l=h=0，P0 的 r/l/h 为 1；量测从下一帧开始。
 * 跟踪进入 Tracking 且本帧同时看见两块同号同色板时，按（数字, 颜色）缓存 r/l/h；
 * 再锁同号同色车时用缓存当几何初值，P0 的 r/l/h 很小。单板不记，避免 r 与车心糊在一起。
 * v/ω：首帧量测前用车心位移/转角差给初值；之后用新息放大 Q（机动跟上）、过小新息缩小 Q（少抖），
 * 每帧再把 v/ω 的 P 略放大，避免协方差收死后永远跟不上加速。
 */
struct VehicleEkfParams {
    // 同济步兵/英雄/平衡：沿板背法线推车心的 r 初值（米）。yaml aim.default_radius_m，默认 0.2。
    double default_radius_m = 0.20;
    double max_r_m = 0.50;           // 相位 0/2 半径 r 上限（米）；yaml max_r_m
    double max_r_l_m = 0.50;         // 相位 1/3 半径 r+l 上限（米）；yaml max_r_l_m
    double q_acc = 100.0;            // 平移加速度方差（同济 piecewise white noise v1）
    double q_alpha = 400.0;          // 角加速度方差（同济 v2）
    double q_adapt_ref = 4.0;        // 新息 χ² 名义值（4 维量测均值）；越大越不轻易放大 Q
    double q_adapt_min = 0.35;       // 新息小时 Q 缩到这么多倍，匀速段更平滑
    double q_adapt_max = 20.0;       // 新息大时 Q 最多放大这么多倍，几帧内跟上加速/起转
    double vel_fade = 1.04;          // 每帧把 v/ω 的 P 乘这么多，防止收死
    double max_dt_s = 0.10;          // 同济相机离线门槛 0.1s
    int min_detect_count = 3;        // 连续匹配这么多帧才从 detecting → tracking
    int max_temp_lost_count = 15;    // 短暂丢失超过这么多帧 → lost
    double nis_fail_rate = 0.40;     // 窗内 NIS 失败次数 ≥ 该比例×window_size 则重置（同济 0.4×100）
};

/**
 * 同号同色车在双板可见时记下的整车几何（与位姿无关）。再锁这辆车时当 r/l/h 初值。
 */
struct VehicleGeomPrior {
    double r_m = 0.20;
    double l_m = 0.0;
    double h_m = 0.0;
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
     * 作用：从 YAML 的 aim 节点读半径初值与上限、Q、自适应缩放、速度遗忘、状态机与 NIS 门槛。
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
     * 作用：丢掉当前车，下一帧重新选最近车辆。不清除同号同色几何缓存。
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
    int tracked_color_ = 0;  // 1 红 / 2 蓝；与 ArmorBoard::color 同源
    int update_count_ = 0;   // 本段跟踪的 ypda 次数
    bool need_vel_bootstrap_ = false;  // 锁车后第一次匹配时用位移给 v/ω
    bool odd_h_resolved_ = false;  // 本段跟踪是否已用 90° 板的 yc±h 误差定过高低
    // 键：(number, color)。reset / seek 不丢，同号同色再锁车时复用 r/l/h。
    std::map<std::pair<int, int>, VehicleGeomPrior> geom_priors_;

    bool setTarget(const std::vector<ArmorBoard>& armors);
    bool updateTarget(const std::vector<ArmorBoard>& armors, double dt);
    void stateMachine(bool found);
    void initEkf(const ArmorBoard& armor, const VehicleGeomPrior* prior);
    /**
     * 作用：双板可见时把当前 r/l/h 写入 geom_priors_。
     * 输入：n_visible_plates 为本帧该车 pnp_ok 且同号同色的板数（大小板都算，90° 邻板常不同 type）。
     * 输出：≥2 且 Tracking 时覆盖该（数字,颜色）缓存；否则不写。
     */
    void maybeSaveGeomPrior(int n_visible_plates);
    /**
     * 作用：把 r、r+l 压进 yaml 上限，防止半径被量测拉飞。
     * 输入：无（读 max_r_m / max_r_l_m 与 x[8]、x[9]、x[10]）。输出：就地改 ekf_.x 的 r、l、h；不改车心。
     */
    void clampArmorRadii();
    /**
     * 作用：锁车后第一帧量测结束，用后验相对预测的位移 / dt 写入 v/ω，避免从 0 爬很多帧。
     * 输入：x_pred 为预测后状态；dt>0。输出：改 ekf_.x 的 vxc/vyc/vzc/vyaw；成功则清 need_vel_bootstrap_。
     */
    void bootstrapVelFromArmor(const Eigen::VectorXd& x_pred, double dt);
    /**
     * 作用：第一次看到相位 1/3 时，用观测 Y 对 yc+h、yc−h 做误差，判定 90° 板更高还是更低。
     * |h|≈0 则 h←观测Y−yc；否则谁更贴观测就保留或翻 h 的符号。不改 yc。
     * 输入：meas_xyz 为该板 PnP 米制（与 armorXyzM 同一 tvec/1000）。
     * 输出：改 x[10]；已决议过则直接返回。
     */
    void resolveOddPairHeight(const Eigen::Vector3d& meas_xyz);
    /**
     * 作用：ypda 量测更新。R 随视线偏角和距离变大（同济 log 式）。
     * 输入：armor 为 pnp_ok 板；phase_id 为 0..n-1；freeze_hub_y 为 true 时冻 yc/vyc
     * （90° 板的高低差只进 h，避免车心被整段高度差拽飞）。
     * 输出：ekf_ 后验；freeze 时 x[2]、x[3] 保持更新前的值。
     */
    void updateYpda(const ArmorBoard& armor, int phase_id, bool freeze_hub_y = false);
    Eigen::Vector3d armorXyz(const Eigen::VectorXd& x, int id) const;
    Eigen::MatrixXd hJacobian(const Eigen::VectorXd& x, int id) const;
    static int armorCountFor(const ArmorBoard& armor);
};

#endif  // AIM_VEHICLE_EKF_H
