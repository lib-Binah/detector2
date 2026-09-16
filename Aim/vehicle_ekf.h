#ifndef AIM_VEHICLE_EKF_H
#define AIM_VEHICLE_EKF_H

#include "ArmorBoard.h"
#include "ekf.h"

#include <Eigen/Dense>
#include <array>
#include <deque>
#include <memory>
#include <string>
#include <vector>

/**
 * 注释契约：本文件说明 lovod 整车状态为什么是 11 维笛卡尔量测、为什么分 r1/r2 与 z1/z2。
 * 禁止删注释；改状态顺序、圆周公式、相位或 Q/R 时必须同步改注释。
 * 板心/法线必须读 ArmorBoard 的 tvec 与 normal_cam，禁止在此另做 PnP。
 * 每个函数须写清作用、输入、输出。
 *
 * 对齐 auto_aim_by_lovod/rm_target_tracker：相机光学系先变到「前-左-上」跟踪系再滤波
 * （x=相机Z 前，y=-相机X 左，z=-相机Y 上），与 lovod 把装甲位姿变到 odom 后的水平圆周一致。
 * 普通目标状态：
 *   [xc, yc, z1, z2, yaw, vx, vy, vz, vyaw, r1, r2]
 * 量测 z=(板x, 板y, 板z, 板yaw)，h：板 = 车心 - r·(cosθ, sinθ)，θ=yaw+相位。
 * 对侧共用一套半径/高度：相位 0/2 用 (z1,r1)，1/3 用 (z2,r2)。看不见的那对用 updatePartial 冻住。
 * 前哨站 id=6：6 维 [xc,yc,zc,yaw,vyaw,r]，三板 120°，r 固定，用高度给 1/2/3 层编号。
 * 无 IMU：跟踪系相对相机固定，视频里车心会跟着相机晃，这是无 TF 时的必然。
 * 四板 overlay 是 h(x) 推出来的刚体：车心/yaw 的卡尔曼增益不够大时，四块板会一起“粘住”慢爬。
 * lovod 的 Q×dt 是给 odom+IMU 的；本仓库无 TF，必须用 yaml 的 q_acc/q_alpha 白噪声，否则 P 收死。
 */
struct VehicleEkfParams {
    double default_radius_m = 0.20;  // 锁车沿板法线推车心的 r；yaml default_radius_m
    double max_r_m = 0.40;           // r1 上限；yaml max_r_m，lovod 源码钳 0.4
    double max_r_l_m = 0.40;         // r2 上限；yaml max_r_l_m
    double max_dt_s = 0.50;          // 帧间隔超过则重 init（lovod 0.5）
    int tracking_threshold = 3;      // DETECTING 里连续匹配 > 该值才 TRACKING；yaml min_detect_count
    int lost_threshold = 10;         // TMP_LOST 超过该帧 → LOST；yaml max_temp_lost_count
    double root_min_variance = 1e-6;
    // 锁俯仰后板 yaw 是朝向主量测。lovod 在 odom 里加 0.5 是为了压 IPPE 双根，
    // 相机系加这么大会把四板转角冻住，默认只加一点。
    double normal_yaw_extra_variance = 0.02;
    double nis_gate = 20.0;  // 超过则抬 R 降权，不再整帧丢更新（丢更新四板会停住）
    double q_acc = 100.0;    // 平移加速度方差；yaml q_acc。相机系必须大，否则四板跟不上整车
    double q_alpha = 400.0;  // 角加速度方差；yaml q_alpha。决定四板转得快不快
    double q_adapt_ref = 4.0;
    double q_adapt_min = 0.35;
    double q_adapt_max = 20.0;
    double vel_fade = 1.04;  // 每帧把 v/ω 的 P 略放大，防止收死后永远跟不上
    int bootstrap_confirm_frames = 5;
    double bootstrap_nll_margin = 10.0;
    int phase_confirm_threshold = 2;  // 换相要连续这么多帧 NLL 才提交
    bool enable_vx_accelerate = true;
    bool enable_vy_accelerate = false;
    bool enable_vyaw_accelerate = true;
    int min_samples = 6;
    double time_threshold = 0.4;
    double vxy_base_gain = 0.01;
    double vxy_max_gain_factor = 80.0;
    double yaw_gate = 0.12;
    double min_vy_area_ratio = 0.8;
    double max_vy_area_ratio = 1.28;
    double z_drift_speed_threshold = 0.03;
    int z_count_threshold = 2;
    double z_smooth_alpha = 0.35;
    double vyaw_consistency_abs = 0.08;
    double vyaw_consistency_ratio = 0.35;
    double vyaw_base_gain = 0.01;
    double vyaw_max_gain_factor = 80.0;
    double high_spin_start_vyaw = 2.0;
    double stop_observer_window_time = 0.2;
    double stop_observer_vyaw_threshold = 0.25;
    double stop_evidence_duration = 0.1;
    double stop_vyaw_time_constant = 0.12;
    double outpost_radius = 0.275;
    double outpost_z_step = 0.102;
    double direction_yaw_threshold = 1.0;
    double outpost_center_z_alpha = 0.35;
    double id_switch_factor = 1.5;  // 新目标离像心近过旧目标这么多倍才换 id
    // 发布端平滑（lovod 给云台）。相机系 overlay 要跟检测板，默认关，否则车心/连线发黏。
    bool enable_smooth_normal = false;
    double smooth_alpha = 0.03;
    double quit_smooth_distance_threshold = 0.25;
    double model_vel_threshold = 0.15;
    bool enable_smooth_vyaw = true;
    double min_vyaw_smooth_vel = 2.0;
    int min_vyaw_smooth_sample = 3;
    int vyaw_smooth_window_size = 6;
    double max_vyaw_smooth_win_diff = 0.5;
    double max_vyaw_smooth_raw_diff = 0.7;
    bool enable_smooth_outpost = true;
    double outpost_smooth_alpha = 0.005;
    double outpost_quit_smooth_distance_threshold = 0.4;
};

enum class VehicleTrackState {
    Lost = 0,
    Detecting = 1,
    Tracking = 2,
    TempLost = 3,
};

class VehicleTracker {
public:
    enum NormalEKFState {
        xc,
        yc,
        z1,
        z2,
        yaw,
        vx,
        vy,
        vz,
        vyaw,
        r1,
        r2,
        state_dim
    };
    enum OutpostEKFState { o_xc, o_yc, o_zc, o_yaw, o_vyaw, o_r, o_state_dim };

    VehicleTracker();

    /**
     * 作用：从 YAML 的 aim 节点读半径/状态机门槛，缺键保持 lovod 默认；Q/R 若有序列也读。
     * 输入：path 为 detector_params.yaml。输出：true 表示文件打开成功。
     */
    bool loadConfig(const std::string& path);

    const VehicleEkfParams& params() const { return params_; }
    VehicleTrackState state() const { return state_; }
    int trackedId() const { return tracked_id_; }
    int armorNum() const { return armor_num_; }
    /**
     * 作用：当前锁的大小板类型，给预测板矩形用同一套物点宽高。
     * 输入：无。输出：锁车时写入的 SMALL/LARGE；无目标时为 SMALL。
     */
    ArmorType trackedType() const { return tracked_type_; }
    /**
     * 作用：本帧已确认的装甲相位（0..n-1），给飞时超前锁定同一块板。
     * 输入：无。输出：无目标时为 0。
     */
    int lastPhaseId() const { return isOutpostTarget() ? outpost_state_.phase_slot : confirmed_phase_idx_; }
    bool hasTarget() const { return state_ != VehicleTrackState::Lost; }
    const EKFCore::State& ekfState() const { return target_state_; }
    double lastNis() const { return last_nis_; }
    double getDZ() const { return dz_; }
    double getZA() const { return za_; }
    bool isOutpostTarget() const { return tracked_id_ == 6; }

    /**
     * 作用：丢掉当前车，下一帧重新选。输入：无。输出：state=Lost。
     */
    void reset();

    /**
     * 作用：本帧装甲 → 选车（像心最近+id 粘滞）→ 预测/匹配/更新。
     * 输入：armors 为已 PnP 的板；dt 为视频帧间隔秒（暂停/seek 应传 0 并先 reset）。
     * 输出：无；成功时 target_state_ 为后验。
     */
    void step(const std::vector<ArmorBoard>& armors, double dt);

    /**
     * 作用：按当前（或给定跟踪系状态）推出 n 块板的相机系 (x,y,z,yaw)，单位 m / rad。
     * yaw 与 ArmorBoard::projectArmorQuad 相同：atan2(nx, nz)。
     * 输入：x 为跟踪系状态，空则用 target_state_.x。输出：长度 armor_num_；无目标时为空。
     */
    std::vector<Eigen::Vector4d> armorXyza(const Eigen::VectorXd* x = nullptr) const;

    /**
     * 作用：只把跟踪系状态按匀速+匀角速推 dt，不改滤波器（给弹道飞时用）。
     * 输入：x 为跟踪系状态；dt 秒。输出：外推后的状态；dt≤0 原样返回。
     */
    Eigen::VectorXd extrapolate(const Eigen::VectorXd& x, double dt) const;

    /**
     * 作用：飞时用：在当前后验上缩放 v/ω 再外推，不改 EKF。
     * 输入：dt 飞时；v_scale 乘 vx/vy/vz；w_scale 乘 vyaw。输出：跟踪系外推状态。
     */
    Eigen::VectorXd extrapolateLead(double dt, double v_scale, double w_scale) const;

    /**
     * 作用：车心相机系 mm（读发布平滑后的 xy 与 getZC），与 PnP 同一原点。
     * 输入：无。输出：无目标时 (0,0,0)。
     */
    cv::Point3d hubMm() const;

    /**
     * 作用：把跟踪系速度变到相机系，给 overlay。输入：无。输出：(vx右, vy下, vz前, ω绕竖直)。
     */
    void cameraTwist(double& vx, double& vy, double& vz, double& vyaw) const;

    /**
     * 作用：当前相位半径与另一对半径（米），与圆周同一对状态。
     * 输入：无。输出：r_now、r_other。
     */
    void radii(double& r_now, double& r_other) const;

private:
    struct PoseRoot {
        Eigen::Vector3d position{0, 0, 0};  // 跟踪系米
        double yaw{0.0};
        Eigen::Matrix4d visual_covariance{Eigen::Matrix4d::Identity()};
        cv::Point3d source_cam_mm{0, 0, 0};  // 相机系板心，给调试
    };

    struct TrackedArmor {
        double stamp{0.0};
        int id{0};
        int color{0};
        ArmorType type{ArmorType::SMALL};
        double distance_to_image_center{0.0};
        Eigen::Vector2d image_center{0.0, 0.0};
        double image_area{0.0};
        std::vector<PoseRoot> pose_roots;
        uint8_t selected_root_idx{0};
        const PoseRoot& selectedRoot() const { return pose_roots[selected_root_idx]; }
    };

    struct ArmorAssociation {
        struct Entry {
            TrackedArmor* armor{nullptr};
            Eigen::Vector4d z{Eigen::Vector4d::Zero()};
            Eigen::Matrix4d R{Eigen::Matrix4d::Identity()};
            EKFCore::PhaseMatch phase_match;
            EKFCore::MeasurementEvaluation evaluation;
        };
        bool valid{false};
        Entry primary;
        bool has_support{false};
        Entry support;
    };

    struct AccelerateWindow {
        bool init{false};
        struct Sample {
            double stamp{0.0};
            Eigen::Vector2d meas_center{0.0, 0.0};
            Eigen::Vector2d predict_center{0.0, 0.0};
            double image_area{0.0};
        };
        double start_time{0.0};
        Eigen::Vector2d start_center{0.0, 0.0};
        Eigen::Vector2d start_predict_center{0.0, 0.0};
        double time_threshold{0.3};
        double yaw_gate{0.12};
        double base_gain{0.05};
        double max_gain_factor{16.0};
        double min_vy_area_ratio{0.8};
        double max_vy_area_ratio{1.28};
        int min_samples{6};
        std::deque<Sample> samples;
        void reset() {
            init = false;
            start_time = 0.0;
            start_center = Eigen::Vector2d::Zero();
            start_predict_center = Eigen::Vector2d::Zero();
            samples.clear();
        }
    };

    struct CenterYawObserver {
        struct Sample {
            double stamp{0.0};
            double yaw{0.0};
        };
        bool initialized{false};
        double last_wrapped_yaw{0.0};
        double unwrapped_yaw{0.0};
        std::deque<Sample> samples;
        void reset() {
            initialized = false;
            last_wrapped_yaw = 0.0;
            unwrapped_yaw = 0.0;
            samples.clear();
        }
    };

    struct ZAccumulator {
        bool init{false};
        int phase_idx{-1};
        int pair_idx{-1};
        double end_time{0.0};
        double z_sum{0.0};
        int z_count{0};
        void reset() {
            init = false;
            phase_idx = -1;
            pair_idx = -1;
            end_time = 0.0;
            z_sum = 0.0;
            z_count = 0;
        }
    };

    struct ZPairState {
        bool prev_valid{false};
        double prev_time{0.0};
        double prev_mean{0.0};
        double smooth_z{0.0};
        double drift_speed{0.0};
        bool active{false};
        int stable_count{0};
    };

    struct RootPhaseCandidate {
        uint8_t root_idx{0};
        Eigen::Vector4d measurement{Eigen::Vector4d::Zero()};
        Eigen::Matrix4d measurement_noise{Eigen::Matrix4d::Identity()};
        EKFCore::PhaseMatch phase_match;
    };

    struct BootstrapTrack {
        std::unique_ptr<EKFCore> ekf;
        uint8_t seed_root_idx{0};
        int phase_idx{0};
        int support{0};
        double accumulated_nll{0.0};
        double last_stamp{0.0};
        TrackedArmor last_armor;
        double nis{std::numeric_limits<double>::quiet_NaN()};
    };

    struct OutpostState {
        int direction{0};
        int phase_slot{0};
        bool has_last_yaw{false};
        double last_yaw{0.0};
        bool static_mode{true};
        double last_event_time{0.0};
        std::array<int, 3> slot_ids{};
        std::array<std::vector<double>, 3> z_samples{};
        std::array<std::vector<double>, 3> turn_z_samples{};
        int turn_phase_count{0};
        double center_z{std::numeric_limits<double>::quiet_NaN()};
        std::vector<double> phase_z_samples;
    };

    VehicleEkfParams params_;
    VehicleTrackState state_ = VehicleTrackState::Lost;
    std::unique_ptr<EKFCore> ekf_;
    EKFCore::State target_state_;
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd R_;
    Eigen::MatrixXd Q_outpost_;
    Eigen::MatrixXd R_outpost_;
    enum class Model { Normal, Outpost };
    Model model_{Model::Normal};

    int tracked_id_{0};
    int armor_num_{4};
    ArmorType tracked_type_ = ArmorType::SMALL;
    int detect_count_{0};
    int lost_count_{0};
    double time_s_{0.0};
    double last_time_{0.0};
    double last_yaw_{0.0};
    TrackedArmor tracked_armor_;
    std::vector<TrackedArmor> frame_armors_;

    std::array<BootstrapTrack, 2> bootstrap_tracks_;
    bool bootstrap_running_{false};
    std::vector<double> phase_offsets_;
    int confirmed_phase_idx_{0};
    int hypothetic_phase_idx_{-1};
    int hypothetic_phase_count_{0};
    double default_r_{0.2};
    double dz_{0.02};
    double za_{0.0};
    double last_nis_{0.0};

    AccelerateWindow accelerate_window_;
    CenterYawObserver center_yaw_observer_;
    ZAccumulator z_accumulator_;
    std::array<ZPairState, 2> z_pair_states_;
    OutpostState outpost_state_;

    double last_phase_change_time_{0.0};
    double last_phase_change_yaw_{0.0};
    double high_spin_start_vyaw_{1.5};
    double stop_observer_window_time_{0.2};
    double stop_observer_vyaw_threshold_{0.25};
    double stop_evidence_duration_{0.1};
    double stop_evidence_time_{0.0};
    bool stop_evidence_ready_{false};
    bool vyaw_recovery_pending_{false};
    bool stop_evidence_armed_{false};
    double stop_vyaw_time_constant_{0.12};
    double outpost_radius_{0.275};
    double outpost_z_step_{0.102};
    double outpost_vyaw_{0.8 * 3.14159265358979323846};
    double outpost_phase_duration_{2.0 * 3.14159265358979323846 / 3.0 / (0.8 * 3.14159265358979323846)};
    double direction_yaw_threshold_{1.0};
    double outpost_center_z_alpha_{0.15};
    int phase_confirm_threshold_{2};
    int bootstrap_confirm_frames_{5};
    double bootstrap_nll_margin_{10.0};
    double root_min_variance_{1e-6};
    double normal_yaw_extra_variance_{0.02};
    double nis_gate_{20.0};
    bool enable_vx_accelerate_{true};
    bool enable_vy_accelerate_{false};
    bool enable_vyaw_accelerate_{true};
    double vyaw_consistency_abs_{0.08};
    double vyaw_consistency_ratio_{0.35};
    double vyaw_base_gain_{0.01};
    double vyaw_max_gain_factor_{50.0};
    double z_drift_speed_threshold_{0.03};
    int z_count_threshold_{1};
    double z_smooth_alpha_{0.35};

    // process_node 发布平滑
    Eigen::Vector2d smooth_pos_{std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN()};
    Eigen::Vector2d outpost_smooth_pos_{std::numeric_limits<double>::quiet_NaN(),
                                        std::numeric_limits<double>::quiet_NaN()};
    bool enable_smooth_x_{false};
    bool enable_smooth_y_{false};
    int quit_count_x_{0};
    int quit_count_y_{0};
    bool enable_outpost_smooth_x_{false};
    bool enable_outpost_smooth_y_{false};
    std::deque<double> vyaw_smooth_window_;
    double vyaw_seg_sum_{0.0};
    int vyaw_seg_count_{0};
    double published_xc_{0.0};
    double published_yc_{0.0};
    double published_vyaw_{0.0};
    bool last_phase_change_{false};

    std::unique_ptr<EKFCore> makeNormalEKF(const int& active_phase_idx) const;
    void buildNormalModel();
    void buildOutpostModel();
    void init(TrackedArmor* armor);
    void initEKF(const TrackedArmor& armor);
    bool startNormalBootstrap(const TrackedArmor& armor);
    bool advanceBootstrap(const TrackedArmor& armor);
    void update(const std::vector<TrackedArmor*>& armors, double stamp);
    void updateSingleFrame(TrackedArmor* armor, double stamp);
    void updateStateMachine(bool matched);
    bool buildRootMeasurement(const PoseRoot& root, double yaw_reference, Eigen::Vector4d& measurement,
                              Eigen::Matrix4d& visual_covariance) const;
    bool selectNormalRoot(TrackedArmor& armor, RootPhaseCandidate& candidate) const;
    bool selectOutpostRoot(const TrackedArmor& armor, RootPhaseCandidate& candidate) const;
    bool associateNormalArmors(const std::vector<TrackedArmor*>& armors, ArmorAssociation& association);
    bool isPair0Phase(int phase_idx) const;
    std::vector<int> getActiveIndices(int phase_idx) const;
    void decoupleInactiveCovariance(Eigen::MatrixXd& covariance, int phase_idx) const;
    double getYawFromOrientation(const TrackedArmor& armor);
    std::vector<double> getPhaseOffsets() const;
    double computeAdaptiveGain(double v_old, double v_new, double max_gain_factor) const;
    void accelerateConvergence(const TrackedArmor& armor, const Eigen::Vector4d& measurement, double stamp,
                               bool& vx_injected, bool& vy_injected);
    void pushWindowSample(AccelerateWindow& window, double stamp, const Eigen::Vector2d& meas_center,
                          const Eigen::Vector2d& predict_center, double image_area);
    bool estimateWindowVelocity(double& vx_meas, double& vy_meas, double& vx_pred, double& vy_pred,
                                double& area_ratio) const;
    bool allowVyInjection(double vy_meas, double vy_pred, double area_ratio) const;
    void resetCenterYawObserver();
    double selectContinuousCenterYaw(double armor_yaw) const;
    void observeCenterYaw(double stamp, double center_yaw);
    bool estimateCenterYawRate(double& meas_vyaw) const;
    void updateStopEvidence(double dt);
    bool applyStopVyawCorrection(double dt);
    void resetZAccumulator();
    void observeZSample(double meas_z, double stamp, int phase_idx, bool phase_changed);
    void updateZAccumulator(bool& active, int& counter, double drift_speed);
    double getSmoothZ(double z, int phase_idx) const;
    void resetOutpostState();
    void finishOutpostZSegment();
    bool tryMapOutpostID();
    void recalibrateOutpostMapping();
    void updateOutpostState(double meas_yaw, double meas_z, double stamp, bool phase_changed,
                            int next_phase);
    double getZC() const;
    double getCurrentRadius() const;
    double getOtherRadius() const;
    int getOutpostID() const { return outpost_state_.slot_ids[static_cast<std::size_t>(outpost_state_.phase_slot)]; }
    bool isOutpostConfirmed() const { return getOutpostID() != 0; }
    bool isOutpostStatic() const { return tracked_id_ == 6 && outpost_state_.static_mode; }
    Eigen::Vector4d measureArmor(const Eigen::VectorXd& x, int phase_idx) const;
    /**
     * 作用：相机系分段白噪声 Q（位置-速度块），再乘新息缩放。
     * 输入：dt 秒。输出：11×11；q_acc/q_alpha 读 yaml，禁止再用 lovod 的 Q×dt。
     */
    Eigen::MatrixXd makeProcessQ(double dt) const;
    /**
     * 作用：上一帧新息大则放大 Q，TempLost 用最大倍率。输入：无。输出：倍率。
     */
    double processQScale() const;
    /**
     * 作用：只把 vx/vy/vz/vyaw 的方差按 vel_fade 放大。输入：P。输出：就地改。
     */
    void fadeVelocityCov(Eigen::MatrixXd& P) const;
    /**
     * 作用：NIS 超门限时按比例抬 R，这一帧仍更新但少改状态。输入：evaluation、门限。输出：改 R。
     */
    static void inflateOutlierR(EKFCore::MeasurementEvaluation& ev, double gate);
    std::vector<TrackedArmor> makeObservations(const std::vector<ArmorBoard>& armors) const;
    std::vector<TrackedArmor*> selectTargetArmors(std::vector<TrackedArmor>& armors);
    void refreshPublishSmooth(bool matched);
    void resetVyawSmooth();
    void pushVyawSample(double vyaw);
    double computeMeanVyaw() const;
    void computeVyawSegment(double raw_vyaw, bool phase_change);
    double smoothVyaw(double raw_vyaw, bool phase_change);
    void copyParamsToMembers();
};

#endif  // AIM_VEHICLE_EKF_H
