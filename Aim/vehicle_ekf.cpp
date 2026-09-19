/**
 * 禁止删本文件注释；改圆周公式、笛卡尔量测、相位 NLL、部分更新或跟踪系定义时必须改注释。
 * 检测读 ArmorBoard::positionMm / normal_cam / dist_m / number，禁止另算 PnP。
 * 每个函数须写清作用/输入/输出；调用处注明意图。
 *
 * 滤波对齐 lovod Tracker：水平圆周在「前-左-上」跟踪系，量测是板 xyz+yaw，
 * 双半径 r1/r2、双高度 z1/z2，换相要连续确认，单板 updatePartial 冻另一对。
 * 跟踪系：x=相机Z（前），y=-相机X（左），z=-相机Y（上），与 lovod odom 水平面同构。
 * 无 IPPE 双根时 bootstrap 自动跳过，走单根 initEKF。
 */
#include "vehicle_ekf.h"

#include "Detector/detector_config.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace {

constexpr double kPi = 3.14159265358979323846;

/**
 * 作用：前哨相位 +1 模 3。输入：0..2。输出：下一槽。
 */
int nextOutpostPhase(int phase_idx) {
    return (phase_idx + 1) % 3;
}

/**
 * 作用：角折到 (-π, π]。输入：任意弧度。输出：主值。
 */
double wrapPi(double angle) {
    while (angle > kPi)
        angle -= 2.0 * kPi;
    while (angle < -kPi)
        angle += 2.0 * kPi;
    return angle;
}

/**
 * 作用：从 from 转到 to 的最短有向角。输入：两角。输出：(-π, π]。
 */
double shortestAngularDistance(double from, double to) {
    return wrapPi(to - from);
}

/**
 * 作用：相机光学系 → 跟踪系（前-左-上）。输入：相机米。输出：跟踪系米。
 */
Eigen::Vector3d camToTrk(const Eigen::Vector3d& cam) {
    return {cam.z(), -cam.x(), -cam.y()};
}

/**
 * 作用：跟踪系 → 相机光学系。输入：跟踪系米。输出：相机米。
 */
Eigen::Vector3d trkToCam(const Eigen::Vector3d& trk) {
    return {-trk.y(), -trk.z(), trk.x()};
}

/**
 * 作用：板背法线在跟踪系水平面的 yaw，与 lovod atan2(dir.y, dir.x) 同一约定。
 * 输入：normal_cam。输出：atan2(-nx, nz)。
 */
double armorYawTrk(const cv::Vec3d& n) {
    return std::atan2(-n[0], n[2]);
}

/**
 * 作用：跟踪系板 yaw → 相机系 projectArmorQuad 用的 yaw（atan2(nx,nz)）。
 * 输入：θ_trk。输出：-θ（法线 (cosθ,sinθ,0) 变到相机后 nx=-sinθ, nz=cosθ）。
 */
double yawTrkToCam(double yaw_trk) {
    return -yaw_trk;
}

}  // namespace

/**
 * 作用：按 lovod 默认 Q/R 建普通模型。输入：无。输出：ekf_ 可 init。
 */
VehicleTracker::VehicleTracker() {
    copyParamsToMembers();  // yaml 缺省时用结构体默认，与 lovod processor_config 对齐
    buildNormalModel();
}

/**
 * 作用：把 params_ 抄到 lovod 同名成员，窗口/前哨门槛跟同一份。
 * 输入：无。输出：成员已同步。
 */
void VehicleTracker::copyParamsToMembers() {
    default_r_ = params_.default_radius_m;
    root_min_variance_ = params_.root_min_variance;
    normal_yaw_extra_variance_ = params_.normal_yaw_extra_variance;
    nis_gate_ = params_.nis_gate;
    bootstrap_confirm_frames_ = params_.bootstrap_confirm_frames;
    bootstrap_nll_margin_ = params_.bootstrap_nll_margin;
    phase_confirm_threshold_ = params_.phase_confirm_threshold;
    enable_vx_accelerate_ = params_.enable_vx_accelerate;
    enable_vy_accelerate_ = params_.enable_vy_accelerate;
    enable_vyaw_accelerate_ = params_.enable_vyaw_accelerate;
    accelerate_window_.time_threshold = params_.time_threshold;
    accelerate_window_.yaw_gate = params_.yaw_gate;
    accelerate_window_.base_gain = params_.vxy_base_gain;
    accelerate_window_.max_gain_factor = params_.vxy_max_gain_factor;
    accelerate_window_.min_samples = params_.min_samples;
    accelerate_window_.min_vy_area_ratio = params_.min_vy_area_ratio;
    accelerate_window_.max_vy_area_ratio = params_.max_vy_area_ratio;
    z_drift_speed_threshold_ = params_.z_drift_speed_threshold;
    z_count_threshold_ = params_.z_count_threshold;
    z_smooth_alpha_ = params_.z_smooth_alpha;
    vyaw_consistency_abs_ = params_.vyaw_consistency_abs;
    vyaw_consistency_ratio_ = params_.vyaw_consistency_ratio;
    vyaw_base_gain_ = params_.vyaw_base_gain;
    vyaw_max_gain_factor_ = params_.vyaw_max_gain_factor;
    high_spin_start_vyaw_ = params_.high_spin_start_vyaw;
    stop_observer_window_time_ = params_.stop_observer_window_time;
    stop_observer_vyaw_threshold_ = params_.stop_observer_vyaw_threshold;
    stop_evidence_duration_ = params_.stop_evidence_duration;
    stop_vyaw_time_constant_ = params_.stop_vyaw_time_constant;
    outpost_radius_ = params_.outpost_radius;
    outpost_z_step_ = params_.outpost_z_step;
    direction_yaw_threshold_ = params_.direction_yaw_threshold;
    outpost_center_z_alpha_ = params_.outpost_center_z_alpha;
    outpost_vyaw_ = 0.8 * kPi;
    outpost_phase_duration_ = (2.0 * kPi / 3.0) / outpost_vyaw_;
}

/**
 * 作用：YAML aim → 半径/状态机/NIS，以及可选 q/r 对角。
 * 输入：path。输出：打开成功为 true。
 */
bool VehicleTracker::loadConfig(const std::string& path) {
    cv::FileStorage fs;
    if (!openDetectorYaml(path, fs))
        return false;
    const cv::FileNode n = fs["aim"];
    readYaml(n, "default_radius_m", params_.default_radius_m);
    readYaml(n, "max_r_m", params_.max_r_m);
    readYaml(n, "max_r_l_m", params_.max_r_l_m);
    readYaml(n, "max_dt_s", params_.max_dt_s);
    readYaml(n, "min_detect_count", params_.tracking_threshold);
    readYaml(n, "max_temp_lost_count", params_.lost_threshold);
    readYaml(n, "root_min_variance", params_.root_min_variance);
    readYaml(n, "normal_yaw_extra_variance", params_.normal_yaw_extra_variance);
    readYaml(n, "nis_gate", params_.nis_gate);
    readYaml(n, "id_switch_factor", params_.id_switch_factor);
    readYaml(n, "q_acc", params_.q_acc);
    readYaml(n, "q_alpha", params_.q_alpha);
    readYaml(n, "q_adapt_ref", params_.q_adapt_ref);
    readYaml(n, "q_adapt_min", params_.q_adapt_min);
    readYaml(n, "q_adapt_max", params_.q_adapt_max);
    readYaml(n, "vel_fade", params_.vel_fade);
    readYaml(n, "enable_smooth_normal", params_.enable_smooth_normal);
    if (!(params_.max_r_m > 1e-6))
        params_.max_r_m = 0.4;
    if (!(params_.max_r_l_m > 1e-6))
        params_.max_r_l_m = 0.4;
    copyParamsToMembers();
    buildNormalModel();  // 用（可能更新过的）Q/R 重建核
    const cv::FileNode qn = n["q"];
    if (!qn.empty() && qn.isSeq() && Q_.rows() == state_dim) {
        for (int i = 0; i < state_dim && i < static_cast<int>(qn.size()); ++i) {
            double v = Q_(i, i);
            qn[i] >> v;  // 对角过程噪声，缺项保持 lovod 默认
            Q_(i, i) = v;
        }
    }
    const cv::FileNode rn = n["r"];
    if (!rn.empty() && rn.isSeq() && R_.rows() == 4) {
        for (int i = 0; i < 4 && i < static_cast<int>(rn.size()); ++i) {
            double v = R_(i, i);
            rn[i] >> v;
            R_(i, i) = v;
        }
    }
    if (ekf_)
        ekf_->setNoise(Q_, R_);
    return true;
}

/**
 * 作用：清空跟踪与平滑。输入：无。输出：Lost。
 */
void VehicleTracker::reset() {
    state_ = VehicleTrackState::Lost;
    tracked_id_ = 0;
    detect_count_ = 0;
    lost_count_ = 0;
    last_nis_ = 0.0;
    bootstrap_running_ = false;
    for (BootstrapTrack& t : bootstrap_tracks_)
        t = BootstrapTrack();
    resetOutpostState();
    accelerate_window_.reset();
    resetZAccumulator();
    resetCenterYawObserver();
    stop_evidence_armed_ = false;
    vyaw_recovery_pending_ = false;
    resetVyawSmooth();
    smooth_pos_.setConstant(std::numeric_limits<double>::quiet_NaN());
    outpost_smooth_pos_.setConstant(std::numeric_limits<double>::quiet_NaN());
    enable_smooth_x_ = false;
    enable_smooth_y_ = false;
    enable_outpost_smooth_x_ = false;
    enable_outpost_smooth_y_ = false;
    published_xc_ = 0.0;
    published_yc_ = 0.0;
    published_vyaw_ = 0.0;
    last_phase_change_ = false;
}

/**
 * 作用：本帧可见板的 active 状态维：车心 xy、yaw、速度，外加本对 z/r。
 * 输入：相位。输出：下标列表。
 */
std::vector<int> VehicleTracker::getActiveIndices(int phase_idx) const {
    const bool pair_0 = isPair0Phase(phase_idx);
    std::vector<int> active_indices{xc, yc, yaw, vx, vy, vz, vyaw};
    active_indices.push_back(pair_0 ? z1 : z2);
    active_indices.push_back(pair_0 ? r1 : r2);
    return active_indices;
}

/**
 * 作用：把非本对 z/r 的交叉协方差清掉，只留对角方差。
 * 输入：P、相位。输出：就地改 P。
 */
void VehicleTracker::decoupleInactiveCovariance(Eigen::MatrixXd& covariance, int phase_idx) const {
    const bool pair_0 = isPair0Phase(phase_idx);
    const int inactive_z = pair_0 ? z2 : z1;
    const int inactive_r = pair_0 ? r2 : r1;
    const double z_var = covariance(inactive_z, inactive_z);
    const double r_var = covariance(inactive_r, inactive_r);
    covariance.row(inactive_z).setZero();
    covariance.col(inactive_z).setZero();
    covariance(inactive_z, inactive_z) = z_var;
    covariance.row(inactive_r).setZero();
    covariance.col(inactive_r).setZero();
    covariance(inactive_r, inactive_r) = r_var;
}

/**
 * 作用：新息缩放。输入：无（读 last_nis_ / 状态机）。输出：Q 倍率。
 */
double VehicleTracker::processQScale() const {
    if (state_ == VehicleTrackState::TempLost)
        return params_.q_adapt_max;
    const double innov = last_nis_;
    if (innov > 1e-9 && params_.q_adapt_ref > 1e-6)
        return std::clamp(innov / params_.q_adapt_ref, params_.q_adapt_min, params_.q_adapt_max);
    return 1.0;
}

/**
 * 作用：匀加速白噪声 Q。xc/yc/z 配 q_acc，yaw 配 q_alpha；r 只给很小对角，避免半径乱跳。
 * 输入：dt。输出：11×11。
 */
Eigen::MatrixXd VehicleTracker::makeProcessQ(double dt) const {
    const double dt_c = std::clamp(dt, 1e-3, 0.1);
    const double q_scale = processQScale();
    const double v1 = params_.q_acc * q_scale;
    const double v2 = params_.q_alpha * q_scale;
    const double a = dt_c * dt_c * dt_c * dt_c / 4.0;
    const double b = dt_c * dt_c * dt_c / 2.0;
    const double c = dt_c * dt_c;
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    auto add_pv = [&](int pos, int vel, double v) {
        Q(pos, pos) += a * v;
        Q(pos, vel) += b * v;
        Q(vel, pos) += b * v;
        Q(vel, vel) += c * v;
    };
    add_pv(xc, vx, v1);
    add_pv(yc, vy, v1);
    add_pv(z1, vz, v1);
    add_pv(z2, vz, v1);
    add_pv(yaw, vyaw, v2);
    Q(r1, r1) = 1e-4 * dt_c * q_scale;
    Q(r2, r2) = 1e-4 * dt_c * q_scale;
    return Q;
}

/**
 * 作用：P 收死后增益≈0，四板不再动；每帧只抬速度维。输入：11×11 P。输出：就地改。
 */
void VehicleTracker::fadeVelocityCov(Eigen::MatrixXd& P) const {
    const double fade = params_.vel_fade;
    if (fade <= 1.0 || P.rows() < state_dim || P.cols() < state_dim)
        return;
    const int idx[4] = {vx, vy, vz, vyaw};
    const double s = std::sqrt(fade);
    for (int k = 0; k < 4; ++k) {
        const int i = idx[k];
        for (int j = 0; j < P.cols(); ++j) {
            if (j == i)
                continue;
            P(i, j) *= s;
            P(j, i) *= s;
        }
        P(i, i) *= fade;
    }
    P = 0.5 * (P + P.transpose());
}

/**
 * 作用：只挡飞点。门限×8 以内的新息必须全权更新，否则机动时跟踪会离开观测。
 * 输入：evaluation、χ² 门限。输出：可能改 ev.R。
 */
void VehicleTracker::inflateOutlierR(EKFCore::MeasurementEvaluation& ev, double gate) {
    const double hard = gate * 8.0;
    if (!std::isfinite(ev.nis) || hard < 1e-6 || ev.nis <= hard)
        return;
    ev.R *= (ev.nis / hard);
}

/**
 * 作用：P 对角加下限，避免收死后增益≈0。输入：P。输出：就地改。
 */
void VehicleTracker::floorStateCov(Eigen::MatrixXd& P) const {
    constexpr double pos_floor = 4e-4;   // σ≥2cm，板心每帧还能把车心拉开
    constexpr double yaw_floor = 4e-4;   // σ≥1.1°
    constexpr double vel_floor = 0.25;
    constexpr double wyaw_floor = 0.25;
    auto lift = [&P](int i, double lo) {
        if (i >= 0 && i < P.rows() && i < P.cols())
            P(i, i) = std::max(P(i, i), lo);
    };
    if (P.rows() == o_state_dim && P.cols() == o_state_dim) {
        lift(o_xc, pos_floor);
        lift(o_yc, pos_floor);
        lift(o_zc, pos_floor);
        lift(o_yaw, yaw_floor);
        lift(o_vyaw, wyaw_floor);
        P = 0.5 * (P + P.transpose());
        return;
    }
    if (P.rows() < state_dim || P.cols() < state_dim)
        return;
    lift(xc, pos_floor);
    lift(yc, pos_floor);
    lift(z1, pos_floor);
    lift(z2, pos_floor);
    lift(yaw, yaw_floor);
    lift(vx, vel_floor);
    lift(vy, vel_floor);
    lift(vz, vel_floor);
    lift(vyaw, wyaw_floor);
    P = 0.5 * (P + P.transpose());
}

/**
 * 作用：普通 11 维过程/量测。过程只积当前相位那对 z；量测是车心−r(cosθ,sinθ)。
 * 输入：active_phase_idx 必须是长期存活的引用（confirmed_phase_idx_ 或 bootstrap 的 phase_idx）。
 * 输出：新 EKF 核。
 */
std::unique_ptr<EKFCore> VehicleTracker::makeNormalEKF(const int& active_phase_idx) const {
    auto q_normal = [this](double dt) {
        return makeProcessQ(dt);  // 白噪声×新息缩放；相机系不能用 lovod 的 Q×dt
    };
    auto f = [this, &active_phase_idx](const Eigen::VectorXd& x, double dt) {
        Eigen::VectorXd x_new = x;
        const bool pair_0 = isPair0Phase(active_phase_idx);
        x_new(xc) += x(vx) * dt;
        x_new(yc) += x(vy) * dt;
        if (pair_0)
            x_new(z1) += x(vz) * dt;
        else
            x_new(z2) += x(vz) * dt;
        x_new(yaw) += x(vyaw) * dt;
        return x_new;
    };
    auto j_f = [this, &active_phase_idx](const Eigen::VectorXd& x, double dt) {
        (void)x;
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(state_dim, state_dim);
        const bool pair_0 = isPair0Phase(active_phase_idx);
        F(xc, vx) = dt;
        F(yc, vy) = dt;
        F(pair_0 ? z1 : z2, vz) = dt;
        F(yaw, vyaw) = dt;
        return F;
    };
    auto h = [this](const Eigen::VectorXd& x, const int phase_idx) {
        return measureArmor(x, phase_idx);
    };
    auto j_h = [this](const Eigen::VectorXd& x, const int phase_idx) {
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(4, state_dim);
        const int n = static_cast<int>(phase_offsets_.size());
        const int idx = (phase_idx >= 0 && phase_idx < n) ? phase_idx : 0;
        const bool pair_0 = (idx == 0 || idx == 2);
        const double theta = x(yaw) + phase_offsets_[static_cast<std::size_t>(idx)];
        const int z_idx = pair_0 ? z1 : z2;
        const int r_idx = pair_0 ? r1 : r2;
        const double r = x(r_idx);
        H(0, xc) = 1.0;
        H(1, yc) = 1.0;
        H(2, z_idx) = 1.0;
        H(3, yaw) = 1.0;
        H(0, yaw) = r * std::sin(theta);
        H(1, yaw) = -r * std::cos(theta);
        H(0, r_idx) = -std::cos(theta);
        H(1, r_idx) = -std::sin(theta);
        return H;
    };
    return std::make_unique<EKFCore>(q_normal, f, h, j_f, j_h, Q_, R_);
}

/**
 * 作用：写入普通目标 Q/R 并重建核。输入：无。输出：model=Normal。
 */
void VehicleTracker::buildNormalModel() {
    model_ = Model::Normal;
    // lovod processor_config.yaml ekf.q / ekf.r
    const std::vector<double> q_diag = {0.01, 0.01, 0.005, 0.005, 0.03, 0.01, 0.01, 0.005, 0.04, 0.001,
                                        0.001};
    Q_ = Eigen::MatrixXd::Identity(state_dim, state_dim);
    for (std::size_t i = 0; i < q_diag.size(); ++i)
        Q_(static_cast<int>(i), static_cast<int>(i)) = q_diag[i];
    // lovod odom 用 σ_xy≈5cm。相机系板心来自 PnP，σ 取 ~1cm 才能贴观测；yaml 若有 r 序列仍覆盖。
    const std::vector<double> r_diag = {1e-4, 1e-4, 4e-4, 2.5e-3};
    R_ = Eigen::MatrixXd::Identity(4, 4);
    for (std::size_t i = 0; i < r_diag.size(); ++i)
        R_(static_cast<int>(i), static_cast<int>(i)) = r_diag[i];
    ekf_ = makeNormalEKF(confirmed_phase_idx_);
}

/**
 * 作用：前哨 6 维：车心几乎不动，只积 yaw；三板高度由层号×z_step。
 * 输入：无。输出：model=Outpost。
 */
void VehicleTracker::buildOutpostModel() {
    model_ = Model::Outpost;
    auto q_outpost = [this](double dt) {
        const double scale_dt = std::clamp(dt, 1e-3, 0.1);
        return Q_outpost_ * scale_dt;
    };
    auto f = [](const Eigen::VectorXd& x, double dt) {
        Eigen::VectorXd x_new = x;
        x_new(o_yaw) += x(o_vyaw) * dt;
        return x_new;
    };
    auto j_f = [](const Eigen::VectorXd& x, double dt) {
        (void)x;
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(o_state_dim, o_state_dim);
        F(o_yaw, o_vyaw) = dt;
        return F;
    };
    auto h = [this](const Eigen::VectorXd& x, const int phase_idx) {
        return measureArmor(x, phase_idx);
    };
    auto j_h = [this](const Eigen::VectorXd& x, const int phase_idx) {
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(4, o_state_dim);
        const int n = static_cast<int>(phase_offsets_.size());
        const int idx = (phase_idx >= 0 && phase_idx < n) ? phase_idx : 0;
        const double theta = x(o_yaw) + phase_offsets_[static_cast<std::size_t>(idx)];
        const double r = x(o_r);
        H(0, o_xc) = 1.0;
        H(1, o_yc) = 1.0;
        H(2, o_zc) = 1.0;
        H(3, o_yaw) = 1.0;
        H(0, o_yaw) = r * std::sin(theta);
        H(1, o_yaw) = -r * std::cos(theta);
        H(0, o_r) = -std::cos(theta);
        H(1, o_r) = -std::sin(theta);
        return H;
    };
    const std::vector<double> q_diag = {0.001, 0.001, 0.001, 0.03, 0.04, 0.001};
    Q_outpost_ = Eigen::MatrixXd::Identity(o_state_dim, o_state_dim);
    for (std::size_t i = 0; i < q_diag.size(); ++i)
        Q_outpost_(static_cast<int>(i), static_cast<int>(i)) = q_diag[i];
    // 前哨也要跟板心；lovod 0.8 是 odom 抖动，相机系会把三板从 PnP 上拉开。
    const std::vector<double> r_diag = {1e-3, 1e-3, 4e-3, 1e-2};
    R_outpost_ = Eigen::MatrixXd::Identity(4, 4);
    for (std::size_t i = 0; i < r_diag.size(); ++i)
        R_outpost_(static_cast<int>(i), static_cast<int>(i)) = r_diag[i];
    ekf_ = std::make_unique<EKFCore>(q_outpost, f, h, j_f, j_h, Q_outpost_, R_outpost_);
}

/**
 * 作用：跟踪系预测板位姿。普通车按相位选 (z,r) 对；前哨加层高。
 * 输入：跟踪系 x、相位。输出：(x,y,z,yaw)。
 */
Eigen::Vector4d VehicleTracker::measureArmor(const Eigen::VectorXd& x, int phase_idx) const {
    Eigen::Vector4d z = Eigen::Vector4d::Zero();
    const int n = std::max(1, static_cast<int>(phase_offsets_.size()));
    const int idx = ((phase_idx % n) + n) % n;
    const double off = phase_offsets_[static_cast<std::size_t>(idx)];
    if (x.size() == o_state_dim) {
        const double theta = x(o_yaw) + off;
        const double r = x(o_r);
        z(0) = x(o_xc) - r * std::cos(theta);
        z(1) = x(o_yc) - r * std::sin(theta);
        const int outpost_id =
            (idx >= 0 && idx < 3) ? outpost_state_.slot_ids[static_cast<std::size_t>(idx)] : 0;
        const double z_offset =
            (outpost_id >= 1 && outpost_id <= 3) ? outpost_z_step_ * (outpost_id - 2) : 0.0;
        z(2) = x(o_zc) + z_offset;
        z(3) = theta;
        return z;
    }
    const bool pair_0 = (idx == 0 || idx == 2);
    const double theta = x(yaw) + off;
    const double r = pair_0 ? x(r1) : x(r2);
    const double z_pair = pair_0 ? x(z1) : x(z2);
    z(0) = x(xc) - r * std::cos(theta);
    z(1) = x(yc) - r * std::sin(theta);
    z(2) = z_pair;
    z(3) = theta;
    return z;
}

/**
 * 作用：锁车。前哨走 6 维；普通车先试 IPPE 双根 bootstrap。
 * 输入：选中的装甲指针。输出：DETECTING。
 */
void VehicleTracker::init(TrackedArmor* armor) {
    if (armor == nullptr)
        return;
    tracked_armor_ = *armor;
    tracked_id_ = armor->id;
    tracked_type_ = armor->type;
    last_time_ = armor->stamp;
    last_yaw_ = 0.0;
    armor_num_ = (tracked_id_ == 6) ? 3 : 4;
    default_r_ = (tracked_id_ == 6) ? outpost_radius_ : params_.default_radius_m;
    phase_offsets_ = getPhaseOffsets();
    accelerate_window_.reset();
    resetZAccumulator();
    resetCenterYawObserver();
    resetOutpostState();
    stop_evidence_armed_ = false;
    state_ = VehicleTrackState::Detecting;
    detect_count_ = 0;
    lost_count_ = 0;
    confirmed_phase_idx_ = 0;
    hypothetic_phase_idx_ = -1;
    hypothetic_phase_count_ = 0;
    last_phase_change_time_ = armor->stamp;
    last_nis_ = 0.0;
    if (tracked_id_ == 6) {
        buildOutpostModel();
        initEKF(tracked_armor_);
    } else {
        buildNormalModel();
        if (startNormalBootstrap(tracked_armor_)) {
            // 双根尚未决胜时也用 0 号假设的车心给 overlay，避免 Detecting 画在原点。
            if (bootstrap_tracks_[0].ekf)
                target_state_ = bootstrap_tracks_[0].ekf->getState();
            if (target_state_.x.size() >= state_dim) {
                published_xc_ = target_state_.x(xc);
                published_yc_ = target_state_.x(yc);
                published_vyaw_ = target_state_.x(vyaw);
            }
            return;
        }
        initEKF(tracked_armor_);
    }
    last_phase_change_yaw_ = isOutpostTarget() ? target_state_.x(o_yaw) : target_state_.x(yaw);
    if (isOutpostTarget() && target_state_.x.size() >= o_state_dim) {
        published_xc_ = target_state_.x(o_xc);
        published_yc_ = target_state_.x(o_yc);
        published_vyaw_ = target_state_.x(o_vyaw);
    } else if (target_state_.x.size() >= state_dim) {
        published_xc_ = target_state_.x(xc);
        published_yc_ = target_state_.x(yc);
        published_vyaw_ = target_state_.x(vyaw);
    }
}

/**
 * 作用：把一根位姿写成 (xyz,yaw) 和视觉协方差。yaw 相对参考角展开，避免 ±π 跳。
 * 输入：root、yaw_reference。输出：measurement、visual_covariance；非法 false。
 */
bool VehicleTracker::buildRootMeasurement(const PoseRoot& root, double yaw_reference,
                                          Eigen::Vector4d& measurement,
                                          Eigen::Matrix4d& visual_covariance) const {
    const Eigen::Vector3d& p = root.position;
    if (!p.allFinite())
        return false;
    const double wrapped_yaw = wrapPi(root.yaw);
    const double yaw = yaw_reference + shortestAngularDistance(yaw_reference, wrapped_yaw);
    measurement << p.x(), p.y(), p.z(), yaw;
    visual_covariance = root.visual_covariance;
    for (int idx = 0; idx < 4; ++idx)
        visual_covariance(idx, idx) = std::max(visual_covariance(idx, idx), root_min_variance_);
    return measurement.allFinite() && visual_covariance.allFinite();
}

/**
 * 作用：在该板所有根×相位里挑 NLL 最小的。滞后时 NIS 大也保留，否则观测进不了更新。
 * 输入：armor（会写 selected_root_idx）。输出：candidate；找不到 false。
 */
bool VehicleTracker::selectNormalRoot(TrackedArmor& armor, RootPhaseCandidate& selected_candidate) const {
    if (!ekf_ || armor.pose_roots.empty())
        return false;
    bool found = false;
    for (uint8_t idx = 0; idx < armor.pose_roots.size(); ++idx) {
        Eigen::Vector4d measurement;
        Eigen::Matrix4d visual_covariance;
        if (!buildRootMeasurement(armor.pose_roots[idx], target_state_.x(yaw), measurement,
                                  visual_covariance))
            continue;
        visual_covariance(3, 3) += normal_yaw_extra_variance_;
        RootPhaseCandidate root_candidate;
        root_candidate.root_idx = idx;
        root_candidate.measurement = measurement;
        root_candidate.measurement_noise = Eigen::Matrix4d(R_) + visual_covariance;
        root_candidate.phase_match = ekf_->selectBestPhase(root_candidate.measurement, phase_offsets_,
                                                           confirmed_phase_idx_,
                                                           root_candidate.measurement_noise);
        const auto evaluation =
            ekf_->evaluateMeasurement(root_candidate.measurement, target_state_.x,
                                      root_candidate.phase_match.phase_idx,
                                      root_candidate.measurement_noise);
        if (!std::isfinite(root_candidate.phase_match.nll) || !std::isfinite(evaluation.nis))
            continue;  // 只丢数值坏点；滞后时 NIS 大也要关联
        if (!found || root_candidate.phase_match.nll < selected_candidate.phase_match.nll) {
            selected_candidate = root_candidate;
            found = true;
        }
    }
    if (found)
        armor.selected_root_idx = selected_candidate.root_idx;
    return found;
}

/**
 * 作用：前哨选根；方向锁定后只允许当前相或下一相。
 * 输入：armor。输出：candidate。
 */
bool VehicleTracker::selectOutpostRoot(const TrackedArmor& armor,
                                       RootPhaseCandidate& selected_candidate) const {
    if (!ekf_ || armor.pose_roots.empty())
        return false;
    const int current_phase = outpost_state_.phase_slot;
    const bool direction_locked = (outpost_state_.direction != 0);
    const int candidate_phase =
        direction_locked && isOutpostConfirmed() ? nextOutpostPhase(current_phase) : current_phase;
    const std::size_t root_count = direction_locked ? armor.pose_roots.size() : 1;
    bool found = false;
    for (std::size_t root_idx = 0; root_idx < root_count; ++root_idx) {
        RootPhaseCandidate root_candidate;
        root_candidate.root_idx = static_cast<uint8_t>(root_idx);
        Eigen::Matrix4d visual_covariance;
        if (!buildRootMeasurement(armor.pose_roots[root_idx], target_state_.x(o_yaw),
                                  root_candidate.measurement, visual_covariance))
            continue;
        root_candidate.measurement_noise = R_outpost_ + visual_covariance;
        root_candidate.phase_match =
            ekf_->selectBestPhase(root_candidate.measurement, phase_offsets_, current_phase,
                                  root_candidate.measurement_noise, candidate_phase);
        const auto evaluation =
            ekf_->evaluateMeasurement(root_candidate.measurement, target_state_.x,
                                      root_candidate.phase_match.phase_idx,
                                      root_candidate.measurement_noise);
        if (!std::isfinite(root_candidate.phase_match.nll) || !std::isfinite(evaluation.nis))
            continue;  // 前哨同样不因 NIS 丢掉唯一板
        if (!found || root_candidate.phase_match.nll < selected_candidate.phase_match.nll) {
            selected_candidate = root_candidate;
            found = true;
        }
    }
    return found;
}

/**
 * 作用：同 id 最多两块板，各挑 NLL 最小相位；近相优先当 primary。
 * 输入：本帧指针。输出：association。滞后时 NIS 大也关联，否则双板进不了更新。
 */
bool VehicleTracker::associateNormalArmors(const std::vector<TrackedArmor*>& armors,
                                           ArmorAssociation& association) {
    association = ArmorAssociation();
    if (isOutpostTarget() || !ekf_)
        return false;
    std::vector<ArmorAssociation::Entry> selected_entries;
    selected_entries.reserve(2);
    for (TrackedArmor* armor : armors) {
        if (armor == nullptr || armor->id != tracked_id_ || armor->pose_roots.empty())
            continue;
        bool found_entry = false;
        ArmorAssociation::Entry best_entry;
        uint8_t best_root_idx = 0;
        for (uint8_t root_idx = 0; root_idx < armor->pose_roots.size(); ++root_idx) {
            Eigen::Vector4d z;
            Eigen::Matrix4d visual_covariance;
            if (!buildRootMeasurement(armor->pose_roots[root_idx], target_state_.x(yaw), z,
                                      visual_covariance))
                continue;
            visual_covariance(3, 3) += normal_yaw_extra_variance_;
            const Eigen::Matrix4d R = R_ + visual_covariance;
            for (int phase_idx = 0; phase_idx < 4; ++phase_idx) {
                ArmorAssociation::Entry candidate;
                candidate.armor = armor;
                candidate.z = z;
                candidate.R = R;
                candidate.evaluation =
                    ekf_->evaluateMeasurement(candidate.z, target_state_.x, phase_idx, candidate.R);
                if (!std::isfinite(candidate.evaluation.nll) || !std::isfinite(candidate.evaluation.nis))
                    continue;  // 只丢数值坏点；滞后时 NIS 大也要关联
                const int phase_diff = std::abs(phase_idx - confirmed_phase_idx_);
                candidate.phase_match.phase_idx = phase_idx;
                candidate.phase_match.phase_distance = std::min(phase_diff, 4 - phase_diff);
                candidate.phase_match.change = (candidate.phase_match.phase_distance == 1);
                candidate.phase_match.nll = candidate.evaluation.nll;
                if (!found_entry || candidate.phase_match.nll < best_entry.phase_match.nll) {
                    found_entry = true;
                    best_entry = candidate;
                    best_root_idx = root_idx;
                }
            }
        }
        if (!found_entry)
            continue;
        armor->selected_root_idx = best_root_idx;
        selected_entries.push_back(best_entry);
    }
    if (selected_entries.empty())
        return false;
    if (selected_entries.size() > 1) {
        std::sort(selected_entries.begin(), selected_entries.end(),
                  [](const ArmorAssociation::Entry& a, const ArmorAssociation::Entry& b) {
                      if (a.phase_match.phase_distance != b.phase_match.phase_distance)
                          return a.phase_match.phase_distance < b.phase_match.phase_distance;
                      return a.phase_match.nll < b.phase_match.nll;
                  });
        association.primary = selected_entries.front();
        association.support = selected_entries.back();
        association.valid = true;
        association.has_support = true;
    } else {
        association.primary = selected_entries.front();
        association.valid = true;
    }
    return true;
}

/**
 * 作用：车心 = 板心 + r·(cos yaw, sin yaw)，两对 z/r 先用同一高度和半径。
 * 输入：armor。输出：ekf_ 初值。
 */
void VehicleTracker::initEKF(const TrackedArmor& armor) {
    const Eigen::Vector3d position = armor.selectedRoot().position;
    const double xa = position.x();
    const double ya = position.y();
    const double za = position.z();
    const double yaw_meas = getYawFromOrientation(armor);
    const double r = (armor.id == 8) ? 0.0 : default_r_;
    dz_ = 0.0;
    za_ = za;
    if (isOutpostTarget()) {
        EKFCore::State initial_state(o_state_dim);
        const double xc0 = xa + r * std::cos(yaw_meas);
        const double yc0 = ya + r * std::sin(yaw_meas);
        initial_state.x << xc0, yc0, za, yaw_meas, 0.0, r;
        initial_state.P = Eigen::MatrixXd::Identity(o_state_dim, o_state_dim) * 0.1;
        initial_state.P(o_r, o_r) = 1e-6;
        ekf_->init(initial_state);
        target_state_ = initial_state;
    } else {
        EKFCore::State initial_state(state_dim);
        const double xc0 = xa + r * std::cos(yaw_meas);
        const double yc0 = ya + r * std::sin(yaw_meas);
        initial_state.x << xc0, yc0, za, za, yaw_meas, 0, 0, 0, 0, r, r;
        // 位置/朝向 P 要大，首帧增益才够把四板贴到观测上；v/ω 也要大，否则永远从 0 爬。
        Eigen::VectorXd p0(state_dim);
        p0 << 1.0, 1.0, 1.0, 1.0, 0.4, 64.0, 64.0, 64.0, 100.0, 1.0, 1.0;
        initial_state.P = p0.asDiagonal();
        ekf_->init(initial_state);
        target_state_ = initial_state;
    }
}

/**
 * 作用：两根 IPPE 各开一条 EKF，后续用累积 NLL 决胜。单根则 false。
 * 输入：armor。输出：bootstrap_running_。
 */
bool VehicleTracker::startNormalBootstrap(const TrackedArmor& armor) {
    bootstrap_running_ = false;
    for (BootstrapTrack& track : bootstrap_tracks_)
        track = BootstrapTrack();
    if (armor.pose_roots.size() != 2)
        return false;
    for (uint8_t idx = 0; idx < 2; ++idx) {
        BootstrapTrack& track = bootstrap_tracks_[idx];
        track.seed_root_idx = idx;
        track.phase_idx = 0;
        track.last_stamp = armor.stamp;
        track.last_armor = armor;
        track.last_armor.selected_root_idx = idx;
        track.ekf = makeNormalEKF(track.phase_idx);
        Eigen::Vector4d measurement;
        Eigen::Matrix4d visual_covariance;
        if (!buildRootMeasurement(armor.pose_roots[idx], 0.0, measurement, visual_covariance)) {
            for (BootstrapTrack& reset_track : bootstrap_tracks_)
                reset_track = BootstrapTrack();
            return false;
        }
        const double r = default_r_;
        EKFCore::State initial_state(state_dim);
        initial_state.x << measurement(0) + r * std::cos(measurement(3)),
            measurement(1) + r * std::sin(measurement(3)), measurement(2), measurement(2),
            measurement(3), 0.0, 0.0, 0.0, 0.0, r, r;
        initial_state.P = Eigen::MatrixXd::Identity(state_dim, state_dim) * 0.25;
        track.ekf->init(initial_state);
        track.support = 1;
        track.accumulated_nll = 0.0;
    }
    bootstrap_running_ = true;
    return true;
}

/**
 * 作用：两根假设各预测+选相更新，NLL 差够大或帧数够了就提交赢家，半径重置为默认。
 * 输入：本帧装甲。输出：true 表示 bootstrap 结束并写入主 EKF。
 */
bool VehicleTracker::advanceBootstrap(const TrackedArmor& armor) {
    if (!bootstrap_running_ || armor.id != tracked_id_ || armor.pose_roots.empty())
        return false;
    for (BootstrapTrack& track : bootstrap_tracks_) {
        if (!track.ekf)
            continue;
        const double dt = std::clamp(armor.stamp - track.last_stamp, 1e-3, 0.1);
        track.ekf->predict(dt);
        track.last_stamp = armor.stamp;
        RootPhaseCandidate best_candidate;
        EKFCore::MeasurementEvaluation best_evaluation;
        bool found = false;
        const Eigen::VectorXd& pred_state = track.ekf->getState().x;
        for (uint8_t idx = 0; idx < armor.pose_roots.size(); ++idx) {
            Eigen::Vector4d measurement;
            Eigen::Matrix4d visual_covariance;
            if (!buildRootMeasurement(armor.pose_roots[idx], pred_state(yaw), measurement,
                                      visual_covariance))
                continue;
            RootPhaseCandidate candidate;
            candidate.root_idx = idx;
            candidate.measurement = measurement;
            candidate.measurement_noise = Eigen::Matrix4d(R_) + visual_covariance;
            candidate.phase_match = track.ekf->selectBestPhase(
                candidate.measurement, phase_offsets_, track.phase_idx, candidate.measurement_noise);
            const auto evaluation = track.ekf->evaluateMeasurement(
                candidate.measurement, pred_state, candidate.phase_match.phase_idx,
                candidate.measurement_noise);
            if (!std::isfinite(evaluation.nll))
                continue;
            if (!found || evaluation.nll < best_evaluation.nll) {
                best_candidate = candidate;
                best_evaluation = evaluation;
                found = true;
            }
        }
        if (!found) {
            track.accumulated_nll += bootstrap_nll_margin_;
            continue;
        }
        track.phase_idx = best_candidate.phase_match.phase_idx;
        track.ekf->update(best_evaluation);
        track.last_armor = armor;
        track.last_armor.selected_root_idx = best_candidate.root_idx;
        track.accumulated_nll += best_evaluation.nll;
        ++track.support;
    }
    std::size_t winner_idx = 0;
    std::size_t loser_idx = 1;
    if (bootstrap_tracks_[1].accumulated_nll < bootstrap_tracks_[0].accumulated_nll) {
        winner_idx = 1;
        loser_idx = 0;
    }
    BootstrapTrack& winner = bootstrap_tracks_[winner_idx];
    BootstrapTrack& loser = bootstrap_tracks_[loser_idx];
    const double nll_margin = loser.accumulated_nll - winner.accumulated_nll;
    const bool enough_support = winner.support >= bootstrap_confirm_frames_;
    const bool clearly_separated = nll_margin >= bootstrap_nll_margin_;
    if (!enough_support && !clearly_separated)
        return false;
    EKFCore::State committed_state = winner.ekf->getState();
    int committed_phase_idx = winner.phase_idx;
    // 半径重置后方差要重新打开；位置/速度用与 initEKF 同量级，否则提交瞬间四板又粘住。
    const auto reset_bootstrap_uncertainty = [&committed_state](int state_idx, double var) {
        committed_state.P.row(state_idx).setZero();
        committed_state.P.col(state_idx).setZero();
        committed_state.P(state_idx, state_idx) = var;
    };
    committed_state.x(r1) = default_r_;
    committed_state.x(r2) = default_r_;
    reset_bootstrap_uncertainty(xc, 1.0);
    reset_bootstrap_uncertainty(yc, 1.0);
    reset_bootstrap_uncertainty(yaw, 0.4);
    reset_bootstrap_uncertainty(vx, 64.0);
    reset_bootstrap_uncertainty(vy, 64.0);
    reset_bootstrap_uncertainty(vyaw, 100.0);
    reset_bootstrap_uncertainty(r1, 1.0);
    reset_bootstrap_uncertainty(r2, 1.0);
    if (enable_vyaw_accelerate_)
        committed_state.P(vyaw, vyaw) = 100.0;
    if (committed_phase_idx != 0 && committed_phase_idx < static_cast<int>(phase_offsets_.size())) {
        committed_state.x(yaw) += phase_offsets_[static_cast<std::size_t>(committed_phase_idx)];
        if (!isPair0Phase(committed_phase_idx)) {
            std::swap(committed_state.x(z1), committed_state.x(z2));
            std::swap(committed_state.x(r1), committed_state.x(r2));
            Eigen::RowVectorXd row = committed_state.P.row(z1);
            committed_state.P.row(z1) = committed_state.P.row(z2);
            committed_state.P.row(z2) = row;
            Eigen::VectorXd col = committed_state.P.col(z1);
            committed_state.P.col(z1) = committed_state.P.col(z2);
            committed_state.P.col(z2) = col;
            row = committed_state.P.row(r1);
            committed_state.P.row(r1) = committed_state.P.row(r2);
            committed_state.P.row(r2) = row;
            col = committed_state.P.col(r1);
            committed_state.P.col(r1) = committed_state.P.col(r2);
            committed_state.P.col(r2) = col;
        }
    }
    confirmed_phase_idx_ = 0;
    ekf_ = makeNormalEKF(confirmed_phase_idx_);
    ekf_->init(committed_state);
    target_state_ = committed_state;
    tracked_armor_ = winner.last_armor;
    last_time_ = winner.last_stamp;
    last_phase_change_time_ = winner.last_stamp;
    last_phase_change_yaw_ = committed_state.x(yaw);
    bootstrap_running_ = false;
    for (BootstrapTrack& track : bootstrap_tracks_)
        track = BootstrapTrack();
    return true;
}

void VehicleTracker::pushWindowSample(AccelerateWindow& window, double stamp,
                                      const Eigen::Vector2d& meas_center,
                                      const Eigen::Vector2d& predict_center, double image_area) {
    AccelerateWindow::Sample sample;
    sample.stamp = stamp;
    sample.meas_center = meas_center;
    sample.predict_center = predict_center;
    sample.image_area = image_area;
    window.samples.push_back(sample);
    while (!window.samples.empty() &&
           (stamp - window.samples.front().stamp) > window.time_threshold + 0.12)
        window.samples.pop_front();
}

bool VehicleTracker::estimateWindowVelocity(double& vx_meas, double& vy_meas, double& vx_pred,
                                            double& vy_pred, double& area_ratio) const {
    vx_meas = vy_meas = vx_pred = vy_pred = 0.0;
    area_ratio = 1.0;
    if (static_cast<int>(accelerate_window_.samples.size()) < accelerate_window_.min_samples)
        return false;
    const auto& first = accelerate_window_.samples.front();
    const auto& last = accelerate_window_.samples.back();
    const double acc_dt = last.stamp - first.stamp;
    if (acc_dt < accelerate_window_.time_threshold)
        return false;
    // 窗口内对 t 的指数加权斜率：越新的样本权越大，给 vx/vy 加速收敛用。
    auto slope = [&](bool is_measured, int axis) {
        const auto& samples = accelerate_window_.samples;
        if (samples.size() < 2)
            return 0.0;
        constexpr double tau = 0.15;
        double w_sum = 0.0;
        double t_sum = 0.0;
        double v_sum = 0.0;
        for (const auto& sample : samples) {
            const double t = sample.stamp - first.stamp;
            const double age = std::max(last.stamp - sample.stamp, 0.0);
            const double w = std::exp(-age / tau);
            const double v = is_measured ? sample.meas_center[axis] : sample.predict_center[axis];
            w_sum += w;
            t_sum += w * t;
            v_sum += w * v;
        }
        if (w_sum < 1e-6)
            return 0.0;
        const double t_mean = t_sum / w_sum;
        const double v_mean = v_sum / w_sum;
        double num = 0.0;
        double den = 0.0;
        for (const auto& sample : samples) {
            const double t = sample.stamp - first.stamp;
            const double age = std::max(last.stamp - sample.stamp, 0.0);
            const double w = std::exp(-age / tau);
            const double v = is_measured ? sample.meas_center[axis] : sample.predict_center[axis];
            num += w * (t - t_mean) * (v - v_mean);
            den += w * (t - t_mean) * (t - t_mean);
        }
        if (den < 1e-6)
            return 0.0;
        return num / den;
    };
    vx_meas = slope(true, 0);
    vy_meas = slope(true, 1);
    vx_pred = slope(false, 0);
    vy_pred = slope(false, 1);
    const double start_area = std::max(first.image_area, 1.0);
    const double end_area = std::max(last.image_area, 1.0);
    area_ratio = end_area / start_area;
    return true;
}

bool VehicleTracker::allowVyInjection(double vy_meas, double vy_pred, double area_ratio) const {
    if (!enable_vy_accelerate_)
        return false;
    if (area_ratio < accelerate_window_.min_vy_area_ratio ||
        area_ratio > accelerate_window_.max_vy_area_ratio)
        return false;
    if (std::abs(vy_meas) < 0.01 && std::abs(vy_pred) < 0.01)
        return false;
    return true;
}

/**
 * 作用：用窗口最小二乘速度往 EKF 的 vx/vy 掺一点，自旋不太快时才做。
 * 输入：板、量测、时刻。输出：vx_injected/vy_injected。
 */
void VehicleTracker::accelerateConvergence(const TrackedArmor& armor, const Eigen::Vector4d& measurement,
                                           double stamp, bool& vx_injected, bool& vy_injected) {
    const double armor_yaw = measurement(3);
    const double center_yaw = armor_yaw - phase_offsets_[static_cast<std::size_t>(confirmed_phase_idx_)];
    const double r = isPair0Phase(confirmed_phase_idx_) ? target_state_.x(r1) : target_state_.x(r2);
    const double xc_meas = measurement(0) + r * std::cos(armor_yaw);
    const double yc_meas = measurement(1) + r * std::sin(armor_yaw);
    const Eigen::Vector2d meas_center(xc_meas, yc_meas);
    const double yaw_error = shortestAngularDistance(center_yaw, target_state_.x(yaw));
    const bool allow_vxy_accelerate = std::abs(target_state_.x(vyaw)) <= 1.5;
    if (!accelerate_window_.init) {
        accelerate_window_.init = true;
        accelerate_window_.start_time = stamp;
        accelerate_window_.start_center = meas_center;
        accelerate_window_.start_predict_center =
            Eigen::Vector2d(target_state_.x(xc), target_state_.x(yc));
        accelerate_window_.samples.clear();
        pushWindowSample(accelerate_window_, stamp, meas_center, accelerate_window_.start_predict_center,
                         armor.image_area);
        return;
    }
    pushWindowSample(accelerate_window_, stamp, meas_center,
                     Eigen::Vector2d(target_state_.x(xc), target_state_.x(yc)), armor.image_area);
    double vx_meas = 0, vy_meas = 0, vx_pred = 0, vy_pred = 0, area_ratio = 1;
    if (!estimateWindowVelocity(vx_meas, vy_meas, vx_pred, vy_pred, area_ratio))
        return;
    auto& ekf_state = ekf_->getMutableState();
    const double vx_est = ekf_state.x(vx);
    const double vy_est = ekf_state.x(vy);
    const bool vx_sign_flip =
        (vx_meas * vx_est < 0.0) && (std::abs(vx_meas) > 0.15 && std::abs(vx_est) > 0.15);
    const bool vy_sign_flip =
        (vy_meas * vy_est < 0.0) && (std::abs(vy_meas) > 0.15 && std::abs(vy_est) > 0.15);
    const bool yaw_consistent = std::abs(yaw_error) < accelerate_window_.yaw_gate;
    const bool same_direction_x = (vx_meas * vx_pred) >= 0.0;
    const bool same_direction_y = (vy_meas * vy_pred) >= 0.0;
    const double vx_factor = vx_sign_flip ? accelerate_window_.max_gain_factor
                                          : (yaw_consistent && same_direction_x)
                                                ? computeAdaptiveGain(vx_est, vx_meas,
                                                                      accelerate_window_.max_gain_factor)
                                                : 1.0;
    const double vy_factor = vy_sign_flip ? accelerate_window_.max_gain_factor
                                          : (yaw_consistent && same_direction_y)
                                                ? computeAdaptiveGain(vy_est, vy_meas,
                                                                      accelerate_window_.max_gain_factor)
                                                : 1.0;
    const double vx_gain = vx_factor * accelerate_window_.base_gain;
    const double vy_gain = vy_factor * accelerate_window_.base_gain;
    if (enable_vx_accelerate_ && allow_vxy_accelerate) {
        ekf_state.x(vx) += vx_gain * (vx_meas - vx_est);
        vx_injected = true;
    }
    if (enable_vy_accelerate_ && allow_vxy_accelerate && allowVyInjection(vy_meas, vy_pred, area_ratio)) {
        ekf_state.x(vy) += vy_gain * (vy_meas - vy_est);
        vy_injected = true;
    }
    target_state_ = ekf_state;
    ekf_->setState(ekf_state);
}

void VehicleTracker::resetCenterYawObserver() {
    center_yaw_observer_.reset();
    stop_evidence_time_ = 0.0;
    stop_evidence_ready_ = false;
}

double VehicleTracker::selectContinuousCenterYaw(double armor_yaw) const {
    const double reference = center_yaw_observer_.initialized ? center_yaw_observer_.unwrapped_yaw
                                                              : target_state_.x(yaw);
    double continuous_yaw = armor_yaw;
    double smallest_delta = std::numeric_limits<double>::max();
    for (const double phase_offset : phase_offsets_) {
        const double candidate =
            reference + shortestAngularDistance(reference, armor_yaw - phase_offset);
        const double delta = std::abs(candidate - reference);
        if (delta < smallest_delta) {
            smallest_delta = delta;
            continuous_yaw = candidate;
        }
    }
    return continuous_yaw;
}

void VehicleTracker::observeCenterYaw(double stamp, double center_yaw) {
    auto& observer = center_yaw_observer_;
    const double wrapped_yaw = wrapPi(center_yaw);
    if (!observer.initialized) {
        observer.initialized = true;
        observer.last_wrapped_yaw = wrapped_yaw;
        observer.unwrapped_yaw = wrapped_yaw;
    } else {
        observer.unwrapped_yaw += shortestAngularDistance(observer.last_wrapped_yaw, wrapped_yaw);
        observer.last_wrapped_yaw = wrapped_yaw;
    }
    observer.samples.push_back({stamp, observer.unwrapped_yaw});
    while (!observer.samples.empty() &&
           (stamp - observer.samples.front().stamp) > stop_observer_window_time_)
        observer.samples.pop_front();
}

bool VehicleTracker::estimateCenterYawRate(double& meas_vyaw) const {
    meas_vyaw = 0.0;
    const auto& samples = center_yaw_observer_.samples;
    if (samples.size() < 4)
        return false;
    const double start_time = samples.front().stamp;
    double sum_t = 0.0;
    double sum_yaw = 0.0;
    for (const auto& sample : samples) {
        const double t = sample.stamp - start_time;
        sum_t += t;
        sum_yaw += sample.yaw;
    }
    const double count = static_cast<double>(samples.size());
    const double mean_t = sum_t / count;
    const double mean_yaw = sum_yaw / count;
    double numerator = 0.0;
    double denominator = 0.0;
    for (const auto& sample : samples) {
        const double t = sample.stamp - start_time;
        const double dt = t - mean_t;
        numerator += dt * (sample.yaw - mean_yaw);
        denominator += dt * dt;
    }
    if (denominator < 1e-6)
        return false;
    meas_vyaw = numerator / denominator;
    return std::isfinite(meas_vyaw);
}

void VehicleTracker::updateStopEvidence(double dt) {
    double meas_vyaw = 0.0;
    const bool rate_valid = estimateCenterYawRate(meas_vyaw);
    const bool meas_near_stop = std::abs(meas_vyaw) <= stop_observer_vyaw_threshold_;
    if (!rate_valid || !meas_near_stop) {
        stop_evidence_time_ = 0.0;
        stop_evidence_ready_ = false;
        return;
    }
    stop_evidence_time_ = std::min(stop_evidence_time_ + dt, stop_evidence_duration_);
    stop_evidence_ready_ = (stop_evidence_time_ >= stop_evidence_duration_);
}

bool VehicleTracker::applyStopVyawCorrection(double dt) {
    if (!stop_evidence_ready_ || isOutpostTarget())
        return false;
    double meas_vyaw = 0.0;
    if (!estimateCenterYawRate(meas_vyaw))
        return false;
    const double safe_time_constant = std::max(stop_vyaw_time_constant_, 1e-3);
    const double gain = 1.0 - std::exp(-dt / safe_time_constant);
    auto& st = ekf_->getMutableState();
    st.x(vyaw) *= (1.0 - gain);
    if (std::abs(st.x(vyaw)) <= stop_observer_vyaw_threshold_) {
        st.x(vyaw) = 0.0;
        vyaw_recovery_pending_ = enable_vyaw_accelerate_;
        stop_evidence_armed_ = false;
        stop_evidence_ready_ = false;
        stop_evidence_time_ = 0.0;
    }
    target_state_ = st;
    ekf_->setState(st);
    return true;
}

void VehicleTracker::resetZAccumulator() {
    z_accumulator_.reset();
    z_pair_states_[0] = ZPairState();
    z_pair_states_[1] = ZPairState();
}

void VehicleTracker::updateZAccumulator(bool& active, int& counter, double drift_speed) {
    if (std::abs(drift_speed) < z_drift_speed_threshold_) {
        ++counter;
        if (counter >= z_count_threshold_) {
            counter = z_count_threshold_;
            active = true;
        }
    } else {
        counter = 0;
        active = false;
    }
}

/**
 * 作用：本帧高度量测。输入：PnP 的 z、相位。输出：原样返回 z。
 */
double VehicleTracker::getSmoothZ(double z, int phase_idx) const {
    (void)phase_idx;
    return z;  // 相机系贴 PnP 高度；段平滑会把量测换成旧均值，板会离检测
}

void VehicleTracker::observeZSample(double meas_z, double stamp, int phase_idx, bool phase_changed) {
    auto finalize_segment = [this](int pair_idx) {
        if (z_accumulator_.z_count <= 0)
            return;
        auto& pair_state = z_pair_states_[static_cast<std::size_t>(pair_idx)];
        const double curr_mean = z_accumulator_.z_sum / z_accumulator_.z_count;
        if (!pair_state.prev_valid) {
            pair_state.prev_valid = true;
            pair_state.prev_time = z_accumulator_.end_time;
            pair_state.prev_mean = curr_mean;
            pair_state.smooth_z = curr_mean;
            pair_state.drift_speed = 0.0;
            pair_state.active = false;
            pair_state.stable_count = 0;
            return;
        }
        const double dt = z_accumulator_.end_time - pair_state.prev_time;
        if (dt > 1e-3) {
            pair_state.drift_speed = std::abs(curr_mean - pair_state.prev_mean) / dt;
            updateZAccumulator(pair_state.active, pair_state.stable_count, pair_state.drift_speed);
        } else {
            pair_state.drift_speed = 0;
        }
        pair_state.smooth_z =
            (1.0 - z_smooth_alpha_) * pair_state.smooth_z + z_smooth_alpha_ * curr_mean;
        pair_state.prev_mean = curr_mean;
        pair_state.prev_time = z_accumulator_.end_time;
        pair_state.prev_valid = true;
    };
    const int curr_pair_idx = isPair0Phase(phase_idx) ? 0 : 1;
    if (!z_accumulator_.init) {
        z_accumulator_.init = true;
        z_accumulator_.phase_idx = phase_idx;
        z_accumulator_.pair_idx = curr_pair_idx;
    }
    const bool phase_switched = phase_changed || (phase_idx != z_accumulator_.phase_idx);
    if (phase_switched) {
        finalize_segment(z_accumulator_.pair_idx);
        z_accumulator_.reset();
        z_accumulator_.init = true;
        z_accumulator_.phase_idx = phase_idx;
        z_accumulator_.pair_idx = curr_pair_idx;
    }
    z_accumulator_.z_sum += meas_z;
    ++z_accumulator_.z_count;
    z_accumulator_.end_time = stamp;
}

/**
 * 作用：双板可见时走联合更新；否则单帧。dt 非法则重新 init。
 * 输入：本 id 的板指针、当前时刻。输出：后验。
 */
void VehicleTracker::update(const std::vector<TrackedArmor*>& armors, double stamp) {
    if (bootstrap_running_ || isOutpostTarget() || armors.size() <= 1) {
        updateSingleFrame(armors.empty() ? nullptr : armors.front(), stamp);
            return;
    }
    const double dt = stamp - last_time_;
    const bool invalid_dt = !std::isfinite(dt) || dt <= 0.0 || dt > params_.max_dt_s;
    if (invalid_dt) {
        const auto seed_it = std::find_if(armors.begin(), armors.end(), [this](const TrackedArmor* armor) {
            return armor != nullptr && armor->id == tracked_id_ && !armor->pose_roots.empty();
        });
        if (seed_it != armors.end()) {
            init(*seed_it);
            return;
        }
        last_time_ = stamp;
        return;
    }
    last_time_ = stamp;
    bool matched = false;
    bool vx_injected = false;
    bool vy_injected = false;
    ekf_->predict(dt);
    target_state_ = ekf_->getState();
    fadeVelocityCov(target_state_.P);  // 防止 v/ω 的 P 收死，四板预测才能跟上加速
    floorStateCov(target_state_.P);    // 位置/朝向下限，本帧增益才能贴 PnP
    ekf_->setState(target_state_);
    if (std::abs(target_state_.x(vyaw)) > high_spin_start_vyaw_)
        stop_evidence_armed_ = true;
    ArmorAssociation association;
    last_phase_change_ = false;
    if (associateNormalArmors(armors, association)) {
        matched = true;
        const ArmorAssociation::Entry& primary = association.primary;
        tracked_armor_ = *primary.armor;
        tracked_type_ = primary.armor->type;
        last_yaw_ = primary.z(3);
        const double center_yaw = selectContinuousCenterYaw(primary.z(3));
        observeCenterYaw(stamp, center_yaw);
        if (stop_evidence_armed_)
            updateStopEvidence(dt);
        bool allow_update = true;
        bool phase_changed = false;
        if (primary.phase_match.phase_idx == confirmed_phase_idx_) {
            hypothetic_phase_idx_ = -1;
            hypothetic_phase_count_ = 0;
            accelerateConvergence(*primary.armor, primary.z, stamp, vx_injected, vy_injected);
        } else if (primary.phase_match.change) {
            if (primary.phase_match.phase_idx == hypothetic_phase_idx_)
                ++hypothetic_phase_count_;
            else {
                hypothetic_phase_idx_ = primary.phase_match.phase_idx;
                hypothetic_phase_count_ = 1;
            }
            if (hypothetic_phase_count_ >= phase_confirm_threshold_) {
                confirmed_phase_idx_ = hypothetic_phase_idx_;
                hypothetic_phase_idx_ = -1;
                hypothetic_phase_count_ = 0;
                if (vyaw_recovery_pending_) {
                    target_state_.P.row(vyaw).setZero();
                    target_state_.P.col(vyaw).setZero();
                    target_state_.P(vyaw, vyaw) = 25.0;
                    ekf_->setState(target_state_);
                    vyaw_recovery_pending_ = false;
                }
                last_phase_change_time_ = stamp;
                last_phase_change_yaw_ = target_state_.x(yaw);
                phase_changed = true;
            }
            // 未确认换相时仍更新：看见的那块板要立刻拉动车心/朝向，否则四板会停两帧
        } else {
            hypothetic_phase_idx_ = -1;
            hypothetic_phase_count_ = 0;
        }
        const int update_phase =
            (primary.phase_match.phase_idx >= 0) ? primary.phase_match.phase_idx : confirmed_phase_idx_;
        Eigen::Vector4d primary_z = primary.z;
        primary_z(2) = getSmoothZ(primary_z(2), update_phase);  // 本帧 PnP 高度，不再换成段均值
        EKFCore::MeasurementEvaluation primary_eval =
            ekf_->evaluateMeasurement(primary_z, target_state_.x, update_phase, primary.R);
        last_nis_ = primary_eval.nis;
        if (!std::isfinite(primary_eval.nis))
            allow_update = false;
        else
            inflateOutlierR(primary_eval, nis_gate_);  // 只挡飞点（门限×8），普通机动全权更新
        bool use_support = false;
        EKFCore::MeasurementEvaluation batch_eval;
        if (allow_update && association.has_support) {
            const ArmorAssociation::Entry& support = association.support;
            EKFCore::MeasurementEvaluation support_eval = ekf_->evaluateMeasurement(
                support.z, target_state_.x, support.phase_match.phase_idx, support.R);
            if (std::isfinite(support_eval.nis)) {
                inflateOutlierR(support_eval, nis_gate_);
                batch_eval = ekf_->evaluateBatch({primary_eval, support_eval});
                if (std::isfinite(batch_eval.nis)) {
                    inflateOutlierR(batch_eval, 2.0 * nis_gate_);
                    use_support = true;
                }
            }
        }
        observeZSample(primary_z(2), stamp, update_phase, phase_changed);
        last_phase_change_ = phase_changed;
        if (allow_update) {
            if (use_support) {
                ekf_->update(batch_eval);
            } else {
                ekf_->updatePartial(primary_eval, getActiveIndices(update_phase));
                target_state_ = ekf_->getState();
                decoupleInactiveCovariance(target_state_.P, update_phase);
                ekf_->setState(target_state_);
            }
        }
        applyStopVyawCorrection(dt);
    }
    updateStateMachine(matched);
    target_state_ = ekf_->getState();
    if (matched) {
        const double z_pair0 = target_state_.x(z1);
        const double z_pair1 = target_state_.x(z2);
        dz_ = 0.5 * std::abs(z_pair0 - z_pair1);
        za_ = isPair0Phase(confirmed_phase_idx_) ? z_pair0 : z_pair1;
    }
    target_state_.x(r1) = std::clamp(target_state_.x(r1), 0.1, params_.max_r_m);
    target_state_.x(r2) = std::clamp(target_state_.x(r2), 0.1, params_.max_r_l_m);
    floorStateCov(target_state_.P);
    ekf_->setState(target_state_);
    refreshPublishSmooth(matched);
}

/**
 * 作用：bootstrap / 前哨 / 单板路径。输入：至多一块板。输出：后验。
 */
void VehicleTracker::updateSingleFrame(TrackedArmor* armor, double stamp) {
    const double dt = stamp - last_time_;
    const bool invalid_dt = !std::isfinite(dt) || dt > params_.max_dt_s || dt <= 0.0;
    if (invalid_dt) {
        if (armor != nullptr && armor->id == tracked_id_ && !armor->pose_roots.empty()) {
            init(armor);
            return;
        }
        last_time_ = stamp;
        updateStateMachine(false);
        return;
    }
    last_time_ = stamp;
    if (bootstrap_running_) {
        const bool bootstrap_observation =
            armor != nullptr && armor->id == tracked_id_ && !armor->pose_roots.empty();
        const bool bootstrap_finished = bootstrap_observation && advanceBootstrap(*armor);
        if (!bootstrap_finished) {
            if (!bootstrap_observation)
                updateStateMachine(false);
            // 尚未决胜：overlay 跟累积 NLL 更小的那条假设。
            const int best = (bootstrap_tracks_[1].ekf &&
                              bootstrap_tracks_[1].accumulated_nll < bootstrap_tracks_[0].accumulated_nll)
                                 ? 1
                                 : 0;
            if (bootstrap_tracks_[static_cast<std::size_t>(best)].ekf) {
                target_state_ = bootstrap_tracks_[static_cast<std::size_t>(best)].ekf->getState();
                if (target_state_.x.size() >= state_dim) {
                    published_xc_ = target_state_.x(xc);
                    published_yc_ = target_state_.x(yc);
                    published_vyaw_ = target_state_.x(vyaw);
                }
            }
            return;
        }
        updateStateMachine(true);
        refreshPublishSmooth(true);
        return;
    }
    ekf_->predict(dt);
    target_state_ = ekf_->getState();
    const bool is_outpost = isOutpostTarget();
    if (!is_outpost) {
        fadeVelocityCov(target_state_.P);
        floorStateCov(target_state_.P);
        decoupleInactiveCovariance(target_state_.P, confirmed_phase_idx_);
        ekf_->setState(target_state_);
    }
    if (!is_outpost && std::abs(target_state_.x(vyaw)) > high_spin_start_vyaw_)
        stop_evidence_armed_ = true;
    RootPhaseCandidate root_candidate;
    const bool root_candidate_valid =
        armor != nullptr && armor->id == tracked_id_ &&
        (is_outpost ? selectOutpostRoot(*armor, root_candidate)
                    : selectNormalRoot(*armor, root_candidate));
    bool matched = false;
    last_phase_change_ = false;
    if (root_candidate_valid) {
        matched = true;
        const OutpostState prev_outpost_state = outpost_state_;
        const EKFCore::State prev_target_state = target_state_;
        const TrackedArmor prev_tracked_armor = tracked_armor_;
        tracked_armor_ = *armor;
        tracked_armor_.selected_root_idx = root_candidate.root_idx;
        tracked_type_ = armor->type;
        const Eigen::Vector4d measurement = root_candidate.measurement;
        const Eigen::Matrix4d measurement_noise = root_candidate.measurement_noise;
        const EKFCore::PhaseMatch phase_match = root_candidate.phase_match;
        if (!is_outpost)
            last_yaw_ = measurement(3);
        int& committed_phase = is_outpost ? outpost_state_.phase_slot : confirmed_phase_idx_;
        int eval_phase = committed_phase;
        if (!is_outpost && phase_match.phase_idx >= 0)
            eval_phase = phase_match.phase_idx;  // 看见哪块就对哪对做量测，四板才转得动
        auto evaluateAt = [&](int phase) {
            Eigen::Vector4d eval_meas = measurement;
            if (!is_outpost)
                eval_meas(2) = getSmoothZ(measurement(2), phase);  // 本帧 PnP 高度
            return ekf_->evaluateMeasurement(eval_meas, target_state_.x, phase, measurement_noise);
        };
        EKFCore::MeasurementEvaluation evaluation;
        bool allow_update = true;
        bool change = false;
        bool vx_injected = false;
        bool vy_injected = false;
        if (!is_outpost) {
            observeCenterYaw(stamp, selectContinuousCenterYaw(measurement(3)));
            if (stop_evidence_armed_)
                updateStopEvidence(dt);
        }
        if (phase_match.phase_idx == committed_phase) {
            hypothetic_phase_idx_ = -1;
            hypothetic_phase_count_ = 0;
            if (is_outpost) {
                updateOutpostState(measurement(3), measurement(2), stamp, false, committed_phase);
                ekf_->setState(target_state_);
                evaluation = evaluateAt(committed_phase);
            } else {
                evaluation = evaluateAt(eval_phase);
                accelerateConvergence(*armor, measurement, stamp, vx_injected, vy_injected);
            }
        } else if (phase_match.change) {
            if (phase_match.phase_idx == hypothetic_phase_idx_)
                ++hypothetic_phase_count_;
            else {
                hypothetic_phase_idx_ = phase_match.phase_idx;
                hypothetic_phase_count_ = 1;
            }
            if (hypothetic_phase_count_ >= phase_confirm_threshold_) {
                const int next_phase = hypothetic_phase_idx_;
                hypothetic_phase_idx_ = -1;
                hypothetic_phase_count_ = 0;
                if (is_outpost)
                    updateOutpostState(measurement(3), measurement(2), stamp, true, next_phase);
                else if (vyaw_recovery_pending_) {
                    target_state_.P.row(vyaw).setZero();
                    target_state_.P.col(vyaw).setZero();
                    target_state_.P(vyaw, vyaw) = 25.0;
                    ekf_->setState(target_state_);
                    vyaw_recovery_pending_ = false;
                }
                committed_phase = next_phase;
                eval_phase = next_phase;
                evaluation = evaluateAt(eval_phase);
                change = true;
            } else {
                evaluation = evaluateAt(eval_phase);
            }
        } else {
            hypothetic_phase_idx_ = -1;
            hypothetic_phase_count_ = 0;
            evaluation = evaluateAt(eval_phase);
        }
        last_nis_ = evaluation.nis;
        if (!is_outpost) {
            if (!std::isfinite(evaluation.nis))
                allow_update = false;
            else
                inflateOutlierR(evaluation, nis_gate_);  // 只挡飞点（门限×8）
        }
        if (is_outpost && allow_update) {
            if (!std::isfinite(evaluation.nis) || !std::isfinite(evaluation.nll)) {
                outpost_state_ = prev_outpost_state;
                target_state_ = prev_target_state;
                tracked_armor_ = prev_tracked_armor;
                ekf_->setState(target_state_);
                allow_update = false;
                matched = false;
                change = false;
            } else {
                inflateOutlierR(evaluation, nis_gate_);  // 前哨也不因普通新息丢更新
                ekf_->setState(target_state_);
                if (change) {
                    last_phase_change_time_ = stamp;
                    last_phase_change_yaw_ = target_state_.x(o_yaw);
                }
            }
        }
        if (!is_outpost)
            observeZSample(measurement[2], stamp, eval_phase, change);
        last_phase_change_ = change;
        if (allow_update) {
            if (is_outpost) {
                ekf_->update(evaluation);
            } else {
                ekf_->updatePartial(evaluation, getActiveIndices(eval_phase));
                target_state_ = ekf_->getState();
                decoupleInactiveCovariance(target_state_.P, eval_phase);
                ekf_->setState(target_state_);
            }
        }
        if (!is_outpost)
            applyStopVyawCorrection(dt);
    }
    updateStateMachine(matched);
    target_state_ = ekf_->getState();
    if (is_outpost) {
        const int outpost_id = getOutpostID();
        const double zc = getZC();
        dz_ = outpost_z_step_;
        za_ = (outpost_id >= 1 && outpost_id <= 3) ? zc + outpost_z_step_ * (outpost_id - 2) : zc;
        target_state_.x(o_r) = outpost_radius_;
        target_state_.P.row(o_r).setZero();
        target_state_.P.col(o_r).setZero();
        target_state_.P(o_r, o_r) = 1e-6;
    } else if (armor != nullptr && armor->id == tracked_id_) {
        const double z_pair0 = target_state_.x(z1);
        const double z_pair1 = target_state_.x(z2);
        dz_ = 0.5 * std::abs(z_pair0 - z_pair1);
        za_ = isPair0Phase(confirmed_phase_idx_) ? z_pair0 : z_pair1;
        target_state_.x(r1) = std::clamp(target_state_.x(r1), 0.1, params_.max_r_m);
        target_state_.x(r2) = std::clamp(target_state_.x(r2), 0.1, params_.max_r_l_m);
    }
    floorStateCov(target_state_.P);
    ekf_->setState(target_state_);
    refreshPublishSmooth(matched);
}

void VehicleTracker::updateStateMachine(bool matched) {
    if (state_ == VehicleTrackState::Detecting) {
        if (matched) {
            ++detect_count_;
            if (detect_count_ > params_.tracking_threshold) {
                detect_count_ = 0;
                state_ = VehicleTrackState::Tracking;
            }
        } else {
            detect_count_ = 0;
            state_ = VehicleTrackState::Lost;
        }
    } else if (state_ == VehicleTrackState::Tracking) {
        if (!matched) {
            state_ = VehicleTrackState::TempLost;
            ++lost_count_;
        }
    } else if (state_ == VehicleTrackState::TempLost) {
        if (!matched) {
            ++lost_count_;
            if (lost_count_ > params_.lost_threshold) {
                reset();
            }
        } else {
            state_ = VehicleTrackState::Tracking;
            lost_count_ = 0;
        }
    }
}

void VehicleTracker::resetOutpostState() {
    outpost_state_ = OutpostState();
    if (ekf_) {
        if (model_ == Model::Outpost)
            ekf_->setNoise(Q_outpost_, R_outpost_);
        else if (Q_.rows() > 0)
            ekf_->setNoise(Q_, R_);
    }
}

void VehicleTracker::finishOutpostZSegment() {
    auto median = [](std::vector<double>& samples, double& value) {
        if (samples.empty())
            return false;
        std::sort(samples.begin(), samples.end());
        const std::size_t middle = samples.size() / 2;
        value = samples.size() % 2 == 0 ? 0.5 * (samples[middle - 1] + samples[middle]) : samples[middle];
        return true;
    };
    auto& outpost = outpost_state_;
    if (isOutpostConfirmed()) {
        double segment_z = 0.0;
        if (!median(outpost.phase_z_samples, segment_z))
            return;
        const int outpost_id = getOutpostID();
        if (outpost_id >= 1 && outpost_id <= 3) {
            const double center_sample = segment_z - outpost_z_step_ * (outpost_id - 2);
            double& center_z = outpost.center_z;
            center_z = std::isfinite(center_z)
                           ? outpost_center_z_alpha_ * center_sample +
                                 (1.0 - outpost_center_z_alpha_) * center_z
                           : center_sample;
        }
        outpost.phase_z_samples.clear();
        return;
    }
    auto& samples = outpost.z_samples[static_cast<std::size_t>(outpost.phase_slot)];
    double segment_z = 0.0;
    if (!median(samples, segment_z))
        return;
    samples.assign(1, segment_z);
}

bool VehicleTracker::tryMapOutpostID() {
    const auto& outpost = outpost_state_;
    const bool all_segments_finished =
        std::all_of(outpost.z_samples.begin(), outpost.z_samples.end(),
                    [](const auto& samples) { return samples.size() == 1; });
    if (outpost.direction == 0 || !all_segments_finished)
        return false;
    std::array<int, 3> order{{0, 1, 2}};
    std::sort(order.begin(), order.end(), [&outpost](int a, int b) {
        return outpost.z_samples[static_cast<std::size_t>(a)][0] <
               outpost.z_samples[static_cast<std::size_t>(b)][0];
    });
    std::array<int, 3> slot_ids{};
    for (int rank = 0; rank < 3; ++rank)
        slot_ids[static_cast<std::size_t>(order[static_cast<std::size_t>(rank)])] = rank + 1;
    for (int slot = 0; slot < 3; ++slot) {
        const int next_slot = nextOutpostPhase(slot);
        const int expected_next_id = (slot_ids[static_cast<std::size_t>(slot)] - 1 + outpost.direction + 3) % 3 + 1;
        if (slot_ids[static_cast<std::size_t>(next_slot)] != expected_next_id)
            return false;
    }
    std::array<double, 3> center_samples{};
    for (int slot = 0; slot < 3; ++slot) {
        const int outpost_id = slot_ids[static_cast<std::size_t>(slot)];
        center_samples[static_cast<std::size_t>(slot)] =
            outpost.z_samples[static_cast<std::size_t>(slot)][0] - outpost_z_step_ * (outpost_id - 2);
    }
    std::sort(center_samples.begin(), center_samples.end());
    const double z_step =
        0.5 * (outpost.z_samples[static_cast<std::size_t>(order[2])][0] -
               outpost.z_samples[static_cast<std::size_t>(order[0])][0]);
    if (std::isfinite(z_step) && z_step > 1e-3) {
        outpost_z_step_ = z_step;
        for (int slot = 0; slot < 3; ++slot) {
            const int outpost_id = slot_ids[static_cast<std::size_t>(slot)];
            center_samples[static_cast<std::size_t>(slot)] =
                outpost.z_samples[static_cast<std::size_t>(slot)][0] -
                outpost_z_step_ * (outpost_id - 2);
        }
        std::sort(center_samples.begin(), center_samples.end());
    }
    outpost_state_.slot_ids = slot_ids;
    outpost_state_.center_z = center_samples[1];
    outpost_state_.phase_z_samples.clear();
    target_state_.x(o_zc) = center_samples[1];
    return true;
}

void VehicleTracker::recalibrateOutpostMapping() {
    auto& outpost = outpost_state_;
    if (std::any_of(outpost.turn_z_samples.begin(), outpost.turn_z_samples.end(),
                    [](const auto& samples) { return samples.empty(); }))
        return;
    auto turn_z_samples = std::move(outpost.turn_z_samples);
    outpost.turn_z_samples = {};
    for (int slot = 0; slot < 3; ++slot) {
        auto& samples = turn_z_samples[static_cast<std::size_t>(slot)];
        if (samples.empty())
            return;
        std::sort(samples.begin(), samples.end());
        const std::size_t middle = samples.size() / 2;
        const double med =
            samples.size() % 2 == 0 ? 0.5 * (samples[middle - 1] + samples[middle]) : samples[middle];
        outpost.z_samples[static_cast<std::size_t>(slot)].assign(1, med);
    }
    tryMapOutpostID();
}

void VehicleTracker::updateOutpostState(double meas_yaw, double meas_z, double stamp, bool phase_changed,
                                        int next_phase) {
    auto& outpost = outpost_state_;
    if (!outpost.has_last_yaw) {
        outpost.has_last_yaw = true;
        outpost.last_yaw = meas_yaw;
        outpost.z_samples[static_cast<std::size_t>(outpost.phase_slot)].push_back(meas_z);
        return;
    }
    const double delta_yaw = shortestAngularDistance(outpost.last_yaw, meas_yaw);
    outpost.last_yaw = meas_yaw;
    const double duration = stamp - outpost.last_event_time;
    const bool yaw_event =
        (std::abs(delta_yaw) > direction_yaw_threshold_) && (duration > 0.5 * outpost_phase_duration_);
    if (outpost.direction == 0) {
        if (!yaw_event) {
            outpost.z_samples[static_cast<std::size_t>(outpost.phase_slot)].push_back(meas_z);
            return;
        }
        finishOutpostZSegment();
        outpost.direction = (delta_yaw > 0) ? -1 : 1;
        outpost.static_mode = false;
        target_state_.x(o_vyaw) = outpost.direction * outpost_vyaw_;
        ekf_->setState(target_state_);
        outpost.phase_slot = nextOutpostPhase(outpost.phase_slot);
        outpost.z_samples[static_cast<std::size_t>(outpost.phase_slot)].push_back(meas_z);
        outpost.last_event_time = stamp;
        return;
    }
    if (phase_changed) {
        finishOutpostZSegment();
        if (isOutpostConfirmed()) {
            if (++outpost.turn_phase_count == 3) {
                recalibrateOutpostMapping();
                outpost.turn_phase_count = 0;
            }
        } else {
            tryMapOutpostID();
        }
        outpost.phase_slot = std::clamp(next_phase, 0, 2);
        outpost.last_event_time = stamp;
        if (isOutpostConfirmed()) {
            outpost.phase_z_samples.push_back(meas_z);
            outpost.turn_z_samples[static_cast<std::size_t>(outpost.phase_slot)].push_back(meas_z);
        } else {
            outpost.z_samples[static_cast<std::size_t>(outpost.phase_slot)].push_back(meas_z);
        }
        return;
    }
    if (isOutpostConfirmed()) {
        outpost.phase_z_samples.push_back(meas_z);
        outpost.turn_z_samples[static_cast<std::size_t>(outpost.phase_slot)].push_back(meas_z);
        return;
    }
    outpost.z_samples[static_cast<std::size_t>(outpost.phase_slot)].push_back(meas_z);
    if (!yaw_event)
        return;
    finishOutpostZSegment();
    tryMapOutpostID();
    outpost.phase_slot = nextOutpostPhase(outpost.phase_slot);
    outpost.last_event_time = stamp;
}

std::vector<double> VehicleTracker::getPhaseOffsets() const {
    if (armor_num_ == 3)
        return {0.0, 2.0 * kPi / 3.0, -2.0 * kPi / 3.0};
    return {0.0, kPi / 2.0, kPi, -kPi / 2.0};
}

bool VehicleTracker::isPair0Phase(int phase_idx) const {
    return phase_idx < 0 || phase_idx == 0 || phase_idx == 2;
}

double VehicleTracker::getZC() const {
    if (isOutpostTarget()) {
        const double center_z = outpost_state_.center_z;
        return std::isfinite(center_z) ? center_z : target_state_.x(o_zc);
    }
    if (target_state_.x.size() < state_dim)
        return 0.0;
    return 0.5 * (target_state_.x(z1) + target_state_.x(z2));
}

double VehicleTracker::getCurrentRadius() const {
    if (isOutpostTarget())
        return target_state_.x.size() > o_r ? target_state_.x(o_r) : outpost_radius_;
    if (target_state_.x.size() < state_dim)
        return default_r_;
    return isPair0Phase(confirmed_phase_idx_) ? target_state_.x(r1) : target_state_.x(r2);
}

double VehicleTracker::getOtherRadius() const {
    if (isOutpostTarget())
        return getCurrentRadius();
    if (target_state_.x.size() < state_dim)
        return default_r_;
    return isPair0Phase(confirmed_phase_idx_) ? target_state_.x(r2) : target_state_.x(r1);
}

double VehicleTracker::getYawFromOrientation(const TrackedArmor& armor) {
    const double yaw_raw = armor.selectedRoot().yaw;
    const double yaw = last_yaw_ + shortestAngularDistance(last_yaw_, yaw_raw);
    last_yaw_ = yaw;
    return yaw;
}

double VehicleTracker::computeAdaptiveGain(double v_old, double v_new, double max_gain_factor) const {
    auto smoothStep = [](double t) {
        t = std::clamp(t, 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    };
    const double ratio = std::abs(v_new) / std::max(std::abs(v_old), 0.1);
    double gain_scale = 1.0;
    if (ratio >= 0.8 && ratio <= 1.2)
        gain_scale = 1.0;
    else if (ratio < 0.5 || ratio >= 2.0)
        gain_scale = max_gain_factor;
    else if (ratio < 0.9) {
        const double t = (ratio - 0.5) / (0.8 - 0.5);
        gain_scale = max_gain_factor + (1.0 - max_gain_factor) * smoothStep(t);
    } else {
        const double t = (ratio - 1.2) / (2.0 - 1.2);
        gain_scale = 1.0 + (max_gain_factor - 1.0) * smoothStep(t);
    }
    return std::clamp(gain_scale, 1.0, max_gain_factor);
}

/**
 * 作用：pnp_ok 板 → 跟踪系一根位姿（锁俯仰后只有一根，双根 bootstrap 自动跳过）。
 * 输入：检测装甲。输出：观测列表。
 */
std::vector<VehicleTracker::TrackedArmor> VehicleTracker::makeObservations(
    const std::vector<ArmorBoard>& armors) const {
    std::vector<TrackedArmor> out;
    const cv::Mat& K = ArmorBoard::cameraMatrix();
    double cx = 0.0;
    double cy = 0.0;
    if (K.rows >= 3 && K.cols >= 3) {
        cx = K.at<double>(0, 2);
        cy = K.at<double>(1, 2);
    }
    for (const auto& a : armors) {
        if (!a.pnp_ok || a.dist_m <= 0.0)
            continue;
        TrackedArmor obs;
        obs.stamp = time_s_;
        obs.id = a.number;
        obs.color = a.color();
        obs.type = a.type;
        obs.image_center = Eigen::Vector2d(a.center.x, a.center.y);
        obs.distance_to_image_center = std::hypot(a.center.x - cx, a.center.y - cy);
        const cv::Point2f d0 = a.right_top - a.left_bottom;
        const cv::Point2f d1 = a.right_bottom - a.left_top;
        obs.image_area = 0.5 * std::abs(d0.x * d1.y - d0.y * d1.x);
        PoseRoot root;
        const cv::Point3d mm = a.positionMm();
        root.source_cam_mm = mm;
        root.position = camToTrk(Eigen::Vector3d(mm.x / 1000.0, mm.y / 1000.0, mm.z / 1000.0));
        root.yaw = armorYawTrk(a.normal_cam);
        root.visual_covariance = Eigen::Matrix4d::Zero();
        const double e = std::max(a.reproj_error_px, 0.0);
        // 重投影只说明 PnP 质量。σ 不能放到厘米以上，否则 R 太大跟踪贴不住板。
        const double pos_sigma = std::clamp(0.004 * (1.0 + 0.1 * e), 0.003, 0.012);
        const double yaw_sigma = std::clamp(0.03 * (1.0 + 0.05 * e), 0.02, 0.06);
        const double pos_var = std::max(root_min_variance_, pos_sigma * pos_sigma);
        root.visual_covariance(0, 0) = pos_var;
        root.visual_covariance(1, 1) = pos_var;
        root.visual_covariance(2, 2) = pos_var;
        root.visual_covariance(3, 3) = std::max(root_min_variance_, yaw_sigma * yaw_sigma);
        obs.pose_roots.push_back(root);
        out.push_back(std::move(obs));
    }
    return out;
}

/**
 * 作用：像心最近优先；已跟踪则粘滞 id，除非新目标近过 id_switch_factor 倍。最多留两块同号板。
 * 输入：本帧观测（可改 selected_root）。输出：指针，指向 frame_armors_ 里的元素。
 */
std::vector<VehicleTracker::TrackedArmor*> VehicleTracker::selectTargetArmors(
    std::vector<TrackedArmor>& armors) {
    std::vector<TrackedArmor*> selected;
    if (armors.empty())
        return selected;
    auto best_it = std::min_element(armors.begin(), armors.end(),
                                    [](const TrackedArmor& lhs, const TrackedArmor& rhs) {
                                        return lhs.distance_to_image_center < rhs.distance_to_image_center;
                                    });
    int selected_id = best_it->id;
    if (state_ != VehicleTrackState::Lost) {
        const int tracked_id = tracked_id_;
        auto tracked_it = std::find_if(armors.begin(), armors.end(),
                                       [tracked_id](const TrackedArmor& armor) { return armor.id == tracked_id; });
        const bool use_new_id =
            tracked_it == armors.end() ||
            (best_it->id != tracked_id &&
             best_it->distance_to_image_center * params_.id_switch_factor <
                 tracked_it->distance_to_image_center);
        if (!use_new_id)
            selected_id = tracked_id;
    }
    for (TrackedArmor& armor : armors) {
        if (armor.id == selected_id)
            selected.push_back(&armor);
    }
    std::sort(selected.begin(), selected.end(), [](const TrackedArmor* lhs, const TrackedArmor* rhs) {
        return lhs->distance_to_image_center < rhs->distance_to_image_center;
    });
    if (selected.size() > 2)
        selected.resize(2);
    return selected;
}

void VehicleTracker::resetVyawSmooth() {
    vyaw_seg_sum_ = 0.0;
    vyaw_seg_count_ = 0;
    vyaw_smooth_window_.clear();
}

void VehicleTracker::pushVyawSample(double sample_vyaw) {
    vyaw_smooth_window_.push_back(sample_vyaw);
    while (static_cast<int>(vyaw_smooth_window_.size()) > params_.vyaw_smooth_window_size)
        vyaw_smooth_window_.pop_front();
}

double VehicleTracker::computeMeanVyaw() const {
    if (vyaw_smooth_window_.empty())
        return std::numeric_limits<double>::quiet_NaN();
    double sum = 0.0;
    for (const double v : vyaw_smooth_window_)
        sum += v;
    return sum / static_cast<double>(vyaw_smooth_window_.size());
}

void VehicleTracker::computeVyawSegment(double raw_vyaw, bool phase_change) {
    if (phase_change) {
        if (vyaw_seg_count_ > 0)
            pushVyawSample(vyaw_seg_sum_ / vyaw_seg_count_);
        vyaw_seg_sum_ = raw_vyaw;
        vyaw_seg_count_ = 1;
        return;
    }
    vyaw_seg_sum_ += raw_vyaw;
    ++vyaw_seg_count_;
}

double VehicleTracker::smoothVyaw(double raw_vyaw, bool phase_change) {
    if (!params_.enable_smooth_vyaw)
        return raw_vyaw;
    if (std::abs(raw_vyaw) <= params_.min_vyaw_smooth_vel) {
        resetVyawSmooth();
        return raw_vyaw;
    }
    computeVyawSegment(raw_vyaw, phase_change);
    if (static_cast<int>(vyaw_smooth_window_.size()) < params_.min_vyaw_smooth_sample)
        return raw_vyaw;
    for (std::size_t i = 1; i < vyaw_smooth_window_.size(); ++i) {
        if (std::abs(vyaw_smooth_window_[i] - vyaw_smooth_window_[i - 1]) >
            params_.max_vyaw_smooth_win_diff)
            return raw_vyaw;
    }
    const double window_mean = computeMeanVyaw();
    if (!std::isfinite(window_mean))
        return raw_vyaw;
    if (std::abs(window_mean - raw_vyaw) > params_.max_vyaw_smooth_raw_diff)
        return raw_vyaw;
    return window_mean;
}

/**
 * 作用：lovod process_node 的 xy/ω 发布平滑，只改 published_*，不改 EKF。
 * 相机系默认发布 EKF 原值；yaml enable_smooth_normal 才做车心低通。
 * 输入：本帧是否匹配。输出：published_xc/yc/vyaw。
 */
void VehicleTracker::refreshPublishSmooth(bool matched) {
    (void)matched;
    if (state_ == VehicleTrackState::Lost || state_ == VehicleTrackState::Detecting ||
        (isOutpostTarget() && !isOutpostStatic() && !isOutpostConfirmed())) {
        smooth_pos_.setConstant(std::numeric_limits<double>::quiet_NaN());
        outpost_smooth_pos_.setConstant(std::numeric_limits<double>::quiet_NaN());
        enable_smooth_x_ = enable_smooth_y_ = false;
        enable_outpost_smooth_x_ = enable_outpost_smooth_y_ = false;
        resetVyawSmooth();
        if (isOutpostTarget() && target_state_.x.size() >= o_state_dim) {
            published_xc_ = target_state_.x(o_xc);
            published_yc_ = target_state_.x(o_yc);
            published_vyaw_ = target_state_.x(o_vyaw);
        } else if (target_state_.x.size() >= state_dim) {
            published_xc_ = target_state_.x(xc);
            published_yc_ = target_state_.x(yc);
            published_vyaw_ = target_state_.x(vyaw);
        }
        return;
    }
    if (isOutpostTarget()) {
        const double raw_x = target_state_.x(o_xc);
        const double raw_y = target_state_.x(o_yc);
        published_vyaw_ = target_state_.x(o_vyaw);
        if (!isOutpostConfirmed()) {
            published_xc_ = raw_x;
            published_yc_ = raw_y;
            outpost_smooth_pos_.setConstant(std::numeric_limits<double>::quiet_NaN());
            return;
        }
        const Eigen::Vector2d raw_position(raw_x, raw_y);
        const double smooth_distance = outpost_smooth_pos_.allFinite()
                                           ? (raw_position - outpost_smooth_pos_).norm()
                                           : std::numeric_limits<double>::infinity();
        const bool smooth_valid =
            outpost_smooth_pos_.allFinite() &&
            smooth_distance < params_.outpost_quit_smooth_distance_threshold;
        if (smooth_valid && params_.enable_smooth_outpost)
            outpost_smooth_pos_ = params_.outpost_smooth_alpha * raw_position +
                                  (1.0 - params_.outpost_smooth_alpha) * outpost_smooth_pos_;
        else
            outpost_smooth_pos_ = raw_position;
        published_xc_ = outpost_smooth_pos_.x();
        published_yc_ = outpost_smooth_pos_.y();
        return;
    }
    const double raw_x = target_state_.x(xc);
    const double raw_y = target_state_.x(yc);
    const double raw_vyaw = target_state_.x(vyaw);
    published_vyaw_ = smoothVyaw(raw_vyaw, last_phase_change_);
    // 相机系 overlay 跟 EKF 原值。lovod 自旋时单独平滑 x 或 y 会让车心离开预测板。
    enable_smooth_x_ = false;
    enable_smooth_y_ = false;
    quit_count_x_ = quit_count_y_ = 0;
    const Eigen::Vector2d raw_position(raw_x, raw_y);
    const bool smooth_valid =
        smooth_pos_.allFinite() &&
        (raw_position - smooth_pos_).norm() < params_.quit_smooth_distance_threshold;
    if (smooth_valid && params_.enable_smooth_normal)
        smooth_pos_ = params_.smooth_alpha * raw_position + (1.0 - params_.smooth_alpha) * smooth_pos_;
    else
        smooth_pos_ = raw_position;
    published_xc_ = params_.enable_smooth_normal ? smooth_pos_.x() : raw_x;
    published_yc_ = params_.enable_smooth_normal ? smooth_pos_.y() : raw_y;
}

/**
 * 作用：选车并跟踪。Lost 时 init 最近像心目标；已锁则 update。
 * 输入：装甲、dt。输出：后验写在 target_state_。
 */
void VehicleTracker::step(const std::vector<ArmorBoard>& armors, double dt) {
    if (dt > 0.0)
        time_s_ += dt;
    frame_armors_ = makeObservations(armors);
    std::vector<TrackedArmor*> selected = selectTargetArmors(frame_armors_);
    if (state_ == VehicleTrackState::Lost) {
        if (!selected.empty())
            init(selected.front());
        return;
    }
    update(selected, time_s_);
}

std::vector<Eigen::Vector4d> VehicleTracker::armorXyza(const Eigen::VectorXd* x) const {
    std::vector<Eigen::Vector4d> out;
    if (state_ == VehicleTrackState::Lost && x == nullptr)
        return out;
    const Eigen::VectorXd& xx = x ? *x : target_state_.x;
    if (xx.size() < 4)
        return out;
    for (int i = 0; i < armor_num_; ++i) {
        const Eigen::Vector4d trk = measureArmor(xx, i);  // 跟踪系板心+yaw
        const Eigen::Vector3d cam = trkToCam(trk.head<3>());
        out.emplace_back(cam.x(), cam.y(), cam.z(), yawTrkToCam(trk(3)));
    }
    return out;
}

Eigen::VectorXd VehicleTracker::extrapolate(const Eigen::VectorXd& x, double dt) const {
    if (dt <= 1e-9)
        return x;
    Eigen::VectorXd x_new = x;
    if (x.size() == o_state_dim) {
        x_new(o_yaw) += x(o_vyaw) * dt;
        return x_new;
    }
    if (x.size() < state_dim)
        return x;
    const bool pair_0 = isPair0Phase(confirmed_phase_idx_);
    x_new(xc) += x(vx) * dt;
    x_new(yc) += x(vy) * dt;
    if (pair_0)
        x_new(z1) += x(vz) * dt;
    else
        x_new(z2) += x(vz) * dt;
    x_new(yaw) += x(vyaw) * dt;
    return x_new;
}

Eigen::VectorXd VehicleTracker::extrapolateLead(double dt, double v_scale, double w_scale) const {
    Eigen::VectorXd x = target_state_.x;
    if (x.size() == o_state_dim) {
        x(o_vyaw) *= w_scale;
        return extrapolate(x, dt);
    }
    if (x.size() < state_dim)
        return x;
    x(vx) *= v_scale;
    x(vy) *= v_scale;
    x(vz) *= v_scale;
    x(vyaw) *= w_scale;
    return extrapolate(x, dt);
}

cv::Point3d VehicleTracker::hubMm() const {
    if (state_ == VehicleTrackState::Lost)
        return {0, 0, 0};
    const Eigen::Vector3d cam = trkToCam(Eigen::Vector3d(published_xc_, published_yc_, getZC()));
    return {cam.x() * 1000.0, cam.y() * 1000.0, cam.z() * 1000.0};
}

void VehicleTracker::cameraTwist(double& vx_cam, double& vy_cam, double& vz_cam, double& vyaw_cam) const {
    vx_cam = vy_cam = vz_cam = vyaw_cam = 0.0;
    if (state_ == VehicleTrackState::Lost)
        return;
    if (isOutpostTarget() && target_state_.x.size() >= o_state_dim) {
        vyaw_cam = -published_vyaw_;  // 跟踪系 +yaw 左旋 → 相机 atan2(x,z) 取反
        return;
    }
    if (target_state_.x.size() < state_dim)
        return;
    const double vx_trk = target_state_.x(vx);
    const double vy_trk = target_state_.x(vy);
    const double vz_trk = target_state_.x(vz);
    vx_cam = -vy_trk;
    vy_cam = -vz_trk;
    vz_cam = vx_trk;
    vyaw_cam = -published_vyaw_;
}

void VehicleTracker::radii(double& r_now, double& r_other) const {
    r_now = getCurrentRadius();
    r_other = getOtherRadius();
}
