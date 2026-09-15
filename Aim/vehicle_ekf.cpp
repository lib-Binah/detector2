/**
 * 禁止删本文件注释；改圆周公式、量测 ypd、相位匹配、Q 白噪声或 init/P0 时必须改注释。
 * 检测读 ArmorBoard::positionMm / normal_cam / dist_m / number，禁止另算 PnP。
 * 每个函数须写清作用/输入/输出；调用处注明意图。
 * 滤波流程对齐同济 Target/Tracker：单板+半径 init，每块板都更新 r，不在首帧量测。
 * 只有 Tracking 且本帧同时看见两块同号同色板才按（数字,颜色）缓存 r/l/h；
 * reset/seek 不丢缓存，再锁同车时直接用。单板不记，避免 r 与车心糊在一起。
 */
#include "vehicle_ekf.h"

#include "Detector/detector_config.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace {

constexpr double kPi = 3.14159265358979323846;

/**
 * 作用：把角折到 (-π, π]（同济 tools::limit_rad：>π 减、≤−π 加）。
 * 输入：任意弧度。输出：主值。
 */
double limitRad(double a) {
    while (a > kPi)
        a -= 2.0 * kPi;
    while (a <= -kPi)
        a += 2.0 * kPi;
    return a;
}

/**
 * 作用：板背法线在水平面（相机 XZ）的 yaw，与 overlay 箭头同一 normal_cam。
 * 输入：pnp_ok 的板。输出：atan2(nx, nz)。
 */
double armorYaw(const ArmorBoard& a) {
    return std::atan2(a.normal_cam[0], a.normal_cam[2]);
}

/**
 * 作用：板心米制。输入：装甲板。输出：tvec/1000。
 */
Eigen::Vector3d armorXyzM(const ArmorBoard& a) {
    const cv::Point3d p = a.positionMm();
    return {p.x / 1000.0, p.y / 1000.0, p.z / 1000.0};
}

/**
 * 作用：视线 ypd。相机 X 右 Y 下 Z 前：yaw=atan2(x,z)，pitch=atan2(y,hypot(x,z))。
 * 对应同济世界系 xyz2ypd 的 atan2(y,x) / atan2(z, hypot(x,y))，轴已换成相机水平面 XZ。
 * 输入：相机系米。输出：(yaw, pitch, distance)。
 */
Eigen::Vector3d xyz2ypd(const Eigen::Vector3d& xyz) {
    const double x = xyz[0];
    const double y = xyz[1];
    const double z = xyz[2];
    const double dist = xyz.norm();
    return {std::atan2(x, z), std::atan2(y, std::hypot(x, z)), dist};
}

/**
 * 作用：ypd 对 xyz 的 3×3 雅可比，给 H = H_ypd · H_xyza。
 * 输入：板心米。输出：∂(yaw,pitch,d)/∂(x,y,z)。
 */
Eigen::Matrix3d xyz2ypdJacobian(const Eigen::Vector3d& xyz) {
    const double x = xyz[0];
    const double y = xyz[1];
    const double z = xyz[2];
    const double rho2 = x * x + z * z;
    const double u = std::sqrt(std::max(rho2, 1e-12));
    const double d2 = std::max(x * x + y * y + z * z, 1e-12);
    const double d = std::sqrt(d2);
    Eigen::Matrix3d J;
    J << z / rho2, 0.0, -x / rho2, -y * x / (d2 * u), u / d2, -y * z / (d2 * u), x / d, y / d, z / d;
    return J;
}

/**
 * 作用：匀速平移 + 匀角速的 F。
 * 输入：dt 秒。输出：11×11。
 */
Eigen::MatrixXd processF(double dt) {
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(11, 11);
    F(0, 1) = dt;
    F(2, 3) = dt;
    F(4, 5) = dt;
    F(6, 7) = dt;
    return F;
}

/**
 * 作用：分段白噪声 Q（同济 rlabbe 公式）。r/l/h 过程噪声为 0，只能量测改半径。
 * 输入：dt、平移/角加速度方差。输出：11×11。
 */
Eigen::MatrixXd processQ(double dt, double v1, double v2) {
    const double a = dt * dt * dt * dt / 4.0;
    const double b = dt * dt * dt / 2.0;
    const double c = dt * dt;
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(11, 11);
    Q(0, 0) = a * v1;
    Q(0, 1) = b * v1;
    Q(1, 0) = b * v1;
    Q(1, 1) = c * v1;
    Q(2, 2) = a * v1;
    Q(2, 3) = b * v1;
    Q(3, 2) = b * v1;
    Q(3, 3) = c * v1;
    Q(4, 4) = a * v1;
    Q(4, 5) = b * v1;
    Q(5, 4) = b * v1;
    Q(5, 5) = c * v1;
    Q(6, 6) = a * v2;
    Q(6, 7) = b * v2;
    Q(7, 6) = b * v2;
    Q(7, 7) = c * v2;
    return Q;
}

}  // namespace

/**
 * 作用：YAML aim → 半径/Q/状态机。输入：path。输出：打开成功为 true。
 */
bool VehicleTracker::loadConfig(const std::string& path) {
    cv::FileStorage fs;
    if (!openDetectorYaml(path, fs))
        return false;
    const cv::FileNode n = fs["aim"];
    readYaml(n, "default_radius_m", params_.default_radius_m);
    readYaml(n, "q_acc", params_.q_acc);
    readYaml(n, "q_alpha", params_.q_alpha);
    readYaml(n, "max_dt_s", params_.max_dt_s);
    readYaml(n, "min_detect_count", params_.min_detect_count);
    readYaml(n, "max_temp_lost_count", params_.max_temp_lost_count);
    readYaml(n, "nis_fail_rate", params_.nis_fail_rate);
    return true;
}

/**
 * 作用：清空跟踪。不丢同号同色几何缓存。
 * 输入：无。输出：Lost。
 */
void VehicleTracker::reset() {
    state_ = VehicleTrackState::Lost;
    ekf_ready_ = false;
    tracked_id_ = 0;
    tracked_color_ = 0;
    detect_count_ = 0;
    temp_lost_count_ = 0;
    last_phase_id_ = 0;
    update_count_ = 0;
}

/**
 * 作用：平衡步兵（大板 3/4/5）两块装甲，其余四块。输入：板。输出：2 或 4。
 */
int VehicleTracker::armorCountFor(const ArmorBoard& armor) {
    if (armor.type == ArmorType::LARGE && (armor.number == 3 || armor.number == 4 || armor.number == 5))
        return 2;
    return 4;
}

/**
 * 作用：本帧同时看到两块同号同色板、且已 Tracking 时，把 r/l/h 记到（数字,颜色）下。
 * 单板不写：正对一块时 r 与车心深度分不清。输入：本帧该车 pnp_ok 板数。输出：满足时写入 geom_priors_。
 */
void VehicleTracker::maybeSaveGeomPrior(int n_visible_plates) {
    if (state_ != VehicleTrackState::Tracking || !ekf_ready_)
        return;
    if (n_visible_plates < 2)
        return;  // 必须同帧两块板才能把半径从车心里拆出来
    if (tracked_id_ <= 0 || (tracked_color_ != 1 && tracked_color_ != 2))
        return;
    VehicleGeomPrior p;
    p.r_m = ekf_.x[8];
    p.l_m = ekf_.x[9];
    p.h_m = ekf_.x[10];
    geom_priors_[{tracked_id_, tracked_color_}] = p;
}

/**
 * 作用：本帧更新整车 EKF。输入：装甲与 dt。输出：无。
 */
void VehicleTracker::step(const std::vector<ArmorBoard>& armors, double dt) {
    if (state_ != VehicleTrackState::Lost && dt > params_.max_dt_s)
        reset();

    bool found = false;
    if (state_ == VehicleTrackState::Lost)
        found = setTarget(armors);
    else
        found = updateTarget(armors, dt);

    stateMachine(found);

    if (state_ != VehicleTrackState::Lost && ekf_ready_) {
        const double r = ekf_.x[8];
        const double r2 = r + ekf_.x[9];
        const bool r_ok = r > 0.05 && r < 0.5;
        const bool l_ok = r2 > 0.05 && r2 < 0.5;
        if (!r_ok || !l_ok)
            reset();
        else {
            // 同济 Tracker：窗内失败次数 ≥ 0.4×window_size（默认 40/100）则丢目标。
            const int nis_fail = std::accumulate(ekf_.recent_nis_failures.begin(),
                                                 ekf_.recent_nis_failures.end(), 0);
            if (nis_fail >= params_.nis_fail_rate * static_cast<double>(ekf_.window_size))
                reset();
            else {
                // 只数本帧这辆车：pnp_ok 且数字/类型/颜色都对上。两块才能把 r 从车心里拆开。
                int n_visible = 0;
                for (const auto& a : armors) {
                    if (!a.pnp_ok)
                        continue;
                    if (a.number == tracked_id_ && a.type == tracked_type_ &&
                        a.color() == tracked_color_)
                        ++n_visible;
                }
                maybeSaveGeomPrior(n_visible);  // 同帧两块同号同色板才记 r/l/h
            }
        }
    }
}

/**
 * 作用：在 pnp_ok 的板里选最近那辆车（同数字合成一辆），只 init、不量测。
 * 同济 set_target：构造 Target 后返回，第一帧不做 update。
 * 输入：本帧装甲。输出：true 表示已 init。
 */
bool VehicleTracker::setTarget(const std::vector<ArmorBoard>& armors) {
    const ArmorBoard* best = nullptr;
    double best_d = std::numeric_limits<double>::infinity();
    for (const auto& a : armors) {
        if (!a.pnp_ok || a.dist_m <= 0.0)
            continue;
        if (a.dist_m < best_d) {
            best_d = a.dist_m;
            best = &a;
        }
    }
    if (!best)
        return false;
    tracked_id_ = best->number;
    tracked_color_ = best->color();  // 与灯条 color 同源：1 红 / 2 蓝
    tracked_type_ = best->type;
    armor_num_ = armorCountFor(*best);
    const VehicleGeomPrior* prior = nullptr;
    VehicleGeomPrior held;
    if (tracked_id_ > 0 && (tracked_color_ == 1 || tracked_color_ == 2)) {
        const auto it = geom_priors_.find({tracked_id_, tracked_color_});
        if (it != geom_priors_.end()) {
            held = it->second;
            prior = &held;  // 同号同色已收敛过：r/l/h 用缓存，车心仍按本帧板推
        }
    }
    initEkf(*best, prior);  // 无缓存则 yaml 半径、l=h=0；有缓存则几何几乎不再重估
    return true;
}

/**
 * 作用：写入 11 维初值。车心 = 本帧板心沿背法线推 r。
 * 无 prior：r=yaml，l=h=0，P0 的 r/l/h=1（同济步兵）。
 * 有 prior：r/l/h 用缓存，P0 的 r/l/h 很小，避免再花一段收敛。
 * 输入：锁车那块板；prior 可空。输出：ekf_ 就绪。
 */
void VehicleTracker::initEkf(const ArmorBoard& armor, const VehicleGeomPrior* prior) {
    const Eigen::Vector3d xyz = armorXyzM(armor);
    const double yaw = armorYaw(armor);
    const double r = prior ? prior->r_m : params_.default_radius_m;
    const double l = prior ? prior->l_m : 0.0;
    const double h = prior ? prior->h_m : 0.0;
    const double p_rlh = prior ? 1e-4 : 1.0;  // 有缓存则几乎锁死几何，只估本帧位姿/速度
    // 相机系：xc=xa+r sinθ，zc=za+r cosθ，对应同济世界系 xc=xa+r cosθ、yc=ya+r sinθ。
    const double xc = xyz[0] + r * std::sin(yaw);
    const double yc = xyz[1];
    const double zc = xyz[2] + r * std::cos(yaw);

    Eigen::VectorXd x0(11);
    x0 << xc, 0.0, yc, 0.0, zc, 0.0, yaw, 0.0, r, l, h;
    Eigen::VectorXd p0_diag(11);
    p0_diag << 1.0, 64.0, 1.0, 64.0, 1.0, 64.0, 0.4, 100.0, p_rlh, p_rlh, p_rlh;
    Eigen::MatrixXd P0 = p0_diag.asDiagonal();
    auto x_add = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
        Eigen::VectorXd c = a + b;
        c[6] = limitRad(c[6]);
        return c;
    };
    ekf_ = ExtendedKalmanFilter(x0, P0, x_add);
    ekf_ready_ = true;
    last_phase_id_ = 0;
    update_count_ = 0;
}

/**
 * 作用：predict(dt) 后，把同 ID 的板按相位匹配进 EKF。
 * 同济 update_target：先预测，再对每块同名板 update；不冻 r。
 */
bool VehicleTracker::updateTarget(const std::vector<ArmorBoard>& armors, double dt) {
    if (!ekf_ready_)
        return false;
    if (dt > 1e-6) {
        const Eigen::MatrixXd F = processF(dt);
        const Eigen::MatrixXd Q = processQ(dt, params_.q_acc, params_.q_alpha);
        auto f = [&](const Eigen::VectorXd& x) {
            Eigen::VectorXd xp = F * x;
            xp[6] = limitRad(xp[6]);
            return xp;
        };
        ekf_.predict(F, Q, f);  // 匀速+匀角速推到本帧，yaw 折 ±π
    }

    int found = 0;
    const ArmorBoard* unlabeled_best = nullptr;
    double unlabeled_d = std::numeric_limits<double>::infinity();

    for (const auto& a : armors) {
        if (!a.pnp_ok)
            continue;
        if (tracked_id_ > 0) {
            if (a.number != tracked_id_ || a.type != tracked_type_)
                continue;
        } else {
            if (a.number != 0)
                continue;
            if (a.dist_m < unlabeled_d) {
                unlabeled_d = a.dist_m;
                unlabeled_best = &a;
            }
            continue;
        }

        const auto xyza = armorXyza();
        int best_id = 0;
        double best_err = std::numeric_limits<double>::infinity();
        // 同济：先按预测距离排，只在最近 3 块里比 yaw+视线角，避免背面相位抢匹配。
        std::vector<std::pair<double, int>> dist_id;
        for (int i = 0; i < armor_num_; ++i) {
            const Eigen::Vector3d ypd = xyz2ypd(xyza[static_cast<std::size_t>(i)].head<3>());
            dist_id.push_back({ypd[2], i});
        }
        std::sort(dist_id.begin(), dist_id.end());
        const int nchk = std::min(3, armor_num_);
        const Eigen::Vector3d meas_xyz = armorXyzM(a);
        const Eigen::Vector3d meas_ypd = xyz2ypd(meas_xyz);
        const double meas_yaw = armorYaw(a);
        for (int k = 0; k < nchk; ++k) {
            const int id = dist_id[static_cast<std::size_t>(k)].second;
            const double pred_yaw = xyza[static_cast<std::size_t>(id)][3];
            const Eigen::Vector3d pred_ypd = xyz2ypd(xyza[static_cast<std::size_t>(id)].head<3>());
            const double err = std::abs(limitRad(meas_yaw - pred_yaw)) +
                               std::abs(limitRad(meas_ypd[0] - pred_ypd[0]));
            if (err < best_err) {
                best_err = err;
                best_id = id;
            }
        }
        updateYpda(a, best_id);  // ypd+装甲 yaw → 后验；r/l/h 按 H 列更新
        last_phase_id_ = best_id;
        ++found;
    }
    if (tracked_id_ == 0 && unlabeled_best) {
        updateYpda(*unlabeled_best, 0);
        last_phase_id_ = 0;
        ++found;
    }
    return found > 0;
}

/**
 * 作用：ypda 量测更新。R 随视线偏角和距离变大（同济 log 式）。
 * 同济每块板都更新 r；前后板 H 的 l/h 列为 0，侧面才改长短轴。
 * 输入：板、相位 id。输出：ekf_ 后验。
 */
void VehicleTracker::updateYpda(const ArmorBoard& armor, int id) {
    Eigen::MatrixXd H = hJacobian(ekf_.x, id);
    const Eigen::Vector3d xyz = armorXyzM(armor);
    const double yaw = armorYaw(armor);
    // 同济 center_yaw=atan2(y,x)；相机水平面用 atan2(x,z)。
    const double center_yaw = std::atan2(xyz[0], xyz[2]);
    const double delta = limitRad(yaw - center_yaw);
    Eigen::VectorXd R_dig(4);
    R_dig << 4e-3, 4e-3, std::log(std::abs(delta) + 1.0) + 1.0,
        std::log(xyz.norm() + 1.0) / 200.0 + 9e-2;
    const Eigen::MatrixXd R = R_dig.asDiagonal();

    auto h = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        const Eigen::Vector3d ypd = xyz2ypd(armorXyz(x, id));
        Eigen::VectorXd z(4);
        z << ypd[0], ypd[1], ypd[2], limitRad(x[6] + id * 2.0 * kPi / armor_num_);
        return z;
    };
    auto z_sub = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
        Eigen::VectorXd c = a - b;
        c[0] = limitRad(c[0]);
        c[1] = limitRad(c[1]);
        c[3] = limitRad(c[3]);
        return c;
    };
    const Eigen::Vector3d ypd = xyz2ypd(xyz);
    Eigen::VectorXd z(4);
    z << ypd[0], ypd[1], ypd[2], yaw;
    ekf_.update(z, H, R, h, z_sub);  // ypd+装甲 yaw → 后验整车状态
    ++update_count_;
}

/**
 * 作用：同济状态机 lost/detecting/tracking/temp_lost。
 * 输入：本帧是否匹配到板。输出：改 state_。
 */
void VehicleTracker::stateMachine(bool found) {
    if (state_ == VehicleTrackState::Lost) {
        if (!found)
            return;
        state_ = VehicleTrackState::Detecting;
        detect_count_ = 1;
    } else if (state_ == VehicleTrackState::Detecting) {
        if (found) {
            ++detect_count_;
            if (detect_count_ >= params_.min_detect_count)
                state_ = VehicleTrackState::Tracking;
        } else {
            detect_count_ = 0;
            reset();
        }
    } else if (state_ == VehicleTrackState::Tracking) {
        if (!found) {
            temp_lost_count_ = 1;
            state_ = VehicleTrackState::TempLost;
        }
    } else if (state_ == VehicleTrackState::TempLost) {
        if (found) {
            state_ = VehicleTrackState::Tracking;
            temp_lost_count_ = 0;
        } else {
            ++temp_lost_count_;
            if (temp_lost_count_ > params_.max_temp_lost_count)
                reset();
        }
    }
}

/**
 * 作用：相位 id 那块板的相机系坐标（米）。
 * 同济：id=0/2 前后用 r、yc；id=1/3 左右用 r+l、yc+h。对侧相同。
 * 输入：状态与相位。输出：xyz。
 */
Eigen::Vector3d VehicleTracker::armorXyz(const Eigen::VectorXd& x, int id) const {
    const double angle = limitRad(x[6] + id * 2.0 * kPi / armor_num_);
    const bool use_lh = (armor_num_ == 4) && (id == 1 || id == 3);
    const double r = use_lh ? x[8] + x[9] : x[8];
    const double xa = x[0] - r * std::sin(angle);
    const double ya = use_lh ? x[2] + x[10] : x[2];
    const double za = x[4] - r * std::cos(angle);
    return {xa, ya, za};
}

/**
 * 作用：H = H_ypda(4×4) · H_xyza(4×11)。输入：状态、相位。输出：4×11。
 */
Eigen::MatrixXd VehicleTracker::hJacobian(const Eigen::VectorXd& x, int id) const {
    const double angle = limitRad(x[6] + id * 2.0 * kPi / armor_num_);
    const bool use_lh = (armor_num_ == 4) && (id == 1 || id == 3);
    const double r = use_lh ? x[8] + x[9] : x[8];
    const double dx_da = -r * std::cos(angle);
    const double dz_da = r * std::sin(angle);
    const double dx_dr = -std::sin(angle);
    const double dz_dr = -std::cos(angle);
    const double dx_dl = use_lh ? -std::sin(angle) : 0.0;
    const double dz_dl = use_lh ? -std::cos(angle) : 0.0;
    const double dy_dh = use_lh ? 1.0 : 0.0;

    Eigen::MatrixXd H_xyza = Eigen::MatrixXd::Zero(4, 11);
    H_xyza(0, 0) = 1.0;
    H_xyza(0, 6) = dx_da;
    H_xyza(0, 8) = dx_dr;
    H_xyza(0, 9) = dx_dl;
    H_xyza(1, 2) = 1.0;
    H_xyza(1, 10) = dy_dh;
    H_xyza(2, 4) = 1.0;
    H_xyza(2, 6) = dz_da;
    H_xyza(2, 8) = dz_dr;
    H_xyza(2, 9) = dz_dl;
    H_xyza(3, 6) = 1.0;

    const Eigen::Vector3d xyz = armorXyz(x, id);
    const Eigen::Matrix3d H_ypd = xyz2ypdJacobian(xyz);
    Eigen::MatrixXd H_ypda = Eigen::MatrixXd::Zero(4, 4);
    H_ypda.block<3, 3>(0, 0) = H_ypd;
    H_ypda(3, 3) = 1.0;
    return H_ypda * H_xyza;
}

/**
 * 作用：n 块板的 xyz+yaw。输入：可选状态。输出：列表。
 */
std::vector<Eigen::Vector4d> VehicleTracker::armorXyza(const Eigen::VectorXd* x) const {
    std::vector<Eigen::Vector4d> out;
    if (!ekf_ready_ && x == nullptr)
        return out;
    const Eigen::VectorXd& xx = x ? *x : ekf_.x;
    for (int i = 0; i < armor_num_; ++i) {
        const Eigen::Vector3d p = armorXyz(xx, i);
        const double ang = limitRad(xx[6] + i * 2.0 * kPi / armor_num_);
        out.push_back(Eigen::Vector4d(p[0], p[1], p[2], ang));
    }
    return out;
}

/**
 * 作用：F 外推，不碰 P。输入：x、dt。输出：新状态。
 */
Eigen::VectorXd VehicleTracker::extrapolate(const Eigen::VectorXd& x, double dt) const {
    if (dt <= 1e-9)
        return x;
    Eigen::VectorXd xp = processF(dt) * x;
    xp[6] = limitRad(xp[6]);
    return xp;
}

/**
 * 作用：车心 mm。输入：无。输出：相机系。
 */
cv::Point3d VehicleTracker::hubMm() const {
    if (!ekf_ready_)
        return {0, 0, 0};
    return {ekf_.x[0] * 1000.0, ekf_.x[2] * 1000.0, ekf_.x[4] * 1000.0};
}
