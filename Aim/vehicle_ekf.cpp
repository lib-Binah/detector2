/**
 * 注释契约：禁止删本文件注释；改圆周公式、量测 ypd、相位匹配或 Q 白噪声时必须改注释。
 * 检测读 ArmorBoard::positionMm / normal_cam / dist_m / number，禁止另算 PnP。
 * 每个函数须写清作用/输入/输出；调用处注明意图。
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
 * 作用：把角折到 (-π, π]。
 * 输入：任意弧度。输出：主值。
 */
double limitRad(double a) {
    while (a > kPi)
        a -= 2.0 * kPi;
    while (a < -kPi)
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
 * 作用：清空跟踪。输入：无。输出：Lost。
 */
void VehicleTracker::reset() {
    state_ = VehicleTrackState::Lost;
    ekf_ready_ = false;
    tracked_id_ = 0;
    detect_count_ = 0;
    temp_lost_count_ = 0;
    last_phase_id_ = 0;
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
        else if (ekf_.data["recent_nis_failures"] >= params_.nis_fail_rate &&
                 ekf_.recent_nis_failures.size() >= 20) {
            // 窗口还太短时不判，避免初始化几帧 NIS 偏大就把车丢掉。
            reset();
        }
    }
}

/**
 * 作用：在 pnp_ok 的板里选最近那辆车（同数字合成一辆），用这块板初始化 EKF。
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
    tracked_type_ = best->type;
    armor_num_ = armorCountFor(*best);
    initEkf(*best);
    return true;
}

/**
 * 作用：板心沿背法线推 r，写入 11 维初值（速度为 0）。
 * 输入：选用板。输出：ekf_ 已 init。
 */
void VehicleTracker::initEkf(const ArmorBoard& armor) {
    const Eigen::Vector3d xyz = armorXyzM(armor);
    const double yaw = armorYaw(armor);
    const double r = params_.default_radius_m;
    // 车心 = 板心 + r * 背法线水平分量（sin/cos 与量测圆周互逆）。
    const double xc = xyz[0] + r * std::sin(yaw);
    const double yc = xyz[1];
    const double zc = xyz[2] + r * std::cos(yaw);
    Eigen::VectorXd x0(11);
    x0 << xc, 0.0, yc, 0.0, zc, 0.0, yaw, 0.0, r, 0.0, 0.0;
    Eigen::VectorXd p0_diag(11);
    p0_diag << 1.0, 64.0, 1.0, 64.0, 1.0, 64.0, 0.4, 100.0, 1.0, 1.0, 1.0;
    Eigen::MatrixXd P0 = p0_diag.asDiagonal();
    auto x_add = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
        Eigen::VectorXd c = a + b;
        c[6] = limitRad(c[6]);
        return c;
    };
    ekf_ = ExtendedKalmanFilter(x0, P0, x_add);
    ekf_ready_ = true;
    last_phase_id_ = 0;
}

/**
 * 作用：predict(dt) 后，把同 ID 的板按相位匹配进 EKF。
 * 输入：本帧装甲、dt。输出：至少更新一块为 true。
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
        updateYpda(a, best_id);
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
 * 输入：板与相位 id。输出：ekf_ 后验。
 */
void VehicleTracker::updateYpda(const ArmorBoard& armor, int id) {
    const Eigen::MatrixXd H = hJacobian(ekf_.x, id);
    const Eigen::Vector3d xyz = armorXyzM(armor);
    const double yaw = armorYaw(armor);
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
 * 作用：相位 id 那块板的相机系坐标（米）。四装甲奇数相位用 r+l 与 yc+h。
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
