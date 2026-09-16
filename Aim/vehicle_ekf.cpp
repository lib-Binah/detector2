/**
 * 禁止删本文件注释；改圆周公式、量测 ypd、相位匹配、Q 白噪声或 init/P0 时必须改注释。
 * 检测读 ArmorBoard::positionMm / normal_cam / dist_m / number，禁止另算 PnP。
 * 每个函数须写清作用/输入/输出；调用处注明意图。
 * 滤波流程对齐同济 Target/Tracker：单板+半径 init，每块板都更新 r，不在首帧量测。
 * r 与 r+l 用 yaml 上限钳位；小于 0.05 仍丢目标。
 * v/ω 用新息自适应 Q + 速度维遗忘 + 首帧位移初值，既少抖又能在机动后几帧跟上。
 * 只有 Tracking 且本帧同时看见两块同号同色板才按（数字,颜色）缓存 r/l/h；
 * reset/seek 不丢缓存，再锁同车时直接用。单板不记，避免 r 与车心糊在一起。
 * 锁车板是相位 0；转到 1/3 时用观测 Y 对 yc±h 定高低，冻车心，不靠 yc 去贴 90° 板。
 * 匹配同号同色即可，不要求大小板类型（90° 邻板经常一大一小）。
 */
#include "vehicle_ekf.h"

#include "Detector/detector_config.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kMaxAbsH = 0.35;   // |h| 上限（米）：装甲高低差不可能到这么大
constexpr double kHSignEps = 0.02;  // |h| 小于此则 yc±h 分不开，用观测相对 yc 定符号

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

/**
 * 作用：只把 vxc/vyc/vzc/vyaw 的方差和相关按 fade 放大，位置/半径不动。
 * 原理：量测多了以后这些维的 P 会收死，匀速假设跟不上加速；每帧略抬高保持增益。
 * 输入：P 为 11×11；fade>1。输出：就地改 P，保持对称。
 */
void fadeVelocityCov(Eigen::MatrixXd& P, double fade) {
    if (fade <= 1.0 || P.rows() < 8 || P.cols() < 8)
        return;
    const int idx[4] = {1, 3, 5, 7};
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
 * 作用：把一对 (r, l) 压进半径上限。r 是 0/2，r+l 是 1/3。
 * 输入：r/l 引用；max_r、max_rl 为上限（米），非正则当 0.5。
 * 输出：就地改 r、l，使 r≤max_r 且 r+l≤max_rl。
 */
void clampRl(double& r, double& l, double max_r, double max_rl) {
    if (!(max_r > 1e-6))
        max_r = 0.5;
    if (!(max_rl > 1e-6))
        max_rl = 0.5;
    r = std::min(r, max_r);
    const double rl = std::min(r + l, max_rl);
    l = rl - r;
}

}  // namespace

/**
 * 作用：YAML aim → 半径初值与上限 / Q / 自适应缩放 / 速度遗忘 / 状态机。
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
    readYaml(n, "q_acc", params_.q_acc);
    readYaml(n, "q_alpha", params_.q_alpha);
    readYaml(n, "q_adapt_ref", params_.q_adapt_ref);
    readYaml(n, "q_adapt_min", params_.q_adapt_min);
    readYaml(n, "q_adapt_max", params_.q_adapt_max);
    readYaml(n, "vel_fade", params_.vel_fade);
    readYaml(n, "max_dt_s", params_.max_dt_s);
    readYaml(n, "min_detect_count", params_.min_detect_count);
    readYaml(n, "max_temp_lost_count", params_.max_temp_lost_count);
    readYaml(n, "nis_fail_rate", params_.nis_fail_rate);
    if (!(params_.max_r_m > 1e-6))
        params_.max_r_m = 0.5;
    if (!(params_.max_r_l_m > 1e-6))
        params_.max_r_l_m = 0.5;
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
    need_vel_bootstrap_ = false;
    odd_h_resolved_ = false;
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
 * 大小板都计入：90° 邻板常不同 type，同帧一大一小才能把 h 从 yc 里拆开。
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
 * 作用：把当前状态的 r、r+l 压进 yaml 上限，并把 |h| 压进装甲高低差上限。
 * 输入：无。输出：改 ekf_.x[8]、x[9]、x[10]。
 */
void VehicleTracker::clampArmorRadii() {
    if (!ekf_ready_ || ekf_.x.size() < 10)
        return;
    double r = ekf_.x[8];
    double l = ekf_.x[9];
    clampRl(r, l, params_.max_r_m, params_.max_r_l_m);  // 0/2 的 r、1/3 的 r+l
    ekf_.x[8] = r;
    ekf_.x[9] = l;
    if (ekf_.x.size() >= 11)
        ekf_.x[10] = std::clamp(ekf_.x[10], -kMaxAbsH, kMaxAbsH);  // 高低差不得用飞点 PnP 撑破
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
        clampArmorRadii();  // r、r+l 压到 yaml 上限，超限不再丢车
        const double r = ekf_.x[8];
        const double r2 = r + ekf_.x[9];
        // 下限仍是 0.05 m：半径瘪掉时圆周退化，丢掉重锁。
        const bool r_ok = r > 0.05;
        const bool l_ok = r2 > 0.05;
        if (!r_ok || !l_ok)
            reset();
        else {
            // 同济 Tracker：窗内失败次数 ≥ 0.4×window_size（默认 40/100）则丢目标。
            const int nis_fail = std::accumulate(ekf_.recent_nis_failures.begin(),
                                                 ekf_.recent_nis_failures.end(), 0);
            if (nis_fail >= params_.nis_fail_rate * static_cast<double>(ekf_.window_size))
                reset();
            else {
                // 只数本帧这辆车：pnp_ok 且数字/颜色对上。大小板都算，两块才能把 r/h 从车心里拆开。
                int n_visible = 0;
                for (const auto& a : armors) {
                    if (!a.pnp_ok)
                        continue;
                    if (a.number == tracked_id_ && a.color() == tracked_color_)
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
    double r = prior ? prior->r_m : params_.default_radius_m;
    double l = prior ? prior->l_m : 0.0;
    double h = prior ? prior->h_m : 0.0;
    clampRl(r, l, params_.max_r_m, params_.max_r_l_m);  // 缓存/初值也不得超过半径上限
    h = std::clamp(h, -kMaxAbsH, kMaxAbsH);
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
    need_vel_bootstrap_ = true;  // 下一帧匹配后用位移填 v/ω
    odd_h_resolved_ = false;     // 尚未用 90° 板对 yc±h 定高低
}

/**
 * 作用：predict(dt) 后，把同 ID 的板按相位匹配进 EKF。
 * 同济 update_target：先预测，再对每块同名板 update；不冻 r。
 * Q 按上一帧新息缩放：匀速缩小、机动放大；预测后再抬 v/ω 的 P。
 */
bool VehicleTracker::updateTarget(const std::vector<ArmorBoard>& armors, double dt) {
    if (!ekf_ready_)
        return false;
    if (dt > 1e-6) {
        const Eigen::MatrixXd F = processF(dt);
        // 4 维量测 χ² 均值约 4。新息小 → 缩 Q 抗 PnP 抖；中等/大 → 放 Q 跟上加速。
        // 极大新息当飞点，不放 Q。丢目标期间把 Q 拉满，再看见时 P 已经够大，v/ω 能立刻改。
        double q_scale = 1.0;
        const double innov = ekf_.last_innov_nis;
        if (state_ == VehicleTrackState::TempLost) {
            q_scale = params_.q_adapt_max;
        } else if (innov > 1e-9 && innov < 40.0 && params_.q_adapt_ref > 1e-6) {
            q_scale = std::clamp(innov / params_.q_adapt_ref, params_.q_adapt_min,
                                 params_.q_adapt_max);
        }
        // 锁车后前几帧强制较大 Q，让位移尽快变成 v/ω，之后再按新息收紧。
        if (need_vel_bootstrap_ || update_count_ < 4)
            q_scale = std::max(q_scale, std::min(8.0, params_.q_adapt_max));
        const Eigen::MatrixXd Q =
            processQ(dt, params_.q_acc * q_scale, params_.q_alpha * q_scale);
        auto f = [&](const Eigen::VectorXd& x) {
            Eigen::VectorXd xp = F * x;
            xp[6] = limitRad(xp[6]);
            return xp;
        };
        ekf_.predict(F, Q, f);  // 匀速+匀角速推到本帧，yaw 折 ±π
        fadeVelocityCov(ekf_.P, params_.vel_fade);  // 只抬 v/ω 不确定度
    }
    // 量测后再用后验位移写 v/ω，避免「先写速度再更新」把同一段位移算两遍。
    const Eigen::VectorXd x_pred = ekf_.x;

    int found = 0;
    const ArmorBoard* unlabeled_best = nullptr;
    double unlabeled_d = std::numeric_limits<double>::infinity();

    for (const auto& a : armors) {
        if (!a.pnp_ok)
            continue;
        if (tracked_id_ > 0) {
            // 同号同色即这辆车；90° 邻板经常 type 不同，不能当另一辆扔掉。
            if (a.number != tracked_id_)
                continue;
            if ((tracked_color_ == 1 || tracked_color_ == 2) && a.color() != tracked_color_)
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
        // 相位 1/3 相对锁车相位 0 差 90°：先用 yc±h 定这块更高还是更低，再冻车心做量测。
        const bool odd90 = (armor_num_ == 4) && (best_id == 1 || best_id == 3);
        if (odd90)
            resolveOddPairHeight(meas_xyz);  // 观测 Y vs yc+h / yc−h → 只改 h
        updateYpda(a, best_id, odd90);  // ypd+装甲 yaw → 后验；1/3 冻 yc，高低差进 h
        last_phase_id_ = best_id;
        ++found;
    }
    if (tracked_id_ == 0 && unlabeled_best) {
        updateYpda(*unlabeled_best, 0);
        last_phase_id_ = 0;
        ++found;
    }
    // 锁车后第一帧量测：v/ω = 后验车心位移 / dt。匹配失败则留到下一帧再试。
    if (need_vel_bootstrap_ && found > 0 && dt > 1e-3)
        bootstrapVelFromArmor(x_pred, dt);
    return found > 0;
}

/**
 * 作用：用「量测后车心 − 预测车心」/ dt 覆盖 v/ω。
 * 锁车时 v=ω=0，Kalman 增益只吃一部分位移；这里写成整段平均速度，下一帧预测就能对齐机动。
 * 输入：x_pred 为预测后、量测前的 11 维；dt>0。输出：改 ekf_.x 的 1/3/5/7；限幅 8 m/s、20 rad/s。
 */
void VehicleTracker::bootstrapVelFromArmor(const Eigen::VectorXd& x_pred, double dt) {
    if (dt < 1e-3 || x_pred.size() < 8 || ekf_.x.size() < 8)
        return;
    ekf_.x[1] = std::clamp((ekf_.x[0] - x_pred[0]) / dt, -8.0, 8.0);
    ekf_.x[3] = std::clamp((ekf_.x[2] - x_pred[2]) / dt, -8.0, 8.0);
    ekf_.x[5] = std::clamp((ekf_.x[4] - x_pred[4]) / dt, -8.0, 8.0);
    ekf_.x[7] = std::clamp(limitRad(ekf_.x[6] - x_pred[6]) / dt, -20.0, 20.0);
    need_vel_bootstrap_ = false;
}

/**
 * 作用：第一次匹配到相位 1/3 时，比较观测 Y 与 yc+h、yc−h，判定 90° 板比锁车板更高还是更低。
 * 原理：单次量测 y=yc+h，yc 与 h 线性相关；若把残差分给 yc，旋转中心会整段跳。
 * |h| 太小则 ± 重合，h←观测Y−yc；否则只在 yc−h 明显更贴时翻符号。不改 yc。
 * 输入：该板 PnP 米制。输出：改 x[10]；已决议则返回。
 */
void VehicleTracker::resolveOddPairHeight(const Eigen::Vector3d& meas_xyz) {
    if (odd_h_resolved_ || !ekf_ready_ || ekf_.x.size() < 11)
        return;
    const double yc = ekf_.x[2];
    double h = ekf_.x[10];
    const double y_obs = meas_xyz[1];  // 相机 Y 向下，与 yc、h 同一轴，读 PnP tvec
    const double err_plus = std::abs(y_obs - (yc + h));
    const double err_minus = std::abs(y_obs - (yc - h));
    if (std::abs(h) < kHSignEps) {
        // ±h 分不开：高低差就是这块板相对锁车高度的观测差
        h = y_obs - yc;
    } else if (err_minus + kHSignEps < err_plus) {
        // yc−h 更贴观测：缓存/滤波把 90° 对的高低搞反了，只翻符号
        h = -h;
    }
    ekf_.x[10] = std::clamp(h, -kMaxAbsH, kMaxAbsH);
    if (ekf_.P.rows() >= 11 && ekf_.P.cols() >= 11)
        ekf_.P(10, 10) = std::max(ekf_.P(10, 10), 1e-3);  // 刚钉住的 h 仍允许后续微调
    odd_h_resolved_ = true;
}

/**
 * 作用：ypda 量测更新。R 随视线偏角和距离变大（同济 log 式）。
 * 同济每块板都更新 r；前后板 H 的 l/h 列为 0，侧面才改长短轴。
 * freeze_hub_y：90° 板更新时把 yc/vyc 列置零并写回更新前的值，残差只进 h。
 * 输入：板、相位 id、是否冻车心高度。输出：ekf_ 后验。
 */
void VehicleTracker::updateYpda(const ArmorBoard& armor, int id, bool freeze_hub_y) {
    Eigen::MatrixXd H = hJacobian(ekf_.x, id);
    const double yc_keep = ekf_.x.size() > 3 ? ekf_.x[2] : 0.0;
    const double vyc_keep = ekf_.x.size() > 3 ? ekf_.x[3] : 0.0;
    if (freeze_hub_y && H.cols() >= 4) {
        H.col(2).setZero();  // 冻 yc：90° 板高度差不得拽旋转中心
        H.col(3).setZero();  // 冻 vyc，避免车心被「拽了一下」后往下冲
    }
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
    if (freeze_hub_y && ekf_.x.size() > 3) {
        ekf_.x[2] = yc_keep;  // 写回：Joseph 交叉项仍可能微移 yc
        ekf_.x[3] = vyc_keep;
    }
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
