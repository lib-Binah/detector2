/**
 * 注释契约：禁止删本文件注释；改增益公式、Joseph 式或 NIS 门槛时必须改注释。
 * 每个函数须写清作用/输入/输出；调用处注明送什么、得到什么。
 *
 * 同济 EKF 核（sp_vision_25 tools::ExtendedKalmanFilter）：预测用 F/Q，
 * 更新用 Joseph 稳定形式，后验残差做 NIS，状态修正做 NEES。
 */
#include "ekf.h"

#include <numeric>
#include <utility>

namespace {

// 同济源码按「4 维 χ² 0.95」写的门槛，数值是 0.711（标准分位实际是 9.488）。
constexpr double kNisThreshold = 0.711;
constexpr double kNeesThreshold = 0.711;

}  // namespace

/**
 * 作用：写入 x0/P0 和角度加法。
 * 输入：x0、P0、x_add。输出：I 与 NIS/NEES 统计清零。
 */
ExtendedKalmanFilter::ExtendedKalmanFilter(const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0,
                                           VectorAdd x_add)
    : x(x0), P(P0), I_(Eigen::MatrixXd::Identity(x0.rows(), x0.rows())), x_add_(std::move(x_add)) {
    data["residual_yaw"] = 0.0;
    data["residual_pitch"] = 0.0;
    data["residual_distance"] = 0.0;
    data["residual_angle"] = 0.0;
    data["nis"] = 0.0;
    data["nees"] = 0.0;
    data["nis_fail"] = 0.0;
    data["nees_fail"] = 0.0;
    data["recent_nis_failures"] = 0.0;
}

/**
 * 作用：线性预测。输入：F、Q。输出：Fx。
 */
Eigen::VectorXd ExtendedKalmanFilter::predict(const Eigen::MatrixXd& F, const Eigen::MatrixXd& Q) {
    return predict(F, Q, [&](const Eigen::VectorXd& x_in) { return F * x_in; });
}

/**
 * 作用：x←f(x)，P←FPFᵀ+Q。输入：F、Q、f。输出：预测状态。
 */
Eigen::VectorXd ExtendedKalmanFilter::predict(const Eigen::MatrixXd& F, const Eigen::MatrixXd& Q,
                                              const VectorFunc& f) {
    P = F * P * F.transpose() + Q;
    x = f(x);
    return x;
}

/**
 * 作用：线性 h(x)=Hx 的更新。输入：z、H、R、z_subtract。输出：后验 x。
 */
Eigen::VectorXd ExtendedKalmanFilter::update(const Eigen::VectorXd& z, const Eigen::MatrixXd& H,
                                             const Eigen::MatrixXd& R, VectorSub z_subtract) {
    return update(
        z, H, R, [&](const Eigen::VectorXd& x_in) { return H * x_in; }, std::move(z_subtract));
}

/**
 * 作用：K=PHᵀ(HPHᵀ+R)⁻¹，x←x⊕K(z-h(x))，P 用 Joseph 式。
 * 输入：z 为量测；h 为非线性量测；角度残差必须走 z_subtract。
 * 输出：后验 x；NIS/NEES 写入 data，供跟踪器按窗口失败次数重置。
 */
Eigen::VectorXd ExtendedKalmanFilter::update(const Eigen::VectorXd& z, const Eigen::MatrixXd& H,
                                             const Eigen::MatrixXd& R, const VectorFunc& h,
                                             VectorSub z_subtract) {
    const Eigen::VectorXd x_prior = x;
    Eigen::MatrixXd K = P * H.transpose() * (H * P * H.transpose() + R).inverse();

    // Joseph：P=(I-KH)P(I-KH)ᵀ+KRKᵀ，比 (I-KH)P 更能保持对称正定（同济/rlabbe）。
    P = (I_ - K * H) * P * (I_ - K * H).transpose() + K * R * K.transpose();
    x = x_add_(x, K * z_subtract(z, h(x)));

    const Eigen::VectorXd residual = z_subtract(z, h(x));
    Eigen::MatrixXd S = H * P * H.transpose() + R;
    const double nis = residual.transpose() * S.inverse() * residual;
    const double nees = (x - x_prior).transpose() * P.inverse() * (x - x_prior);

    if (nis > kNisThreshold) {
        ++nis_count_;
        data["nis_fail"] = 1.0;
    }
    if (nees > kNeesThreshold) {
        ++nees_count_;
        data["nees_fail"] = 1.0;
    }
    ++total_count_;
    last_nis = nis;

    recent_nis_failures.push_back(nis > kNisThreshold ? 1 : 0);
    if (recent_nis_failures.size() > window_size)
        recent_nis_failures.pop_front();
    const int recent_fail =
        std::accumulate(recent_nis_failures.begin(), recent_nis_failures.end(), 0);
    const double recent_rate =
        static_cast<double>(recent_fail) / static_cast<double>(recent_nis_failures.size());

    data["residual_yaw"] = residual[0];
    data["residual_pitch"] = residual[1];
    data["residual_distance"] = residual[2];
    data["residual_angle"] = residual[3];
    data["nis"] = nis;
    data["nees"] = nees;
    data["recent_nis_failures"] = recent_rate;
    return x;
}
