/**
 * 注释契约：禁止删本文件注释；改增益公式、Joseph 式、NLL 或部分更新时必须改注释。
 * 每个函数须写清作用/输入/输出；调用处注明送什么、得到什么。
 *
 * lovod EKF 核：预测用 f/F/Q(dt)，选相用高斯 NLL，更新用 Joseph 保持 P 对称正定。
 * yaw 残差必须 wrap 到 ±π。S 用 LDLT，不正定就放弃这一次评价。
 */
#include "ekf.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace {

constexpr double kPi = 3.14159265358979323846;

/**
 * 作用：把角折到 (-π, π]。输入：任意弧度。输出：主值。
 */
double warp2Pi(double angle) {
    while (angle > kPi)
        angle -= 2.0 * kPi;
    while (angle < -kPi)
        angle += 2.0 * kPi;
    return angle;
}

}  // namespace

/**
 * 作用：检查函数指针并记下维数。输入：过程/量测及其雅可比、Q、R。输出：I 已分配。
 */
EKFCore::EKFCore(const ProcessNoiseFunc& process_noise_func, const ProcessFunc& process_func,
                 const MeasurementFunc& measurement_func, const ProcessJacFunc& process_jacobian,
                 const MeasurementJacFunc& measurement_jacobian, Eigen::MatrixXd Q,
                 Eigen::MatrixXd R)
    : process_noise_func_(process_noise_func),
      process_func_(process_func),
      measurement_func_(measurement_func),
      process_jacobian_(process_jacobian),
      measurement_jacobian_(measurement_jacobian),
      Q_(std::move(Q)),
      R_(std::move(R)) {
    if (!process_func_ || !measurement_func_ || !process_jacobian_ || !measurement_jacobian_)
        throw std::invalid_argument("EKFCore: process/measurement functors must be valid.");
    state_dim_ = static_cast<int>(Q_.rows());
    measurement_dim_ = static_cast<int>(R_.rows());
    I_ = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
}

/**
 * 作用：写入初值。输入：x/P。输出：is_initialized=true。
 */
void EKFCore::init(const EKFCore::State& init_state) {
    state_ = init_state;
    is_initialized_ = true;
}

/**
 * 作用：x←f(x,dt)，P←F P Fᵀ + Q(dt)。输入：dt。输出：预测后的 state_。
 */
void EKFCore::predict(double dt) {
    if (!is_initialized_)
        throw std::runtime_error("EKFCore::predict: not initialized.");
    const Eigen::MatrixXd F = process_jacobian_(state_.x, dt);  // ∂f/∂x
    state_.x = process_func_(state_.x, dt);                     // 匀速+匀角速（由模型决定谁积 z）
    const Eigen::MatrixXd Q = process_noise_func_ ? process_noise_func_(dt) : Q_;
    state_.P = F * state_.P * F.transpose() + Q;
}

/**
 * 作用：标准 Kalman 增益。输入：P、H、R。输出：K。
 */
Eigen::MatrixXd EKFCore::computeKalmanGain(const Eigen::MatrixXd& P, const Eigen::MatrixXd& H,
                                           const Eigen::MatrixXd& R) const {
    const Eigen::MatrixXd S = H * P * H.transpose() + R;
    return P * H.transpose() * S.inverse();
}

/**
 * 作用：逐相位 evaluateMeasurement，NLL 最小者胜。
 * 输入：z、相位表、参考相位、R、可选只比两相。输出：PhaseMatch。
 */
EKFCore::PhaseMatch EKFCore::selectBestPhase(const Eigen::VectorXd& z,
                                             const std::vector<double>& phase_offsets,
                                             int reference_phase_idx,
                                             const Eigen::MatrixXd& measurement_noise,
                                             int candidate_phase) {
    const int phase_count = static_cast<int>(phase_offsets.size());
    double best_nll = std::numeric_limits<double>::max();
    int best_phase_idx = reference_phase_idx >= 0 ? reference_phase_idx : 0;
    int best_phase_distance = 0;

    for (int idx = 0; idx < phase_count; ++idx) {
        if (candidate_phase >= 0 && idx != reference_phase_idx && idx != candidate_phase)
            continue;
        const MeasurementEvaluation evaluation =
            evaluateMeasurement(z, state_.x, idx, measurement_noise);  // 该相位的 NIS/NLL
        int phase_distance = 0;
        if (reference_phase_idx >= 0) {
            const int phase_diff = std::abs(idx - reference_phase_idx);
            phase_distance = std::min(phase_diff, phase_count - phase_diff);
        }
        if (evaluation.nll < best_nll) {
            best_nll = evaluation.nll;
            best_phase_idx = idx;
            best_phase_distance = phase_distance;
        }
    }

    PhaseMatch result;
    result.phase_idx = best_phase_idx;
    result.nll = best_nll;
    result.phase_distance = best_phase_distance;
    if (phase_count == 4 && reference_phase_idx >= 0 && best_phase_idx != reference_phase_idx) {
        // 四板圆周：0/2 一对、1/3 一对，换对才叫 change（要切 z/r）
        result.change = ((best_phase_idx % 2) != (reference_phase_idx % 2));
    } else if (phase_count == 3 && reference_phase_idx >= 0) {
        result.change = (best_phase_idx != reference_phase_idx);
    }
    return result;
}

/**
 * 作用：用默认 R_ 评价。输入：z、假设 x、相位。输出：evaluation。
 */
EKFCore::MeasurementEvaluation EKFCore::evaluateMeasurement(const Eigen::VectorXd& z,
                                                            const Eigen::VectorXd& x_hypothesis,
                                                            int phase_idx) {
    return evaluateMeasurement(z, x_hypothesis, phase_idx, R_);
}

/**
 * 作用：y=z-h(x)，yaw 折角；S=HPHᵀ+R 对称化后 LDLT 求 NIS 与 log|S|。
 * 输入：z、x_hypothesis（H 在此线性化）、相位、R。输出：NIS=yᵀS⁻¹y，NLL=½(NIS+log|S|)。
 */
EKFCore::MeasurementEvaluation EKFCore::evaluateMeasurement(const Eigen::VectorXd& z,
                                                            const Eigen::VectorXd& x_hypothesis,
                                                            int phase_idx,
                                                            const Eigen::MatrixXd& measurement_noise) {
    MeasurementEvaluation evaluation;
    evaluation.H = measurement_jacobian_(x_hypothesis, phase_idx);  // ∂h/∂x 在假设点
    const Eigen::VectorXd z_pred = measurement_func_(x_hypothesis, phase_idx);
    evaluation.y = z - z_pred;
    if (evaluation.y.size() >= 4)
        evaluation.y(3) = warp2Pi(evaluation.y(3));  // 只折装甲 yaw

    if (measurement_noise.rows() != evaluation.y.size() ||
        measurement_noise.cols() != evaluation.y.size() || !measurement_noise.allFinite())
        return evaluation;

    evaluation.R = measurement_noise;
    evaluation.S = evaluation.H * state_.P * evaluation.H.transpose() + evaluation.R;
    evaluation.S = 0.5 * (evaluation.S + evaluation.S.transpose());

    Eigen::LDLT<Eigen::MatrixXd> ldlt(evaluation.S);
    if (ldlt.info() != Eigen::Success || (ldlt.vectorD().array() <= 0.0).any())
        return evaluation;

    const Eigen::VectorXd solved = ldlt.solve(evaluation.y);
    if (ldlt.info() != Eigen::Success || !solved.allFinite())
        return evaluation;

    evaluation.nis = evaluation.y.dot(solved);
    if (evaluation.nis < 0.0) {
        evaluation.nis = std::numeric_limits<double>::max();
        return evaluation;
    }
    const double log_det_s = ldlt.vectorD().array().log().sum();
    evaluation.nll = 0.5 * (evaluation.nis + log_det_s);
    return evaluation;
}

/**
 * 作用：块对角拼 R、竖拼 H/y，再对联合新息算 NIS/NLL。
 * 输入：各板 evaluation。输出：batch evaluation。
 */
EKFCore::MeasurementEvaluation EKFCore::evaluateBatch(
    const std::vector<MeasurementEvaluation>& observations) {
    MeasurementEvaluation evaluation;
    if (observations.empty())
        return evaluation;

    int measurement_rows = 0;
    for (const auto& observation : observations) {
        const int rows = static_cast<int>(observation.y.size());
        if (rows <= 0 || observation.H.rows() != rows || observation.H.cols() != state_dim_ ||
            observation.R.rows() != rows || observation.R.cols() != rows ||
            !observation.R.allFinite() || !observation.y.allFinite() || !observation.H.allFinite())
            return evaluation;
        measurement_rows += rows;
    }

    evaluation.H = Eigen::MatrixXd::Zero(measurement_rows, state_dim_);
    evaluation.y = Eigen::VectorXd::Zero(measurement_rows);
    evaluation.R = Eigen::MatrixXd::Zero(measurement_rows, measurement_rows);
    int offset = 0;
    for (const auto& observation : observations) {
        const int rows = static_cast<int>(observation.y.size());
        evaluation.H.middleRows(offset, rows) = observation.H;
        evaluation.y.segment(offset, rows) = observation.y;
        evaluation.R.block(offset, offset, rows, rows) = observation.R;
        offset += rows;
    }

    evaluation.S = evaluation.H * state_.P * evaluation.H.transpose() + evaluation.R;
    evaluation.S = 0.5 * (evaluation.S + evaluation.S.transpose());
    Eigen::LDLT<Eigen::MatrixXd> ldlt(evaluation.S);
    if (ldlt.info() != Eigen::Success || (ldlt.vectorD().array() <= 0.0).any())
        return evaluation;
    const Eigen::VectorXd solved = ldlt.solve(evaluation.y);
    if (ldlt.info() != Eigen::Success || !solved.allFinite())
        return evaluation;
    evaluation.nis = evaluation.y.dot(solved);
    if (evaluation.nis < 0.0) {
        evaluation.nis = std::numeric_limits<double>::max();
        return evaluation;
    }
    const double log_det_s = ldlt.vectorD().array().log().sum();
    evaluation.nll = 0.5 * (evaluation.nis + log_det_s);
    return evaluation;
}

/**
 * 作用：x←x+Ky，P←(I-KH)P(I-KH)ᵀ+KRKᵀ。输入：evaluation。输出：后验。
 */
void EKFCore::update(const MeasurementEvaluation& evaluation) {
    if (!is_initialized_)
        throw std::runtime_error("EKFCore::update: not initialized.");
    const Eigen::MatrixXd& H = evaluation.H;
    const Eigen::MatrixXd K = computeKalmanGain(state_.P, H, evaluation.R);  // 全维增益
    state_.x = state_.x + K * evaluation.y;
    const Eigen::MatrixXd A = I_ - K * H;
    state_.P = A * state_.P * A.transpose() + K * evaluation.R * K.transpose();
    state_.P = 0.5 * (state_.P + state_.P.transpose());
}

/**
 * 作用：先把非 active 维从 P/H 里解耦，再 Joseph；写回后这些维只保留原方差。
 * 输入：evaluation、active 下标。输出：后验。
 */
void EKFCore::updatePartial(const MeasurementEvaluation& evaluation,
                            const std::vector<int>& active_indices) {
    if (!is_initialized_)
        throw std::runtime_error("EKFCore::updatePartial: not initialized.");
    if (active_indices.empty())
        return;

    Eigen::MatrixXd H = evaluation.H;
    Eigen::MatrixXd P_masked = state_.P;
    std::vector<bool> active_mask(static_cast<std::size_t>(state_dim_), false);
    for (const int idx : active_indices) {
        if (idx >= 0 && idx < state_dim_)
            active_mask[static_cast<std::size_t>(idx)] = true;
    }
    for (int i = 0; i < state_dim_; ++i) {
        if (active_mask[static_cast<std::size_t>(i)])
            continue;
        const double var = P_masked(i, i);
        P_masked.row(i).setZero();
        P_masked.col(i).setZero();
        P_masked(i, i) = var;
        H.col(i).setZero();  // 非本对 z/r 不进增益
    }

    const Eigen::MatrixXd K = computeKalmanGain(P_masked, H, evaluation.R);
    state_.x += K * evaluation.y;
    const Eigen::MatrixXd A = I_ - K * H;
    state_.P = A * P_masked * A.transpose() + K * evaluation.R * K.transpose();
    for (int i = 0; i < state_dim_; ++i) {
        if (active_mask[static_cast<std::size_t>(i)])
            continue;
        const double var = P_masked(i, i);
        state_.P.row(i).setZero();
        state_.P.col(i).setZero();
        state_.P(i, i) = var;
    }
    state_.P = 0.5 * (state_.P + state_.P.transpose());
}
