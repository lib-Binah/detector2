#ifndef AIM_EKF_H
#define AIM_EKF_H

#include <Eigen/Dense>
#include <deque>
#include <functional>
#include <map>

/**
 * 注释契约：本文件说明同济式 EKF 为什么这样更新。禁止删注释；改 Joseph 式、
 * NIS 窗口或残差折角时必须改注释。每个函数须写清作用、输入、输出。
 *
 * 通用扩展卡尔曼滤波（对齐 TongjiSuperPower/sp_vision_25 tools::ExtendedKalmanFilter）。
 * 过程/量测函数由整车模型注入；本类只做 x=f(x)、P=FPFᵀ+Q、Joseph 更新、NIS/NEES。
 * yaw 等角度维用 x_add / z_subtract 折到 ±π，禁止直接 a-b 把 359° 当成大残差。
 * NIS 门槛与同济源码一致为 0.711（他们按 4 维 χ² 0.95 标注，数值本身过严）。
 * last_innov_nis 是更新前的新息卡方（yᵀS⁻¹y，S=HPHᵀ+R），给下一帧放大/缩小 Q：
 * 机动时新息大 → Q 变大几帧内跟上 v/ω；飞点时新息过大则抬 R 不追这一帧。
 */
class ExtendedKalmanFilter {
public:
    using VectorAdd = std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>;
    using VectorFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;
    using VectorSub = std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>;

    Eigen::VectorXd x;
    Eigen::MatrixXd P;
    std::map<std::string, double> data;  // 同济：残差、nis/nees、窗口失败率
    std::deque<int> recent_nis_failures{0};
    std::size_t window_size = 100;
    double last_nis = 0.0;
    double last_innov_nis = 0.0;  // 更新前新息卡方；自适应 Q 读这个，不要读后验 last_nis

    ExtendedKalmanFilter() = default;

    /**
     * 作用：用初值均值/协方差构造滤波器。
     * 输入：x0 为状态；P0 为协方差；x_add 为状态加法（默认向量加，角度维可折 ±π）。
     * 输出：x/P 已写入；尚未 predict。
     */
    ExtendedKalmanFilter(const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0,
                         VectorAdd x_add = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
                             return a + b;
                         });

    /**
     * 作用：线性过程 x←Fx，P←FPFᵀ+Q。
     * 输入：F 为过程雅可比；Q 为过程噪声。输出：预测后的 x。
     */
    Eigen::VectorXd predict(const Eigen::MatrixXd& F, const Eigen::MatrixXd& Q);

    /**
     * 作用：非线性过程：x←f(x)，P 仍用 F 线性化传播。
     * 输入：F、Q 同上；f 为非线性过程（同济用来对 yaw 做 limit_rad）。
     * 输出：预测后的 x。
     */
    Eigen::VectorXd predict(const Eigen::MatrixXd& F, const Eigen::MatrixXd& Q, const VectorFunc& f);

    /**
     * 作用：线性量测更新。
     * 输入：z 为量测；H、R 为雅可比和量测噪声；z_subtract 为残差（角度维折 ±π）。
     * 输出：更新后的 x。
     */
    Eigen::VectorXd update(const Eigen::VectorXd& z, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R,
                           VectorSub z_subtract = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
                               return a - b;
                           });

    /**
     * 作用：非线性量测更新（Joseph 形式保持 P 对称正定），并记新息/后验 NIS、NEES。
     * 输入：z、H、R、h(x)、z_subtract。输出：更新后的 x；data["innov_nis"] 为更新前卡方。
     * 新息大于 χ²_4≈16 时把 R 按比例抬高再算 K，PnP 飞点不把 v/ω 拽飞。
     */
    Eigen::VectorXd update(const Eigen::VectorXd& z, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R,
                           const VectorFunc& h, VectorSub z_subtract);

private:
    Eigen::MatrixXd I_;
    VectorAdd x_add_;
    int nees_count_ = 0;
    int nis_count_ = 0;
    int total_count_ = 0;
};

#endif  // AIM_EKF_H
