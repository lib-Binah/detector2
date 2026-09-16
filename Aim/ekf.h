#ifndef AIM_EKF_H
#define AIM_EKF_H

#include <Eigen/Dense>
#include <functional>
#include <limits>
#include <vector>

/**
 * 注释契约：本文件说明 lovod EKFCore 为什么这样预测/选相/更新。禁止删注释；
 * 改 Joseph 式、NLL 相位、部分更新或残差折角时必须改注释。
 * 每个函数须写清作用、输入、输出。
 *
 * 通用扩展卡尔曼滤波（对齐 auto_aim_by_lovod/rm_target_tracker EKFCore）。
 * 过程/量测由整车模型注入；本类只做 x=f(x,dt)、P=FPFᵀ+Q、按相位算新息、Joseph 更新。
 * 量测是笛卡尔 (x,y,z,yaw)，yaw 残差折到 ±π，禁止直接 a-b 把 359° 当成大残差。
 * 选相用高斯 NLL = ½(yᵀS⁻¹y + log|S|)，不是同济的 ypd+卡方窗。
 * updatePartial 把非本对 z/r 从增益里摘掉，避免一块板改另一对半径/高度。
 */
class EKFCore {
public:
    struct State {
        State() = default;
        /**
         * 作用：按维数分配零均值、单位协方差。
         * 输入：dim 为状态维。输出：x=0，P=I。
         */
        explicit State(int dim) {
            x = Eigen::VectorXd::Zero(dim);
            P = Eigen::MatrixXd::Identity(dim, dim);
        }
        Eigen::VectorXd x;
        Eigen::MatrixXd P;
    };

    struct PhaseMatch {
        int phase_idx{-1};
        double nll{std::numeric_limits<double>::max()};
        int phase_distance{0};
        bool change{false};  // 四板：奇偶对换；三板：相位下标变了
    };

    struct MeasurementEvaluation {
        Eigen::MatrixXd H;
        Eigen::VectorXd y;
        Eigen::MatrixXd R;
        Eigen::MatrixXd S;
        double nis{std::numeric_limits<double>::max()};
        double nll{std::numeric_limits<double>::max()};
    };

    using ProcessNoiseFunc = std::function<Eigen::MatrixXd(double)>;
    using ProcessFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd&, double)>;
    using ProcessJacFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd&, double)>;
    using MeasurementFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd&, int)>;
    using MeasurementJacFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd&, int)>;

    EKFCore() = default;

    /**
     * 作用：注入过程/量测及其雅可比，以及对角 Q/R 底噪。
     * 输入：五个函数指针都必须有效；Q 行数=状态维，R 行数=量测维。
     * 输出：可 init；函数为空会抛异常。
     */
    EKFCore(const ProcessNoiseFunc& process_noise_func, const ProcessFunc& process_func,
            const MeasurementFunc& measurement_func, const ProcessJacFunc& process_jacobian,
            const MeasurementJacFunc& measurement_jacobian, Eigen::MatrixXd Q, Eigen::MatrixXd R);

    /**
     * 作用：写入初值均值/协方差。输入：init_state。输出：之后允许 predict/update。
     */
    void init(const State& init_state);

    /**
     * 作用：非线性过程 x←f(x,dt)，P←FPFᵀ+Q(dt)。
     * 输入：dt 秒。输出：改 state_；未 init 抛异常。
     */
    void predict(double dt);

    /**
     * 作用：K=PHᵀ(HPHᵀ+R)⁻¹。输入：当前 P、H、R。输出：增益。
     */
    Eigen::MatrixXd computeKalmanGain(const Eigen::MatrixXd& P, const Eigen::MatrixXd& H,
                                      const Eigen::MatrixXd& R) const;

    /**
     * 作用：对每个相位算 NLL，挑最小的；四板时只在邻相/当前相里比（candidate≥0）。
     * 输入：z 为量测；phase_offsets 为各相位相对车 yaw 的偏置；reference 为当前锁定相位；
     *       measurement_noise 为该根的 R；candidate_phase&lt;0 表示搜全部相位。
     * 输出：最佳相位、NLL、与参考相位的环距；change 表示换了对侧/换了前哨相位。
     */
    PhaseMatch selectBestPhase(const Eigen::VectorXd& z, const std::vector<double>& phase_offsets,
                               int reference_phase_idx, const Eigen::MatrixXd& measurement_noise,
                               int candidate_phase = -1);

    /**
     * 作用：在假设状态上算残差/NIS/NLL，S 仍用当前滤波器 P（lovod 原逻辑）。
     * 输入：z、x_hypothesis、相位、R。输出：evaluation；S 不正定则 nis/nll 保持 max。
     */
    MeasurementEvaluation evaluateMeasurement(const Eigen::VectorXd& z,
                                              const Eigen::VectorXd& x_hypothesis, int phase_idx);
    MeasurementEvaluation evaluateMeasurement(const Eigen::VectorXd& z,
                                              const Eigen::VectorXd& x_hypothesis, int phase_idx,
                                              const Eigen::MatrixXd& measurement_noise);

    /**
     * 作用：把多块板的 (H,y,R) 拼成一块联合量测再算 NIS/NLL。
     * 输入：observations 为已 evaluate 的各板。输出：堆叠后的 evaluation；任一块非法则 nis=max。
     */
    MeasurementEvaluation evaluateBatch(const std::vector<MeasurementEvaluation>& observations);

    /**
     * 作用：Joseph 更新，整维都吃增益。输入：evaluate 结果。输出：后验 x/P。
     */
    void update(const MeasurementEvaluation& evaluation);

    /**
     * 作用：只让 active_indices 那些状态吃增益，其余维保持方差、交叉项清零。
     * 原理：本帧只看见一对装甲时，另一对的 z/r 不应被牵着走。
     * 输入：evaluation、要更新的下标。输出：后验；空列表直接返回。
     */
    void updatePartial(const MeasurementEvaluation& evaluation,
                       const std::vector<int>& active_indices);

    bool isInitialized() const { return is_initialized_; }
    const State& getState() const { return state_; }
    State& getMutableState() { return state_; }
    void setState(const State& state) { state_ = state; }
    void setNoise(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R) {
        Q_ = Q;
        R_ = R;
    }
    void setProcessNoiseFunc(const ProcessNoiseFunc& f) { process_noise_func_ = f; }

private:
    State state_;
    ProcessNoiseFunc process_noise_func_;
    ProcessFunc process_func_;
    MeasurementFunc measurement_func_;
    ProcessJacFunc process_jacobian_;
    MeasurementJacFunc measurement_jacobian_;

    Eigen::MatrixXd I_;
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd R_;

    int state_dim_{0};
    int measurement_dim_{0};
    bool is_initialized_{false};
};

#endif  // AIM_EKF_H
