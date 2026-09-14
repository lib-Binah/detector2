#ifndef ARMOR_BOARD_H
#define ARMOR_BOARD_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "Light.h"

/**
 * 注释契约（给后续修改者 / 智能体）：
 * 本文件注释说明装甲板几何表示的作用与原理。禁止为“精简代码”删除注释；
 * 改字段含义、灯条端点定义、大小板判定、物点尺寸或 PnP 点序时，必须同步改注释。
 * 配对筛选和 overlay 显示必须读同一个属性或同一个函数（计算方式相同）；
 * 禁止筛选用 light_ratio、显示另算 height1/height2。改公式只改本类，两边一起变。
 * 每个成员函数须写清作用、输入、输出；调用处也要注明。
 *
 * 由左右两根同色灯条构成的一块装甲板。
 * 原理：灯条四顶点 → 灯条外侧四角 → 判定大小板后沿左右边把高度缩放到实际板面 → 数字 ROI（仍用灯条四角）→ PnP。
 * 灯条矩形：宽=左右外侧距（小板 140、大板 235），高=灯条长 55mm。
 * 实际板面：宽不变，高缩放到小板 125、大板 127。left_top 等在 isArmor 通过后写入缩放结果。
 * 画框、PnP、重投影、预测板都读缩放后的四角 + 实际物点高。数字透视读 Light::pts 灯条外侧。
 * containLight 仍用 topOf/bottomOf（灯条上下边中点）。
 * Light::pts：[0]左上 [1]右上 [2]左下 [3]右下。
 *
 * light_ratio：min(Light::height)/max(Light::height)，构造时写入。
 * center_distance：两灯 Light::center() 距离 / 平均 Light::height（无量纲，远近近似不变）。
 * angle：两中心连线相对水平的倾角（度）。
 * light_angle_diff：|left.angle - right.angle|，读的就是灯条 Light::angle（右上→右下相对竖直）。
 * 绝对值大于 max_light_angle_diff（配置默认 10°）不当装甲板。筛选和 overlay 都读这个字段。
 * 大小板：center_distance 落入小板或大板区间；分界读 min_large_center_distance。
 * PnP 物点：宽与灯条外侧距相同，高为实际板面（默认小板 140×125、大板 235×127），yaml pnp 节覆盖。
 * 物系：板在 z=0，X 右、Y 下（与图像一致），+Z = X×Y。
 * 俯仰跳变：IPPE 双根俯仰接近相反。同济方案是把安装俯仰当已知量，只搜 yaw。
 * armor_pitch_deg（默认 15°）= 板面朝斜上的安装角；锁进 R 后背离相机的法线略朝下，不会再上下翻。
 * 流程：IPPE 给初值 → 固定 pitch 网格搜 yaw → 固定 R 最小二乘求 t。
 * normal_cam：锁俯仰后的相机系单位法线，远离相机（n_z≥0）；overlay / 车心必须读它。
 */
enum class ArmorType { SMALL = 0, LARGE = 1 };

class ArmorBoard {
public:
    Light left;   // 中心 x 较小的灯条
    Light right;  // 中心 x 较大的灯条
    cv::Point2f center;  // (left.center()+right.center())/2；overlay 锚点

    // 装甲四角：判定 type 后把灯条外侧矩形沿左右边缩放到实际板高。改 Light::pts 顺序时同步改构造初值。
    cv::Point2f left_top;
    cv::Point2f left_bottom;
    cv::Point2f right_top;
    cv::Point2f right_bottom;

    ArmorType type = ArmorType::SMALL;

    // 下面四个量构造时写一次；配对门槛和 overlay 都读它们。
    double light_ratio = 0.0;       // 短灯高 / 长灯高，读 Light::height
    double center_distance = 0.0;   // 中心距 / 平均 Light::height
    double angle = 0.0;             // 中心连线相对水平的倾角（度）
    double light_angle_diff = 0.0;  // |Light::angle 左 - Light::angle 右|（度）；大于配置 max_light_angle_diff 丢掉

    int number = 0;                 // 识别标签 1~8；0=未识别 / 拒识
    float confidence = 0.f;         // CNN softmax 最大类概率
    std::string classification_result;  // overlay 用，例如 "3:_92.0%"
    cv::Mat number_img;             // 单通道数字 ROI：大板 20×28，小板 26×28（宽+30%），与 CNN 同源；overlay 左上角贴这张

    cv::Mat rvec;                   // 锁俯仰后的 Rodrigues 向量（IPPE 只作 yaw 初值）
    cv::Mat tvec;                   // PnP 平移，单位 mm（与物点一致）
    cv::Vec3d normal_cam{0, 0, 0};  // 锁俯仰后的相机系单位法线，远离相机（n_z≥0）
    double dist_m = -1.0;           // ||tvec||/1000；<=0 表示未解或失败
    double reproj_error_px = -1.0;  // 四点平均重投影误差（像素）；<0 表示未解
    bool pnp_ok = false;

    /**
     * 作用：空板，字段保持默认。
     * 输入：无。输出：未填充的 ArmorBoard，检测路径不会用这个。
     */
    ArmorBoard() = default;

    /**
     * 作用：由两根灯条填几何（左右按中心 x 排序），并写入 light_ratio/center_distance/angle/light_angle_diff。
     * 输入：两根已通过灯条筛选的 Light；不必预先分左右。
     * 输出：left/right、灯条外侧四角（尚未按板高缩放）、center、四个几何量；type 由 isArmor 写入后再缩放。
     */
    ArmorBoard(const Light& a, const Light& b);

    /**
     * 作用：灯条上边中点（左上与右上平均），给 containLight 量灯条竖直范围，不是装甲 PnP 角点。
     * 输入：读 Light::pts。输出：图像坐标。
     */
    static cv::Point2f topOf(const Light& light);

    /**
     * 作用：灯条下边中点（左下与右下平均），给 containLight 量灯条竖直范围，不是装甲 PnP 角点。
     * 输入：读 Light::pts。输出：图像坐标。
     */
    static cv::Point2f bottomOf(const Light& light);

    /**
     * 作用：板颜色跟左右灯条走（配对时已要求同色）。
     * 输入：无。输出：1 红 / 2 蓝 / 其他未知。
     */
    int color() const { return left.color; }

    /**
     * 作用：把装甲板四边形 + X 画到图上；小板金黄，大板绿。边框 1px，便于和重投影轮廓对比。
     * 输入：img 为要画的 BGR 图（就地修改）。输出：无返回值。
     */
    void draw_armor(cv::Mat& img) const;

    /**
     * 作用：把板法线画成箭头，方向是远离相机（读 normal_cam，与 solvePnP 同一向量）。
     * 原理：相机系 tvec 为起点，沿 normal_cam 走约 5 倍板高（远距离再按深度加长），
     * 用与 PnP 相同的 K/D 投到像素。方向是远离相机（锁 15° 安装俯仰后略朝下）。
     * 输入：img 为要画的 BGR 图（就地修改）。pnp_ok 为 false 时不画。
     * 输出：无返回值。
     */
    void draw_normal(cv::Mat& img) const;

    /**
     * 作用：IPPE 初值 + 同济锁俯仰搜 yaw，得到相机系位姿和远离相机的法线。
     * 输入：无（读 type、四端点、内参、pnp.armor_pitch_deg）。
     * 输出：true 表示解出且深度合法；写入 rvec/tvec/normal_cam/dist_m/reproj_error_px/pnp_ok。
     */
    bool solvePnP();

    /**
     * 作用：灯条外侧矩形沿左右边把高度缩放到实际板面（宽不变）。画框/PnP/重投影都读缩放后的四角。
     * 输入：type 已写入；left_top 等仍是灯条外侧。输出：四角变成实际板在图像上的四边形。只调用一次。
     */
    void scaleToActualArmorSize();

    /**
     * 作用：板心在相机系的坐标（mm），与 tvec、弹道解算同一原点。
     * 输入：无。输出：pnp_ok 时为 tvec；否则 (0,0,0)。
     */
    cv::Point3d positionMm() const;

    /**
     * 作用：给 Aim 投影用，与 PnP 同一套内参。
     * 输入：无。输出：3×3 相机矩阵常引用。
     */
    static const cv::Mat& cameraMatrix() { return camera_matrix_; }

    /**
     * 作用：给 Aim 投影用，与 PnP 同一套畸变。
     * 输入：无。输出：1×5 畸变常引用。
     */
    static const cv::Mat& distCoeffs() { return dist_coeffs_; }

    /**
     * 作用：EKF 预测板心+yaw → 板面四角像素，物点尺寸和锁俯仰 R 必须与 solvePnP 同一套。
     * 原理：物点左下/左上/右上/右下（mm，Y 向下），R = 锁安装俯仰的 yaw 旋转，t = 板心。
     * 输入：center_mm 与 positionMm 同一相机系；yaw_rad 与 atan2(normal.x, normal.z) 同一水平角；type 定大小板。
     * 输出：4 个像素点（左下、左上、右上、右下）；任一点在相机后方则空。
     */
    static std::vector<cv::Point2f> projectArmorQuad(const cv::Point3d& center_mm, double yaw_rad,
                                                     ArmorType type);

    /**
     * 作用：本板 PnP 位姿把同一套物点投回像素，与 reproj_error_px / solvePnP 同源。
     * 输入：无（读 type、rvec、tvec、内参）。输出：左下、左上、右上、右下；失败或点在相机后为空。
     */
    std::vector<cv::Point2f> reprojectQuad() const;

    /**
     * 作用：从 YAML 的 pnp / camera 节点写入物点尺寸和内参。
     * 输入：path 为 detector_params.yaml。输出：true 表示文件打开成功；缺键保持默认。
     */
    static bool loadPnpConfig(const std::string& path);

private:
    // 物点 mm。高是实际板面；灯条高 light_height_mm_ 只用来把像点沿竖直边缩放。loadPnpConfig 覆盖。
    static float small_width_mm_;
    static float small_height_mm_;
    static float large_width_mm_;
    static float large_height_mm_;
    static float light_height_mm_;  // 左右灯条矩形高度（mm），默认 55；像点缩放比 = 板高/本值
    static float armor_pitch_deg_;  // 板面朝斜上的安装角；锁进 R，避免 IPPE 俯仰跳变
    static cv::Mat camera_matrix_;
    static cv::Mat dist_coeffs_;
    /**
     * 作用：实际板面物点四角（mm）。solvePnP、reprojectQuad、projectArmorQuad 必须读这里。
     * 输入：大小板。输出：左下、左上、右上、右下，Y 向下，z=0；高为 125/127 不是灯条 55。
     */
    static std::vector<cv::Point3f> objectCornersMm(ArmorType type);
};

#endif  // ARMOR_BOARD_H
