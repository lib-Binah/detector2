#ifndef ARMOR_BOARD_H
#define ARMOR_BOARD_H

#include <opencv2/opencv.hpp>
#include <string>

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
 * 原理：灯条四顶点 → 上/下端点（边中点）→ 中心距/长度比/倾角 → 大小板 → 数字 ROI → PnP。
 * 灯条端点约定：上 = (pts[0]+pts[1])/2，下 = (pts[2]+pts[3])/2。
 * 这与 clone_detector 的 Light::top/bottom 同一几何；PnP、数字透视、画框都读这四个点。
 *
 * light_ratio：min(Light::height)/max(Light::height)，构造时写入。
 * center_distance：两灯 Light::center() 距离 / 平均 Light::height（无量纲，远近近似不变）。
 * angle：两中心连线相对水平的倾角（度）。
 * light_angle_diff：|left.angle - right.angle|，读的就是灯条 Light::angle（右上→右下相对竖直）。
 * 绝对值大于 max_light_angle_diff（默认 6°）不当装甲板。筛选和 overlay 都读这个字段。
 * 大小板：center_distance 落入小板或大板区间；分界读 min_large_center_distance。
 * PnP 物点：小板 135×55 mm、大板 225×55 mm（灯条矩形，不是整板外扩后的数字面）。
 */
enum class ArmorType { SMALL = 0, LARGE = 1 };

class ArmorBoard {
public:
    Light left;   // 中心 x 较小的灯条
    Light right;  // 中心 x 较大的灯条
    cv::Point2f center;  // (left.center()+right.center())/2；overlay 锚点

    // 灯条上/下端点。改 pts 顶点顺序时必须同步改 topOf/bottomOf。
    cv::Point2f left_top;
    cv::Point2f left_bottom;
    cv::Point2f right_top;
    cv::Point2f right_bottom;

    ArmorType type = ArmorType::SMALL;

    // 下面四个量构造时写一次；配对门槛和 overlay 都读它们。
    double light_ratio = 0.0;       // 短灯高 / 长灯高，读 Light::height
    double center_distance = 0.0;   // 中心距 / 平均 Light::height
    double angle = 0.0;             // 中心连线相对水平的倾角（度）
    double light_angle_diff = 0.0;  // |Light::angle 左 - Light::angle 右|（度）；>6° 丢掉

    int number = 0;                 // 识别标签 1~8；0=未识别 / 拒识
    float confidence = 0.f;         // CNN softmax 最大类概率
    std::string classification_result;  // overlay 用，例如 "3:_92.0%"
    cv::Mat number_img;             // 20×28 灰度，CNN 输入；overlay 可把这张图画出来

    cv::Mat rvec;                   // PnP 旋转向量
    cv::Mat tvec;                   // PnP 平移，单位 mm（与物点一致）
    double dist_m = -1.0;           // ||tvec||/1000；<=0 表示未解或失败
    bool pnp_ok = false;

    /**
     * 作用：空板，字段保持默认。
     * 输入：无。输出：未填充的 ArmorBoard，检测路径不会用这个。
     */
    ArmorBoard() = default;

    /**
     * 作用：由两根灯条填几何（左右按中心 x 排序），并写入 light_ratio/center_distance/angle/light_angle_diff。
     * 输入：两根已通过灯条筛选的 Light；不必预先分左右。
     * 输出：left/right、四端点、center、四个几何量；type/number/PnP 由检测器后写。
     */
    ArmorBoard(const Light& a, const Light& b);

    /**
     * 作用：灯条上端点 = 左上与右上的中点，与 clone_detector Light::top 同一几何。
     * 输入：读 Light::pts。输出：图像坐标。
     */
    static cv::Point2f topOf(const Light& light);

    /**
     * 作用：灯条下端点 = 左下与右下的中点，与 clone_detector Light::bottom 同一几何。
     * 输入：读 Light::pts。输出：图像坐标。
     */
    static cv::Point2f bottomOf(const Light& light);

    /**
     * 作用：板颜色跟左右灯条走（配对时已要求同色）。
     * 输入：无。输出：1 红 / 2 蓝 / 其他未知。
     */
    int color() const { return left.color; }

    /**
     * 作用：把装甲板四边形 + X 画到图上；小板金黄，大板绿。
     * 输入：img 为要画的 BGR 图（就地修改）。输出：无返回值。
     */
    void draw_armor(cv::Mat& img) const;

    /**
     * 作用：用四灯条端点做 IPPE PnP，得到相机系平移（mm）和距离（m）。
     * 输入：无（读 type、四端点和写死的相机内参）。
     * 输出：true 表示解出且 tvec.z>0；写入 rvec/tvec/dist_m/pnp_ok。
     */
    bool solvePnP();
};

#endif  // ARMOR_BOARD_H
