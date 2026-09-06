#ifndef NUMBER_CLASSIFIER_H
#define NUMBER_CLASSIFIER_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "ArmorBoard.h"

/**
 * 注释契约：禁止为“精简”删除本文件注释；改输入尺寸、归一化、类别映射或拒识规则时必须改注释。
 * CNN 输入：20×28 单通道（默认 Otsu 二值，可在 yaml number.use_binary 关掉）、
 * (x/255-0.5)/0.5、NCHW [1,1,28,20]。类别 0~7 → 数字 1~8，类别 8 → other。
 * 每个函数须写清作用、输入、输出；调用处也要注明。
 *
 * 数字分类器：透视把两灯条之间的数字区域拉成 20×28，可选 Otsu 二值，再跑 ONNX CNN。
 * 透视几何跟 clone_detector NumberClassifier::extractNumbers：灯条端点映到
 * 高 28、灯条长 12 的条带，再从中间裁 20×28。
 * number_img 必须与送进 preprocess 的是同一张图；左上角缩略图只读这个字段。
 */
class NumberClassifier {
public:
    static constexpr int kOtherClass = 8;     // 第 9 类：不是 1~8
    static constexpr int kInputW = 20;        // 与训练 resize 宽一致
    static constexpr int kInputH = 28;        // 与训练 resize 高一致
    // softmax 最大类概率必须严格大于该值才算装甲板（默认 80%）。改门槛时同步改本注释。
    static constexpr float kMinConfidence = 0.80f;

    /**
     * 作用：空分类器，尚未读模型；classify 会原样留下装甲板。
     * 输入：无。输出：loaded()==false 的实例。
     */
    NumberClassifier() = default;

    /**
     * 作用：从 ONNX 读 CNN（图里已含 Softmax）。
     * 输入：model_path 为 digit_cnn_fixed.onnx；读失败则 loaded()==false。
     * 输出：true 表示 net 非空，之后可以 extractNumbers/classify。
     */
    bool loadModel(const std::string& model_path);

    /**
     * 作用：模型是否已加载。
     * 输入：无。输出：true 表示可以 classify。
     */
    bool loaded() const { return loaded_; }

    /**
     * 作用：写入透视条带尺寸（须与训练裁图一致）。
     * 输入：灯条竖向像素、条带高、小板/大板条带宽。输出：无；之后 extractNumbers 读这些成员。
     */
    void setWarpParams(int light_length, int warp_height, int small_w, int large_w);

    /**
     * 作用：从 YAML 的 number 节点读透视尺寸与 use_binary。
     * 输入：path。输出：true 表示文件打开成功。
     */
    bool loadConfig(const std::string& path);

    /**
     * 作用：把每块板的数字区域透视成 20×28，可选 Otsu 二值，写入 ArmorBoard::number_img。
     * 输入：bgr 为当前帧（与灯条坐标同一分辨率）；armors 就地改。
     * 输出：无返回值；失败的板 number_img 为空。use_binary_ 为 true 时 number_img 为 0/255。
     */
    void extractNumbers(const cv::Mat& bgr, std::vector<ArmorBoard>& armors) const;

    /**
     * 作用：对 number_img 做 CNN 前向；other / 置信度未严格大于门槛 / 大板非 1 或小板为 1 的板丢掉。
     * 输入：armors 须已 extractNumbers；threshold 为置信度门槛（须 confidence > threshold）。
     * 输出：就地删除未通过的板；通过的写入 number/confidence/classification_result。
     */
    void classify(std::vector<ArmorBoard>& armors, float threshold);

private:
    /**
     * 作用：number_img → blob（二值或灰度都走同一套归一化）。
     * 输入：单通道图，通常已是 20×28。输出：CV_32F blob [1,1,28,20]。
     */
    cv::Mat preprocess(const cv::Mat& gray) const;

    cv::dnn::Net net_;
    bool loaded_ = false;
    bool use_binary_ = true;        // true：Otsu 二值写入 number_img 并送 CNN；yaml number.use_binary
    int light_length_ = 12;         // 透视条带里灯条竖向像素，与 yaml number.light_length 同源
    int warp_height_ = 28;          // 条带总高
    int small_armor_width_ = 32;    // 小板条带宽
    int large_armor_width_ = 54;    // 大板条带宽
};

#endif  // NUMBER_CLASSIFIER_H
