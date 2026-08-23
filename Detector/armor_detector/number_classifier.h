#ifndef NUMBER_CLASSIFIER_H
#define NUMBER_CLASSIFIER_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "ArmorBoard.h"

/**
 * 注释契约：禁止为“精简”删除本文件注释；改输入尺寸、归一化、类别映射或拒识规则时必须改注释。
 * CNN 输入必须与 PythonProject/train_digit_cnn.py 一致：20×28 灰度、
 * (x/255-0.5)/0.5、NCHW [1,1,28,20]。类别 0~7 → 数字 1~8，类别 8 → other。
 * 每个函数须写清作用、输入、输出；调用处也要注明。
 *
 * 数字分类器：透视把两灯条之间的数字区域拉成 20×28，再跑 ONNX CNN。
 * 透视几何跟 clone_detector NumberClassifier::extractNumbers：灯条端点映到
 * 高 28、灯条长 12 的条带，再从中间裁 20×28。
 * 与 MLP 版不同：这里不做 Otsu。模型是按灰度照片训的，二值化会把分布弄偏。
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

    bool loaded() const { return loaded_; }

    /**
     * 作用：把每块板的数字区域透视成 20×28 灰度，写入 ArmorBoard::number_img。
     * 输入：bgr 为当前帧（与灯条坐标同一分辨率）；armors 就地改。
     * 输出：无返回值；失败的板 number_img 为空。
     */
    void extractNumbers(const cv::Mat& bgr, std::vector<ArmorBoard>& armors) const;

    /**
     * 作用：对 number_img 做 CNN 前向；other / 置信度未严格大于门槛 / 大小板数字矛盾的板丢掉。
     * 输入：armors 须已 extractNumbers；threshold 为置信度门槛（须 confidence > threshold）。
     * 输出：就地删除未通过的板；通过的写入 number/confidence/classification_result。
     */
    void classify(std::vector<ArmorBoard>& armors, float threshold);

private:
    /**
     * 作用：灰度图 → 与训练相同的 blob。
     * 输入：gray 任意尺寸单通道。输出：CV_32F blob [1,1,28,20]。
     */
    cv::Mat preprocess(const cv::Mat& gray) const;

    cv::dnn::Net net_;
    bool loaded_ = false;
};

#endif  // NUMBER_CLASSIFIER_H
