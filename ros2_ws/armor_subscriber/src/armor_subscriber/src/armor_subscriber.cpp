/**
 * @file armor_subscriber.cpp
 * @brief 订阅 /image_raw，每收到一帧就跑灯条+装甲板（按接收帧率处理）。
 *
 * 不读 VideoCapture：出图节奏在 video_publisher；本节点只在回调里检测。
 * QoS KeepLast(1)：检测慢于出图时丢掉积压的旧帧，只处理最新一帧。
 * 检测与 overlay 读 Light / ArmorBoard 同一字段；waitKey(1) 只刷窗，不当节拍器。
 * 每个函数须写清作用/输入/输出；调用处注明调谁、送什么、得到什么。
 */

#include <chrono>
#include <fstream>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "ArmorBoard.h"
#include "Light.h"
#include "armor_detector.h"
#include "light_detector.h"

namespace {

constexpr int kFontThick = 1;

/**
 * 作用：在若干候选路径里找 ONNX；colcon 安装目录相对路径对不上时走绝对路径。
 * 输入：无。输出：存在的路径；都找不到返回空串。
 */
std::string findModelPath() {
    const std::string names[] = {"digit_cnn_fixed.onnx"};
    const std::string prefixes[] = {
        "/home/zzy/Clion_project/TDT/detector2/model/",
        "../../../../model/",
        "../../../model/",
        "../../model/",
        "../model/",
        "model/",
    };
    for (const std::string& pre : prefixes) {
        for (const std::string& name : names) {
            const std::string cand = pre + name;
            std::ifstream in(cand);
            if (in.good())
                return cand;
        }
    }
    return {};
}

}  // namespace

class ArmorSubscriber : public rclcpp::Node {
public:
    /**
     * 作用：订 /image_raw、加载 CNN、可选打开调试窗。
     * 输入：ROS 参数 topic / show_gui；gray/color/conf 与 armor_test 同源。
     * 输出：订阅建立后由 spin 驱动 onImage。
     */
    ArmorSubscriber() : Node("armor_subscriber") {
        const auto topic = declare_parameter<std::string>("topic", "/image_raw");
        show_gui_ = declare_parameter<bool>("show_gui", true);
        gray_thres_ = static_cast<int>(declare_parameter<int>("gray_thres", LightParams::kGrayThresDefault));
        detect_color_ = static_cast<int>(declare_parameter<int>("detect_color", 0));
        conf_thres_pct_ = static_cast<int>(declare_parameter<int>("conf_thres_pct", 80));

        const std::string model_path = findModelPath();
        if (model_path.empty() || !armor_detector_.loadClassifier(model_path)) {
            RCLCPP_WARN(get_logger(), "CNN 未加载；几何+PnP 仍跑，数字跳过");
            if (!model_path.empty())
                RCLCPP_WARN(get_logger(), "tried: %s", model_path.c_str());
        } else {
            RCLCPP_INFO(get_logger(), "CNN loaded: %s", model_path.c_str());
        }

        // 深度 1：处理不过来时中间帧直接丢，画面仍是最新
        sub_ = create_subscription<sensor_msgs::msg::Image>(
            topic, rclcpp::QoS(rclcpp::KeepLast(1)).reliable(),
            std::bind(&ArmorSubscriber::onImage, this, std::placeholders::_1));

        if (show_gui_) {
            cv::namedWindow(win_, cv::WINDOW_NORMAL);
            cv::createTrackbar("gray", win_, &gray_thres_, 255);   // 灯条 get_bin 灰度门槛
            cv::createTrackbar("color", win_, &detect_color_, 2);  // 0 both / 1 red / 2 blue
            cv::createTrackbar("conf", win_, &conf_thres_pct_, 100);  // CNN 须严格大于该百分比
        }

        RCLCPP_INFO(get_logger(), "订阅 %s，KeepLast(1)，按接收帧率检测", topic.c_str());
    }

    ~ArmorSubscriber() override {
        if (show_gui_)
            cv::destroyAllWindows();
    }

private:
    /**
     * 作用：对一帧 BGR 跑灯条+装甲板，门槛读进度条/参数（与 armor_test::processArmors 同一套调用）。
     * 输入：bgr 为当前帧；写出 bin/lights/armors。
     * 输出：无返回值；失败时 lights/armors 可能为空。
     */
    void processArmors(const cv::Mat& bgr, cv::Mat& bin, std::vector<Light>& lights,
                       std::vector<ArmorBoard>& armors) {
        light_detector_.setGrayThres(gray_thres_);  // 进度条 int → 灯条检测器 [0,255]
        lights = light_detector_.get_lights(bgr, &bin);  // BGR → 灯条 + 同源二值

        ArmorParams p = armor_detector_.params();
        p.detect_color = detect_color_;
        p.classifier_threshold = static_cast<float>(conf_thres_pct_) / 100.f;
        armor_detector_.setParams(p);
        // get_armors：灯条列表 + BGR 帧 → 配对/数字/PnP 后的装甲板
        armors = armor_detector_.get_armors(lights, bgr);
    }

    /**
     * 作用：在已画框的图上写 ArmorBoard 的 number/conf/dist（读同一字段，不另算几何）。
     * 输入：img 为 BGR；armors 为 get_armors 的结果。输出：无。
     */
    void drawArmorText(cv::Mat& img, const std::vector<ArmorBoard>& armors) const {
        for (const auto& armor : armors) {
            const cv::Point org(cvRound(armor.center.x + 8), cvRound(armor.center.y));
            const std::string num = armor.classification_result.empty()
                                        ? (armor.number > 0 ? std::to_string(armor.number) : "?")
                                        : armor.classification_result;
            const std::string dist = armor.pnp_ok ? cv::format(" %.2fm", armor.dist_m) : " fail";
            cv::putText(img, num + cv::format(" p:%.2f", armor.confidence) + dist, org,
                        cv::FONT_HERSHEY_PLAIN, 0.95, cv::Scalar(0, 255, 255), kFontThick, cv::LINE_AA);
        }
    }

    /**
     * 作用：消息一到就检测；频率 = 订阅回调频率（接收帧率，慢则丢旧帧）。
     * 输入：msg 为 bgr8 Image。输出：日志里的 Hz；show_gui 时刷新窗口。
     */
    void onImage(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        cv::Mat bgr;
        try {
            // ROS Image → 自带拷贝的 BGR Mat（回调结束后 msg 可能被放掉，不能 toCvShare 长期持有）
            bgr = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge: %s", e.what());
            return;
        }
        if (bgr.empty())
            return;

        cv::Mat bin;
        std::vector<Light> lights;
        std::vector<ArmorBoard> armors;
        processArmors(bgr, bin, lights, armors);  // 本帧 BGR → 灯条/装甲板

        const auto now = std::chrono::steady_clock::now();
        const double hz = 1.0 / std::chrono::duration<double>(now - last_).count();
        last_ = now;

        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                             "recv+process ≈ %.1f Hz  lights=%zu armors=%zu  %dx%d",
                             hz, lights.size(), armors.size(), bgr.cols, bgr.rows);

        if (!show_gui_)
            return;

        cv::Mat vis = bgr.clone();
        for (auto& light : lights)
            light.draw_light(vis);  // Light 几何框，与筛选同一字段
        for (auto& armor : armors)
            armor.draw_armor(vis);  // ArmorBoard 四顶点，与配对同一对象
        drawArmorText(vis, armors);

        cv::putText(vis, cv::format("Hz:%.1f L:%zu A:%zu", hz, lights.size(), armors.size()),
                    cv::Point(8, 24), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 200, 100),
                    kFontThick, cv::LINE_AA);
        cv::imshow(win_, vis);
        const int key = cv::waitKey(1);  // 只刷 GUI，1ms，不决定检测节拍
        if (key == 27 || (key & 0xFF) == 'q')
            rclcpp::shutdown();
    }

    LightDetector light_detector_;
    ArmorDetector armor_detector_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    std::chrono::steady_clock::time_point last_{std::chrono::steady_clock::now()};
    bool show_gui_ = true;
    int gray_thres_ = LightParams::kGrayThresDefault;
    int detect_color_ = 0;
    int conf_thres_pct_ = 80;
    const std::string win_{"armor recv"};
};

/**
 * 作用：起 ROS 上下文并 spin，帧来了才进 onImage。
 * 输入：argc/argv，可含 --ros-args -p topic:=/image_raw -p show_gui:=false。
 * 输出：0 正常结束。
 */
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ArmorSubscriber>());
    rclcpp::shutdown();
    return 0;
}
