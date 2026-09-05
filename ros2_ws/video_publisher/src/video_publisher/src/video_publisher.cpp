/**
 * @file video_publisher.cpp
 * @brief 按视频文件自身帧率把 BGR 帧发到 /image_raw。
 *
 * 节奏在发布端：wall timer 周期 = 1/CAP_PROP_FPS，不是订阅端 waitKey。
 * QoS KeepLast(1)：检测慢于出图时只留最新一帧，避免队列积压导致画面滞后。
 * 每个函数须写清作用/输入/输出；调用处注明调谁、送什么、得到什么。
 */

#include <chrono>
#include <string>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

namespace {

// 与 armor_test 同一套文件名，video_index 选片时走这里。
const char* kVideoNames[] = {
    "2025sj.avi",
    "hero_blue.avi",
    "Infantry_blue.avi",
    "Infantry_red1.avi",
    "Infantry_red2.avi",
    "2026_10.mp4",
    "2026_TDT_4.mp4",
};
constexpr int kVideoNameCount = 7;

const char* kVideoPrefixes[] = {
    "/home/zzy/Clion_project/TDT/Video/",
};

/**
 * 作用：按 armor_test 的目录习惯解析视频路径。
 * 输入：index 为 names 下标；index<0 时用 video_path。
 * 输出：能打开的绝对/相对路径；都失败则空串。
 */
std::string resolveVideoPath(int index, const std::string& video_path) {
    if (index < 0) {
        cv::VideoCapture probe(video_path);  // 用户给的路径 → 能否打开
        if (probe.isOpened())
            return video_path;
        return {};
    }
    if (index >= kVideoNameCount)
        return {};
    for (const char* pre : kVideoPrefixes) {
        const std::string cand = std::string(pre) + kVideoNames[index];
        cv::VideoCapture probe(cand);  // 前缀+文件名 → 能否打开
        if (probe.isOpened())
            return cand;
    }
    return {};
}

}  // namespace

class VideoPublisher : public rclcpp::Node {
public:
    /**
     * 作用：读参数、打开视频、按文件 fps 起定时器发图。
     * 输入：无（参数从 ROS 声明：video_path / video_index / topic / loop / frame_id）。
     * 输出：构造失败时抛异常，让 main 以非 0 退出。
     */
    VideoPublisher() : Node("video_publisher") {
        const auto video_path = declare_parameter<std::string>(
            "video_path",
            "/home/zzy/Clion_project/TDT/Video/Infantry_blue.avi");
        const int video_index = declare_parameter<int>("video_index", -1);
        const auto topic = declare_parameter<std::string>("topic", "/image_raw");
        loop_ = declare_parameter<bool>("loop", true);
        frame_id_ = declare_parameter<std::string>("frame_id", "camera");

        const std::string path = resolveVideoPath(video_index, video_path);
        if (path.empty()) {
            throw std::runtime_error(
                "打不开视频。video_index=" + std::to_string(video_index) +
                " video_path=" + video_path);
        }

        cap_.open(path);  // 已验证可打开的路径 → VideoCapture
        if (!cap_.isOpened()) {
            throw std::runtime_error("reopen failed: " + path);
        }

        double file_fps = cap_.get(cv::CAP_PROP_FPS);  // 容器元数据帧率；有的 avi 会给 0
        if (file_fps < 1.0)
            file_fps = 30.0;
        file_fps_ = file_fps;

        // 深度 1：订阅端处理不过来时中间帧直接丢，只留最新
        pub_ = create_publisher<sensor_msgs::msg::Image>(
            topic, rclcpp::QoS(rclcpp::KeepLast(1)).reliable());

        const auto period = std::chrono::duration<double>(1.0 / file_fps_);
        timer_ = create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(period),
            std::bind(&VideoPublisher::onTick, this));

        RCLCPP_INFO(get_logger(), "发布 %s @ %.1f fps → %s",
                    path.c_str(), file_fps_, topic.c_str());
    }

private:
    /**
     * 作用：定时器触发时读一帧并发布，周期对齐视频 fps。
     * 输入：无（读 cap_ 当前游标）。
     * 输出：成功则往 topic 发一帧 bgr8 Image；播完且 loop=false 则关上下文。
     */
    void onTick() {
        cv::Mat frame;
        cap_ >> frame;  // 下一帧 BGR；空 = 文件结束
        if (frame.empty()) {
            if (!loop_) {
                RCLCPP_INFO(get_logger(), "视频结束，loop=false，退出");
                rclcpp::shutdown();
                return;
            }
            cap_.set(cv::CAP_PROP_POS_FRAMES, 0);  // 回到第 0 帧，循环播
            cap_ >> frame;
            if (frame.empty()) {
                RCLCPP_ERROR(get_logger(), "循环后仍读不到帧，退出");
                rclcpp::shutdown();
                return;
            }
        }

        std_msgs::msg::Header header;
        header.stamp = now();  // 发出瞬间，给后面 PnP/预测用
        header.frame_id = frame_id_;
        // BGR Mat + 头 → sensor_msgs/Image（bgr8，与 OpenCV 默认通道序一致）
        const auto msg = cv_bridge::CvImage(header, "bgr8", frame).toImageMsg();
        pub_->publish(*msg);
    }

    cv::VideoCapture cap_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    bool loop_ = true;
    std::string frame_id_;
    double file_fps_ = 30.0;
};

/**
 * 作用：起 ROS 上下文、spin 发布节点。
 * 输入：argc/argv，可含 --ros-args -p video_path:=... 或 -p video_index:=2。
 * 输出：0 正常结束；打开视频失败为 1。
 */
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    try {
        rclcpp::spin(std::make_shared<VideoPublisher>());
    } catch (const std::exception& e) {
        fprintf(stderr, "video_publisher: %s\n", e.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
