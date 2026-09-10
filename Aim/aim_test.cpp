/**
 * @file aim_test.cpp
 * @brief 装甲检测 + 同济整车 EKF + 弹道预瞄的播放调试（不改 armor_test）
 *
 * 注释契约：禁止随意删除本文件注释；改窗口、进度条或快捷键时必须改注释。
 * 画面上的车心/四板/瞄准点必须读 AimTarget 已有量；法线读 ArmorBoard::normal_cam。
 * 检测门槛只从 yaml 读，调灯条/配对请用 armor_test。每个函数须写清作用、输入、输出。
 *
 * 两个窗口：
 *   displayimg   原图 + 可选灯条/装甲/距离/法线/车心/预测板/预瞄；顶栏 FPS；只留 frame 进度条
 *   aim vis      0/1 进度条：light / armor / dist / nml / hub / plates / aim / hud
 *                       nml 默认关；dist 读 ArmorBoard::dist_m
 *
 * 快捷键：空格暂停，a/d 或方向键逐帧，1 开关帧率，q/Esc 退出。Seek 会重置 EKF。
 */
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "ArmorBoard.h"
#include "Light.h"
#include "aimer.h"
#include "armor_detector.h"
#include "light_detector.h"
#include "Detector/detector_config.h"

using namespace cv;
using namespace std;

Mat current_frame;
Mat current_gray;
bool paused = false;

double display_scale = 1.0;

constexpr int kHudH = 36;
constexpr int kFontThick = 1;
constexpr int kMaxWinW = 1920;
constexpr int kMaxWinH = 1080;
constexpr int kWinChromeW = 40;
constexpr int kWinTitleH = 40;
constexpr int kTaskbarH = 48;
constexpr int kTrackbarH = 46;
constexpr int kDisplayTrackbars = 1;

int overlay_light = 1;   // Light::draw_light
int overlay_armor = 1;   // ArmorBoard::draw_armor
int overlay_dist = 1;    // ArmorBoard::dist_m（与 PnP 同一距离）
int overlay_nml = 0;     // ArmorBoard::draw_normal，默认关
int overlay_hub = 1;     // EKF 车心
int overlay_plates = 1;  // EKF 预测的 n 块板
int overlay_aim = 1;     // 弹道预瞄十字（v、ω 飞时超前 + 重力）
int overlay_hud = 1;     // 状态文字

/**
 * 作用：只在播出新帧时刷新 FPS。
 * 输入：fps 要写入；enabled 为 false 则不改。
 * 输出：通过 fps 写回 1/Δt。
 */
void updateFps(double& fps, bool enabled) {
    if (!enabled)
        return;
    static auto last = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    fps = 1.0 / std::chrono::duration<double>(now - last).count();
    last = now;
}

/** 进度条拖动：GTK 不允许在回调里 setTrackbarPos，用 seek 标志在主循环里跳帧。 */
struct FrameScrubber {
    int want = 0;
    bool seek = false;
    bool syncing = false;
};

/**
 * 作用：frame 进度条回调。只置 seek。
 * 输入：pos 为目标帧号；userdata 为 FrameScrubber*。输出：写入 want/seek。
 */
void onFrameTrackbar(int pos, void* userdata) {
    auto* s = static_cast<FrameScrubber*>(userdata);
    if (s->syncing)
        return;
    s->want = pos;
    s->seek = true;
}

bool isLeftKey(int key) {
    const int k = key & 0xFF;
    return key == 81 || key == 2 || key == 2424832 || key == 65361 || k == 'a';
}

bool isRightKey(int key) {
    const int k = key & 0xFF;
    return key == 83 || key == 3 || key == 2555904 || key == 65363 || k == 'd';
}

/**
 * 作用：只缩小不放大，让整窗落在 1920×1080 里。
 * 输入：img；scale_out 可空；n_trackbars 为该窗进度条条数。
 * 输出：缩小后的图，或原图。
 */
Mat scaleTo1080(const Mat& img, double* scale_out, int n_trackbars) {
    const int max_w = std::max(1, kMaxWinW - kWinChromeW);
    const int max_h = std::max(1, kMaxWinH - kWinTitleH - kTaskbarH - n_trackbars * kTrackbarH);
    const double s = std::min({1.0, static_cast<double>(max_w) / std::max(1, img.cols),
                               static_cast<double>(max_h) / std::max(1, img.rows)});
    if (scale_out)
        *scale_out = s;
    if (s >= 1.0)
        return img;
    Mat out;
    resize(img, out, Size(), s, s, INTER_AREA);
    return out;
}

/**
 * 作用：找数字 CNN 的 ONNX。
 * 输入：无。输出：存在的路径；都找不到返回空串。
 */
string findModelPath() {
    const string names[] = {"digit_cnn_fixed.onnx"};
    const string prefixes[] = {
        "../../model/",
        "../model/",
        "model/",
        "/home/zzy/Clion_project/TDT/detector2/model/",
        "/mnt/d/0学业/数学建模/PythonProject/",
    };
    for (const string& pre : prefixes) {
        for (const string& name : names) {
            const string cand = pre + name;
            std::ifstream in(cand);
            if (in.good())
                return cand;
        }
    }
    return {};
}

/**
 * 作用：对干净原图跑灯条+装甲板（门槛已在检测器里，来自 yaml）。
 * 输入：raw 为 BGR。输出：bin/lights/armors。
 */
void processArmors(const Mat& raw, LightDetector& light_detector, ArmorDetector& armor_detector,
                   Mat& bin, std::vector<Light>& lights, std::vector<ArmorBoard>& armors) {
    current_frame = raw.clone();
    cvtColor(current_frame, current_gray, COLOR_BGR2GRAY);
    lights = light_detector.get_lights(current_frame, &bin);  // BGR → 灯条 + 同源二值
    armors = armor_detector.get_armors(lights, current_frame);  // 配对/数字/锁俯仰 PnP
}

/**
 * 作用：在每块已 PnP 的板旁写距离，必须读 ArmorBoard::dist_m。
 * 输入：img 为 BGR；armors 为 get_armors 结果。输出：无。
 */
void drawArmorDistances(Mat& img, const std::vector<ArmorBoard>& armors) {
    for (const auto& armor : armors) {
        const string d = armor.pnp_ok ? cv::format("d:%.2fm", armor.dist_m) : "d:fail";
        Point org(cvRound(armor.center.x + 8), cvRound(armor.center.y - 4));
        org.x = std::clamp(org.x, 2, std::max(2, img.cols - 90));
        org.y = std::clamp(org.y, 12, std::max(12, img.rows - 8));
        putText(img, d, org, FONT_HERSHEY_PLAIN, 0.95, Scalar(0, 255, 255), kFontThick, LINE_AA);
    }
}

/**
 * 作用： vis 开关说明图。输入：无。输出：BGR 说明。
 */
Mat makeVisLegend() {
    Mat panel(222, 520, CV_8UC3, Scalar(28, 28, 28));
    putText(panel, "0=hide  1=show   (displayimg)", Point(8, 22), FONT_HERSHEY_SIMPLEX, 0.42,
            Scalar(0, 200, 100), kFontThick, LINE_AA);
    putText(panel, "light / armor = detector boxes", Point(8, 44), FONT_HERSHEY_SIMPLEX, 0.42,
            Scalar(0, 255, 255), kFontThick, LINE_AA);
    putText(panel, "dist=PnP ||t|| meters  (ArmorBoard::dist_m)", Point(8, 66), FONT_HERSHEY_SIMPLEX,
            0.42, Scalar(0, 255, 255), kFontThick, LINE_AA);
    putText(panel, "nml=magenta  locked-pitch away normal (off)", Point(8, 88), FONT_HERSHEY_SIMPLEX,
            0.42, Scalar(255, 0, 255), kFontThick, LINE_AA);
    putText(panel, "hub=cyan  EKF rotation center", Point(8, 110), FONT_HERSHEY_SIMPLEX, 0.42,
            Scalar(255, 255, 0), kFontThick, LINE_AA);
    putText(panel, "plates=green  predicted armor centers", Point(8, 132), FONT_HERSHEY_SIMPLEX, 0.42,
            Scalar(0, 220, 0), kFontThick, LINE_AA);
    putText(panel, "aim=orange  lead with v,w then gravity", Point(8, 154), FONT_HERSHEY_SIMPLEX,
            0.42, Scalar(0, 140, 255), kFontThick, LINE_AA);
    putText(panel, "hud=id / state / |v| / omega / t", Point(8, 176), FONT_HERSHEY_SIMPLEX, 0.42,
            Scalar(180, 180, 180), kFontThick, LINE_AA);
    putText(panel, "detection params: yaml only (use armor_test)", Point(8, 198),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(180, 180, 180), kFontThick, LINE_AA);
    return panel;
}

/**
 * 作用：检测画面接到 FPS 顶栏下面。
 * 输入：vis、fps。输出：带顶栏的画布。
 */
Mat makeDisplayCanvas(const Mat& vis, double fps) {
    Mat canvas(vis.rows + kHudH, vis.cols, CV_8UC3, Scalar(28, 28, 28));
    vis.copyTo(canvas(Rect(0, kHudH, vis.cols, vis.rows)));
    putText(canvas, cv::format("FPS:%.1f  aim_test  nearest-vehicle EKF", fps), Point(8, 24),
            FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0, 200, 100), kFontThick, LINE_AA);
    return canvas;
}

/**
 * 作用：打开视频，循环检测+EKF 预瞄并显示。
 * 输入：可选 argv[1] 为 YAML；stdin 读视频序号。输出：0 成功。
 */
int main(int argc, char** argv) {
    const string names[] = {
        "2025sj.avi",      "hero_blue.avi",  "Infantry_blue.avi", "Infantry_red1.avi",
        "Infantry_red2.avi", "2026_10.mp4",    "2026_TDT_4.mp4",
    };
    const string prefixes[] = {
        "../../Video/",
        "../Video/",
        "Video/",
        "/home/zzy/Clion_project/TDT/Video/",
    };

    int i = 0;
    std::cout << "请输入数字,按照Video目录顺序选取视频: ";
    cin >> i;
    if (i < 0 || i >= static_cast<int>(sizeof(names) / sizeof(names[0]))) {
        cout << "Invalid index\n";
        return -1;
    }

    string video_path;
    for (const string& pre : prefixes) {
        const string cand = pre + names[i];
        VideoCapture probe(cand);
        if (probe.isOpened()) {
            video_path = cand;
            probe.release();
            break;
        }
    }
    if (video_path.empty()) {
        cout << "Cannot open video\n";
        return -1;
    }

    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cout << "Cannot open video\n";
        return -1;
    }

    int frame_count = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));
    if (frame_count < 2) {
        frame_count = 0;
        Mat probe;
        while (cap.read(probe))
            ++frame_count;
        cap.release();
        cap.open(video_path);
        if (!cap.isOpened()) {
            cout << "Cannot reopen video" << endl;
            return -1;
        }
    }
    if (frame_count < 1)
        frame_count = 1;
    const int last_index = std::max(0, frame_count - 1);
    const int trackbar_max = std::max(1, last_index);
    double video_fps = cap.get(CAP_PROP_FPS);
    if (video_fps < 1.0)
        video_fps = 30.0;
    const double frame_dt = 1.0 / video_fps;  // EKF 用视频时间，不用 waitKey 墙钟

    LightDetector light_detector;
    ArmorDetector armor_detector;
    Aimer aimer;
    const string cfg_preferred = (argc >= 2) ? argv[1] : string();
    const string cfg_path = findDetectorConfigPath(cfg_preferred);
    if (!cfg_path.empty()) {
        const bool light_ok = light_detector.loadConfig(cfg_path);
        const bool armor_ok = armor_detector.loadConfig(cfg_path);
        const bool pnp_ok = ArmorBoard::loadPnpConfig(cfg_path);
        aimer.loadConfig(cfg_path);  // yaml aim → 弹速 + EKF
        if (light_ok && armor_ok && pnp_ok)
            cout << "config: " << cfg_path << "\n";
        else
            cout << "config incomplete: " << cfg_path << "\n";
    } else {
        cout << "config not loaded; using built-in defaults\n";
    }
    const string model_path = findModelPath();
    if (model_path.empty() || !armor_detector.loadClassifier(model_path)) {
        cout << "CNN model not loaded; geometry+PnP still run, numbers skipped\n";
    } else {
        cout << "CNN loaded: " << model_path << "\n";
    }

    Mat raw;
    Mat bin;
    std::vector<Light> lights;
    std::vector<ArmorBoard> armors;
    AimTarget aim_tgt;
    bool is_show_fps = true;
    double fps = 0.0;
    FrameScrubber scrubber;
    int shown_index = 0;

    const string win_display = "displayimg";
    const string win_vis = "aim vis";
    namedWindow(win_display);
    namedWindow(win_vis);
    createTrackbar("frame", win_display, nullptr, trackbar_max, onFrameTrackbar, &scrubber);
    createTrackbar("light", win_vis, &overlay_light, 1);
    createTrackbar("armor", win_vis, &overlay_armor, 1);
    createTrackbar("dist", win_vis, &overlay_dist, 1);  // 0/1：板旁距离，读 dist_m
    createTrackbar("nml", win_vis, &overlay_nml, 1);  // 默认 0
    createTrackbar("hub", win_vis, &overlay_hub, 1);
    createTrackbar("plates", win_vis, &overlay_plates, 1);
    createTrackbar("aim", win_vis, &overlay_aim, 1);
    createTrackbar("hud", win_vis, &overlay_hud, 1);

    auto syncTrackbar = [&]() {
        scrubber.syncing = true;
        setTrackbarPos("frame", win_display, shown_index);
        scrubber.syncing = false;
    };

    auto runAim = [&](double dt) {
        aim_tgt = aimer.update(armors, dt);  // 最近车 EKF + 弹道
    };

    auto grabAt = [&](int idx) {
        idx = std::clamp(idx, 0, last_index);
        cap.set(CAP_PROP_POS_FRAMES, idx);
        cap >> raw;
        if (raw.empty())
            return false;
        shown_index = idx;
        processArmors(raw, light_detector, armor_detector, bin, lights, armors);
        aimer.reset();  // seek 后时间不连续，必须重锁车
        runAim(0.0);
        syncTrackbar();
        return true;
    };

    auto grabNext = [&]() {
        cap >> raw;
        if (raw.empty())
            return false;
        const int pos = static_cast<int>(cap.get(CAP_PROP_POS_FRAMES));
        shown_index = std::clamp(pos - 1, 0, last_index);
        processArmors(raw, light_detector, armor_detector, bin, lights, armors);
        runAim(frame_dt);
        syncTrackbar();
        return true;
    };

    while (true) {
        if (scrubber.seek) {
            scrubber.seek = false;
            paused = true;
            grabAt(scrubber.want);
        } else if (!paused) {
            if (!grabNext())
                paused = true;
            else
                updateFps(fps, is_show_fps);
        }

        if (raw.empty() || current_frame.empty())
            break;

        Mat vis = current_frame.clone();
        if (overlay_light) {
            for (auto& light : lights)
                light.draw_light(vis);
        }
        if (overlay_armor) {
            for (auto& armor : armors)
                armor.draw_armor(vis);
        }
        if (overlay_dist)
            drawArmorDistances(vis, armors);  // 读 ArmorBoard::dist_m
        if (overlay_nml) {
            for (const auto& armor : armors)
                armor.draw_normal(vis);  // 读 normal_cam
        }
        const AimDrawFlags flags{overlay_hub != 0, overlay_plates != 0, overlay_aim != 0,
                                 overlay_hud != 0};
        aimer.draw(vis, aim_tgt, flags);  // 读 AimTarget 像素

        Mat shown = scaleTo1080(makeDisplayCanvas(vis, is_show_fps ? fps : 0.0), &display_scale,
                                kDisplayTrackbars);
        imshow(win_display, shown);
        imshow(win_vis, makeVisLegend());

        const int key = waitKeyEx(15);
        if (key < 0)
            continue;
        const int key8 = key & 0xFF;
        if (key8 == 27 || key8 == 'q')
            break;
        if (key8 == '1')
            is_show_fps = !is_show_fps;
        if (key8 == ' ')
            paused = !paused;
        else if (isLeftKey(key)) {
            paused = true;
            grabAt(shown_index - 1);
        } else if (isRightKey(key)) {
            paused = true;
            grabAt(shown_index + 1);
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
