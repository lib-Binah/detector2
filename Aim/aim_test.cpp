/**
 * @file aim_test.cpp
 * @brief 装甲检测 + lovod 整车 EKF + 弹道预瞄的播放调试（不改 armor_test）
 *
 * 注释契约：禁止随意删除本文件注释；改窗口、进度条或快捷键时必须改注释。
 * 画面上的车心/四板/瞄准点必须读各车 AimTarget 已有量；法线读 ArmorBoard::normal_cam。
 * 同帧多种 ArmorBoard::number 各一套 EKF，各自画 overlay，禁止把异号并进同一刚体。
 * 检测门槛只从 yaml 读，调灯条/配对请用 armor_test。每个函数须写清作用、输入、输出。
 *
 * 两个窗口：
 *   displayimg   原图 + 可选灯条/装甲/距离/法线/车心/速度/预测板/预瞄；顶栏 FPS 与本帧车辆数；只留 frame 进度条
 *   aim vis      0/1 进度条：light / armor / dist / nml / hub / vel / plates / aim / hud / txt
 *                       nml 默认关；dist 读 ArmorBoard::dist_m；vel 读 EKF 车心 v、ω、r1 与 r2
 *                       plates 为预测板色块（与 PnP 同尺寸）；hub 开时青线连车心到正对预测板，
 *                       并在该板心画 ω 切向箭头（读 EKF vyaw）
 *                       txt=0 时不画画面上的字
 *
 * 快捷键：空格暂停，a/d 或方向键逐帧，1 开关帧率，q/Esc 退出。Seek 会重置 EKF。
 * 搜索 ★★★ 可跳到：灯条检测、装甲板检测、整车 EKF（main 里 grabAt/grabNext/runAim）。
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
int overlay_hub = 1;     // EKF 车心青圈 + 连到正对预测板 + 该板心 ω 切向箭头
int overlay_vel = 1;     // 车心线速度 |v|、角速度 ω，读 AimTarget 的 EKF 速度
int overlay_plates = 1;  // EKF 预测板面色块（与 PnP 同尺寸四角）
int overlay_aim = 1;     // 弹道预瞄十字（v、ω 飞时超前 + 重力）
int overlay_hud = 1;     // 状态文字：id / tracking / 飞时
int overlay_txt = 0;     // 总开关：0 不画画面文字（dist/vel/hud），几何 overlay 不受影响

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
    // ★★★【灯条检测】★★★  唯一入口：BGR 原图 → lights；bin 是同源二值，给调试窗用。
    lights = light_detector.get_lights(current_frame, &bin);  // BGR → 灯条 + 同源二值
    // ★★★【装甲板检测】★★★  唯一入口：灯条配对 + 数字 + 锁俯仰 PnP → armors（EKF 只吃这些板）。
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
    Mat panel(288, 540, CV_8UC3, Scalar(28, 28, 28));
    putText(panel, "0=hide  1=show   (displayimg)", Point(8, 22), FONT_HERSHEY_SIMPLEX, 0.42,
            Scalar(0, 200, 100), kFontThick, LINE_AA);
    putText(panel, "light / armor = detector boxes", Point(8, 44), FONT_HERSHEY_SIMPLEX, 0.42,
            Scalar(0, 255, 255), kFontThick, LINE_AA);
    putText(panel, "dist=PnP ||t|| meters  (ArmorBoard::dist_m)", Point(8, 66), FONT_HERSHEY_SIMPLEX,
            0.42, Scalar(0, 255, 255), kFontThick, LINE_AA);
    putText(panel, "nml=magenta  locked-pitch away normal (off)", Point(8, 88), FONT_HERSHEY_SIMPLEX,
            0.42, Scalar(255, 0, 255), kFontThick, LINE_AA);
    putText(panel, "hub=cyan  center, line, omega arrow on facing plate", Point(8, 110), FONT_HERSHEY_SIMPLEX, 0.42,
            Scalar(255, 255, 0), kFontThick, LINE_AA);
    putText(panel, "vel=hub v/w and r1, r2  (camera twist, current/other radius)", Point(8, 132),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(255, 255, 0), kFontThick, LINE_AA);
    putText(panel, "plates=fill  EKF armor quads (PnP size, 4 colors)", Point(8, 154), FONT_HERSHEY_SIMPLEX, 0.42,
            Scalar(0, 220, 0), kFontThick, LINE_AA);
    putText(panel, "aim=orange  lead with kv*v, kw*w then gravity", Point(8, 176), FONT_HERSHEY_SIMPLEX,
            0.42, Scalar(0, 140, 255), kFontThick, LINE_AA);
    putText(panel, "hud=id / state / t", Point(8, 198), FONT_HERSHEY_SIMPLEX, 0.42,
            Scalar(180, 180, 180), kFontThick, LINE_AA);
    putText(panel, "txt=overlay text (dist/vel/hud); boxes stay", Point(8, 220), FONT_HERSHEY_SIMPLEX,
            0.42, Scalar(180, 180, 180), kFontThick, LINE_AA);
    putText(panel, "detection params: yaml only (use armor_test)", Point(8, 242),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(180, 180, 180), kFontThick, LINE_AA);
    putText(panel, "multi-id: each ArmorBoard::number has its own EKF", Point(8, 264),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(180, 180, 180), kFontThick, LINE_AA);
    return panel;
}

/**
 * 作用：检测画面接到 FPS 顶栏下面。
 * 输入：vis、fps、n_cars（本帧 AimTarget 条数）。输出：带顶栏的画布。
 */
Mat makeDisplayCanvas(const Mat& vis, double fps, int n_cars) {
    Mat canvas(vis.rows + kHudH, vis.cols, CV_8UC3, Scalar(28, 28, 28));
    vis.copyTo(canvas(Rect(0, kHudH, vis.cols, vis.rows)));
    // n_cars 读本帧 AimTarget 条数，与 Aimer 按 number 开的 EKF 套数同一来源。
    putText(canvas, cv::format("FPS:%.1f  aim_test  cars:%d  multi-EKF", fps, n_cars), Point(8, 24),
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
    std::vector<AimTarget> aim_tgts;
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
    createTrackbar("vel", win_vis, &overlay_vel, 1);  // 0/1：车心 v 与 ω，读 EKF
    createTrackbar("plates", win_vis, &overlay_plates, 1);
    createTrackbar("aim", win_vis, &overlay_aim, 1);
    createTrackbar("hud", win_vis, &overlay_hud, 1);
    createTrackbar("txt", win_vis, &overlay_txt, 1);  // 0/1：画面文字总开关，几何仍画

    // 下面四个小函数共用这些量：
    //   shown_index  画面上正在看的帧号（0 起）
    //   last_index   最后一帧号 = 总帧数-1
    //   frame_dt     相邻两帧间隔秒 = 1/视频帧率，只给连续播放的 EKF 用
    //   raw          刚从视频读出的 BGR 原图
    //   lights/armors 本帧检测结果；aim_tgts 本帧各车瞄准结果
    //   scrubber     进度条：want=想跳到哪一帧，seek=用户拖过了，syncing=程序自己改滑块时挡住回调
    //   paused       true=停住不读下一帧，仍会重画 overlay
    /**
     * 作用：把进度条滑块拨到当前正在看的那一帧，避免滑块和画面差一帧。
     * 输入：无（读 shown_index）。输出：无。
     * 原理：setTrackbarPos 会触发 onFrameTrackbar；先把 syncing 置真，回调里直接 return，
     *       否则程序会以为用户又拖了一次进度条，陷入反复跳帧。
     */
    auto syncTrackbar = [&]() {
        scrubber.syncing = true;  // 告诉回调：这次是程序自己改滑块，不是人手拖
        setTrackbarPos("frame", win_display, shown_index);  // 滑块数字 = 当前帧号
        scrubber.syncing = false;
    };

    /**
     * 作用：用本帧检测到的装甲跑一遍瞄准：每号车各一套 EKF，再算弹道。
     * 输入：dt 为这一帧相对上一帧过了多少秒。连续播放传 frame_dt；跳帧/seek 传 0（不让滤波当时间在走）。
     * 输出：写入 aim_tgts。一条 AimTarget = 一辆未 Lost 的车（车心、四板、预瞄点）。
     */
    auto runAim = [&](double dt) {
        // ★★★【整车 EKF】★★★  main 里唯一调用：本帧 armors + dt → 每号各一套滤波 + 弹道。
        //     真正 predict/update 在 Aimer::update → VehicleTracker::step（Aim/vehicle_ekf.cpp）。
        aim_tgts = aimer.update(armors, dt);  // 每号各一套 EKF + 弹道
    };

    /**
     * 作用：跳到指定帧（拖进度条、左右键逐帧）。时间不连续，所以 EKF 必须重锁。
     * 输入：idx 为目标帧号（0 起）。输出：读到图返回 true；空帧返回 false。
     */
    auto grabAt = [&](int idx) {
        idx = std::clamp(idx, 0, last_index);  // 夹到合法范围，避免越界
        cap.set(CAP_PROP_POS_FRAMES, idx);     // 告诉解码器：下一张读第 idx 帧
        cap >> raw;                            // 读出 BGR 原图
        if (raw.empty())
            return false;
        shown_index = idx;  // 画面上现在就是这一帧
        // ★★★【灯条检测 + 装甲板检测】★★★  跳帧路径：原图 → lights / armors。
        processArmors(raw, light_detector, armor_detector, bin, lights, armors);  // 原图 → 灯条 + 装甲
        aimer.reset();  // seek 后时间不连续，必须重锁车
        // ★★★【整车 EKF】★★★  跳帧路径：dt=0，不按视频间隔预测，只在本帧初始化/显示。
        runAim(0.0);    // dt=0：只在这一帧上初始化/显示，不按视频间隔去预测
        syncTrackbar(); // 进度条跟上 shown_index
        return true;
    };

    /**
     * 作用：连续播放时读下一帧，用真实帧间隔去推 EKF。
     * 输入：无。输出：读到图返回 true；视频结束空帧返回 false。
     * 说明：OpenCV 读完一帧后，POS_FRAMES 已经指向「下一帧」的下标，所以当前帧号是 pos-1。
     */
    auto grabNext = [&]() {
        cap >> raw;  // 顺序读下一张 BGR
        if (raw.empty())
            return false;
        const int pos = static_cast<int>(cap.get(CAP_PROP_POS_FRAMES));  // 解码器眼里的「下一帧」下标
        shown_index = std::clamp(pos - 1, 0, last_index);               // 刚读出来的才是当前帧
        // ★★★【灯条检测 + 装甲板检测】★★★  连续播放路径：原图 → lights / armors。
        processArmors(raw, light_detector, armor_detector, bin, lights, armors);  // 原图 → 灯条 + 装甲
        // ★★★【整车 EKF】★★★  连续播放路径：dt=1/视频帧率，按视频时间预测。
        runAim(frame_dt);  // dt=1/视频帧率，EKF 按视频时间预测，不按墙钟
        syncTrackbar();
        return true;
    };

    // 主循环：先决定「要不要读新帧」，再把当前帧画出来，最后等键盘。
    while (true) {
        if (scrubber.seek) {
            // 用户拖了 frame 进度条：回调只记下 want，真正跳帧放在这里（GTK 不允许在回调里 setTrackbarPos）。
            scrubber.seek = false;
            paused = true;           // 跳完停住，方便看这一帧
            grabAt(scrubber.want);  // want = 进度条上的目标帧号
        } else if (!paused) {
            // 未暂停：按视频顺序播下一帧；播完了就自动暂停。
            if (!grabNext())
                paused = true;
            else
                updateFps(fps, is_show_fps);  // 只在真正播出新帧时刷 FPS
        }
        // paused==true 且没 seek：不读新帧，下面仍会把上一帧再画一遍（改 overlay 开关立刻能看见）。

        if (raw.empty() || current_frame.empty())
            break;  // 第一帧就读失败，或画面还没建立

        Mat vis = current_frame.clone();  // 在副本上画 overlay，不改 current_frame
        if (overlay_light) {
            for (auto& light : lights)
                light.draw_light(vis);  // 检测灯条框，和筛灯条用同一套几何
        }
        if (overlay_armor) {
            for (auto& armor : armors)
                armor.draw_armor(vis);  // 检测装甲四边形
        }
        if (overlay_dist && overlay_txt)
            drawArmorDistances(vis, armors);  // 读 ArmorBoard::dist_m；受 txt 总开关
        if (overlay_nml) {
            for (const auto& armor : armors)
                armor.draw_normal(vis);  // 读 normal_cam
        }
        // txt=0 时 vel/hud 的 putText 关掉，hub（连线+ω箭头）/plates/aim 几何仍按各自开关画
        // flags 五个开关：车心几何、预测板色块、预瞄十字、状态字、速度字
        const AimDrawFlags flags{overlay_hub != 0, overlay_plates != 0, overlay_aim != 0,
                                 overlay_hud != 0 && overlay_txt != 0,
                                 overlay_vel != 0 && overlay_txt != 0};
        aimer.draw(vis, aim_tgts, flags);  // 每辆车读自己的 AimTarget 像素

        // 顶栏写 FPS 和本帧车辆数（= aim_tgts 条数），再缩到 1080p 窗口里
        Mat shown = scaleTo1080(makeDisplayCanvas(vis, is_show_fps ? fps : 0.0,
                                                  static_cast<int>(aim_tgts.size())),
                                &display_scale,
                                kDisplayTrackbars);
        imshow(win_display, shown);       // 大窗：视频 + overlay
        imshow(win_vis, makeVisLegend()); // 小窗：各进度条含义说明

        const int key = waitKeyEx(15);  // 最多等 15ms；没按键返回负数，画面照常刷新
        if (key < 0)
            continue;  // 没按键：进入下一圈（暂停时等于原地重画）
        const int key8 = key & 0xFF;  // 取低 8 位，兼容 Esc/字母；方向键要用原始 key 判断
        if (key8 == 27 || key8 == 'q')
            break;  // Esc 或 q：退出播放
        if (key8 == '1')
            is_show_fps = !is_show_fps;  // 顶栏 FPS 显示开/关
        if (key8 == ' ')
            paused = !paused;  // 空格：播 / 停
        else if (isLeftKey(key)) {
            paused = true;
            grabAt(shown_index - 1);  // a 或左方向：上一帧（内部会 reset EKF）
        } else if (isRightKey(key)) {
            paused = true;
            grabAt(shown_index + 1);  // d 或右方向：下一帧（内部会 reset EKF）
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
