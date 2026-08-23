/**
 * @file armor_test.cpp
 * @brief 灯条 + 装甲板配对 / 数字 CNN / PnP 的播放调试（不改灯条检测源码）
 *
 * 注释契约：禁止随意删除本文件注释；改窗口布局、HUD、进度条或快捷键时必须改注释。
 * overlay 画的 type/lr/cd/ang/num/conf/dist 必须读 ArmorBoard 的同一属性，与配对/分类/PnP
 * 计算方式相同；禁止显示时另算中心距或长度比。灯条框仍读 Light 字段。
 * 每个函数须写清作用、输入、输出；调用处也要注明。
 *
 * 三个窗口：
 *   displayimg          原图+灯条框+装甲板框及可选标注；顶栏 FPS/meanGray；frame/gray 进度条
 *   Frame with Armors   上半原图+框（无数字），下半灰度二值
 *   armor overlay       0/1 进度条：type / 灯条长度比 / 归一化中心距 / 倾角 / 数字 / 置信度 / 距离
 *
 * 快捷键：空格暂停，a/d 或方向键逐帧，1 开关帧率，q/Esc 退出。
 */
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "ArmorBoard.h"
#include "Light.h"
#include "armor_detector.h"
#include "light_detector.h"

using namespace cv;
using namespace std;

Mat current_frame;
Mat current_gray;
bool paused = false;

int display_hud_h = 0;
double display_scale = 1.0;
int combined_y_offset = 0;
double combined_scale = 1.0;

constexpr int kHudH = 36;
constexpr int kFontThick = 1;
constexpr int kMaxWinW = 1920;
constexpr int kMaxWinH = 1080;

int gray_thres = LightParams::kGrayThresDefault;
int detect_color = 0;     // 0=红蓝都配，1=只红，2=只蓝；与 ArmorParams::detect_color 同源
int conf_thres_pct = 80;  // CNN 置信度门槛 ×100（须大于该值）；写入 classifier_threshold，默认 80%

// overlay 0/1：只改 displayimg 文字，不改检测。GTK 进度条下限是 0。
int overlay_type = 1;   // ArmorBoard::type
int overlay_lr = 1;     // ArmorBoard::light_ratio，与 isArmor 同一字段
int overlay_cd = 1;     // ArmorBoard::center_distance
int overlay_ang = 1;    // ArmorBoard::angle
int overlay_num = 1;    // number + classification_result
int overlay_conf = 1;   // ArmorBoard::confidence
int overlay_dist = 1;   // ArmorBoard::dist_m（PnP）

/**
 * 作用：只在真正播出新帧时刷新 FPS，暂停时沿用上次。
 * 输入：fps 为要写入的变量；enabled 为 false 则不改 fps。
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
 * 作用：frame 进度条回调。只置 seek，主循环再跳帧。
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
 * 作用：只缩小不放大，保证窗口内容不超过 1920×1080。
 * 输入：img 任意尺寸；scale_out 非空时写出缩放比。
 * 输出：缩小后的图，或原图（已够小）。
 */
Mat scaleTo1080(const Mat& img, double* scale_out) {
    const double s = std::min({1.0,
                               static_cast<double>(kMaxWinW) / std::max(1, img.cols),
                               static_cast<double>(kMaxWinH) / std::max(1, img.rows)});
    if (scale_out)
        *scale_out = s;
    if (s >= 1.0)
        return img;
    Mat out;
    resize(img, out, Size(), s, s, INTER_AREA);
    return out;
}

/**
 * 作用：在若干候选路径里找 ONNX；相对 cmake-build-debug 时 ../../model 指向工程 model/。
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
 * 作用：对干净原图跑灯条+装甲板，并刷新 current_frame / current_gray。
 * 输入：raw 为 BGR；light_detector 用 gray_thres；armor_detector 用 detect_color/conf。
 * 输出：bin=灯条二值；lights/armors 为检测结果。
 */
void processArmors(const Mat& raw, LightDetector& light_detector, ArmorDetector& armor_detector,
                   Mat& bin, std::vector<Light>& lights, std::vector<ArmorBoard>& armors) {
    current_frame = raw.clone();
    cvtColor(current_frame, current_gray, COLOR_BGR2GRAY);
    light_detector.setGrayThres(gray_thres);  // 进度条 int → 灯条检测器 [0,255]
    lights = light_detector.get_lights(current_frame, &bin);  // BGR → 灯条 + 同源二值

    ArmorParams p = armor_detector.params();
    p.detect_color = detect_color;
    p.classifier_threshold = static_cast<float>(conf_thres_pct) / 100.f;
    armor_detector.setParams(p);
    // get_armors：灯条列表 + BGR 帧 → 配对/数字/PnP 后的装甲板
    armors = armor_detector.get_armors(lights, current_frame);
}

/**
 * 作用：按 overlay 开关在装甲板旁画 type/lr/cd/ang/num/conf/dist（关一行就上移）。
 * 输入：img 为已画框的 BGR；armors 为 get_armors 的结果。
 * 输出：无返回值。数字必须读 ArmorBoard 字段，禁止另算中心距或长度比。
 */
void drawArmorOverlay(Mat& img, const std::vector<ArmorBoard>& armors) {
    const int line_h = 14;
    for (const auto& armor : armors) {
        const int nlines = (overlay_type ? 1 : 0) + (overlay_lr ? 1 : 0) +
                           (overlay_cd ? 1 : 0) + (overlay_ang ? 1 : 0) +
                           (overlay_num ? 1 : 0) + (overlay_conf ? 1 : 0) +
                           (overlay_dist ? 1 : 0);
        if (nlines == 0)
            continue;

        Point org(cvRound(armor.center.x + 8), cvRound(armor.center.y));
        org.x = std::clamp(org.x, 2, std::max(2, img.cols - 140));
        org.y = std::clamp(org.y, 12, std::max(12, img.rows - line_h * nlines - 4));

        const Scalar text(0, 255, 255);
        int y = org.y;
        if (overlay_type) {
            const char* t = (armor.type == ArmorType::LARGE) ? "LARGE" : "SMALL";
            putText(img, t, Point(org.x, y), FONT_HERSHEY_PLAIN, 0.95, text, kFontThick, LINE_AA);
            y += line_h;
        }
        if (overlay_lr) {
            putText(img, cv::format("lr:%.2f", armor.light_ratio), Point(org.x, y),
                    FONT_HERSHEY_PLAIN, 0.95, text, kFontThick, LINE_AA);
            y += line_h;
        }
        if (overlay_cd) {
            putText(img, cv::format("cd:%.2f", armor.center_distance), Point(org.x, y),
                    FONT_HERSHEY_PLAIN, 0.95, text, kFontThick, LINE_AA);
            y += line_h;
        }
        if (overlay_ang) {
            putText(img, cv::format("ang:%.1f", armor.angle), Point(org.x, y),
                    FONT_HERSHEY_PLAIN, 0.95, text, kFontThick, LINE_AA);
            y += line_h;
        }
        if (overlay_num) {
            const string num = armor.classification_result.empty()
                                   ? (armor.number > 0 ? std::to_string(armor.number) : "?")
                                   : armor.classification_result;
            putText(img, num, Point(org.x, y), FONT_HERSHEY_PLAIN, 0.95, text, kFontThick, LINE_AA);
            y += line_h;
        }
        if (overlay_conf) {
            putText(img, cv::format("p:%.2f", armor.confidence), Point(org.x, y),
                    FONT_HERSHEY_PLAIN, 0.95, text, kFontThick, LINE_AA);
            y += line_h;
        }
        if (overlay_dist) {
            const string d = armor.pnp_ok ? cv::format("d:%.2fm", armor.dist_m) : "d:fail";
            putText(img, d, Point(org.x, y), FONT_HERSHEY_PLAIN, 0.95, text, kFontThick, LINE_AA);
        }
    }
}

/**
 * 作用：把 20×28 数字 ROI 贴到画面左上，方便核对 CNN 输入是否同源。
 * 输入：img 为 BGR 画布；armors 带 number_img。输出：无。
 */
void drawNumberThumbnails(Mat& img, const std::vector<ArmorBoard>& armors) {
    int x = 8;
    const int y = 8;
    for (const auto& armor : armors) {
        if (armor.number_img.empty())
            continue;
        Mat vis;
        cvtColor(armor.number_img, vis, COLOR_GRAY2BGR);
        resize(vis, vis, Size(40, 56), 0, 0, INTER_NEAREST);
        if (x + vis.cols >= img.cols || y + vis.rows >= img.rows)
            break;
        vis.copyTo(img(Rect(x, y, vis.cols, vis.rows)));
        x += vis.cols + 4;
    }
}

/**
 * 作用：给 overlay 窗生成说明图（进度条名字太短，用画面写 0/1 和几何定义）。
 * 输入：无。输出：BGR 说明图，每帧 imshow，避免空窗被后端丢掉。
 */
Mat makeOverlayLegend() {
    Mat panel(160, 520, CV_8UC3, Scalar(28, 28, 28));
    putText(panel, "0=hide  1=show   (displayimg only)", Point(8, 22),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(0, 200, 100), kFontThick, LINE_AA);
    putText(panel, "type=SMALL/LARGE from center_distance", Point(8, 44),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(0, 255, 255), kFontThick, LINE_AA);
    putText(panel, "lr=min(H)/max(H)  cd=centerDist/avg H", Point(8, 66),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(0, 255, 255), kFontThick, LINE_AA);
    putText(panel, "ang=center-line vs horizontal (deg)", Point(8, 88),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(0, 255, 255), kFontThick, LINE_AA);
    putText(panel, "num/conf=CNN 1-8  dist=PnP ||t|| meters", Point(8, 110),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(0, 255, 255), kFontThick, LINE_AA);
    putText(panel, "color 0=both 1=red 2=blue   conf=CNN must >N%", Point(8, 132),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(0, 200, 100), kFontThick, LINE_AA);
    putText(panel, "H=Light.height  same field as matching", Point(8, 154),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(180, 180, 180), kFontThick, LINE_AA);
    return panel;
}

/**
 * 作用：把检测画面接到带标签的顶栏下面（FPS/meanGray）。
 * 输入：vis 为已画框的画面；fps、mean_gray 为顶栏左侧数字。
 * 输出：高 = vis.rows+kHudH 的画布。
 */
Mat makeDisplayCanvas(const Mat& vis, double fps, double mean_gray) {
    Mat canvas(vis.rows + kHudH, vis.cols, CV_8UC3, Scalar(28, 28, 28));
    vis.copyTo(canvas(Rect(0, kHudH, vis.cols, vis.rows)));
    display_hud_h = kHudH;
    const string left = cv::format("FPS:%.1f  meanGray:%.1f", fps, mean_gray);
    putText(canvas, left, Point(8, 24), FONT_HERSHEY_SIMPLEX, 0.55,
            Scalar(0, 200, 100), kFontThick, LINE_AA);
    return canvas;
}

/**
 * 作用：打开视频，循环灯条+装甲板检测并显示。
 * 输入：stdin 读 0–4 选视频。输出：0 成功，非 0 打不开视频或找不到模型时仍可跑几何。
 */
int main() {
    const string names[] = {
        "2025sj.avi",
        "hero_blue.avi",
        "Infantry_blue.avi",
        "Infantry_red1.avi",
        "Infantry_red2.avi",
    };
    const string prefixes[] = {
        "../../Video/",
        "../Video/",
        "Video/",
        "/home/zzy/Clion_project/TDT/Video/",
    };

    int i = 0;
    std::cout << "请输入0-4,按照Video目录顺序选取视频: ";
    cin >> i;
    if (i < 0 || i > 4) {
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

    LightDetector light_detector;
    ArmorDetector armor_detector;
    const string model_path = findModelPath();
    if (model_path.empty() || !armor_detector.loadClassifier(model_path)) {
        cout << "CNN model not loaded; geometry+PnP still run, numbers skipped\n";
        if (!model_path.empty())
            cout << "tried: " << model_path << "\n";
    } else {
        cout << "CNN loaded: " << model_path << "\n";
    }

    Mat raw;
    Mat bin;
    std::vector<Light> lights;
    std::vector<ArmorBoard> armors;
    bool is_show_fps = true;
    double fps = 0.0;
    FrameScrubber scrubber;
    int shown_index = 0;
    int last_gray = -1;
    int last_color = -1;
    int last_conf = -1;

    const string win_display = "displayimg";
    const string win_combined = "Frame with Armors";
    const string win_overlay = "armor overlay";
    namedWindow(win_display);
    namedWindow(win_combined);
    namedWindow(win_overlay);
    createTrackbar("frame", win_display, nullptr, trackbar_max, onFrameTrackbar, &scrubber);
    createTrackbar("gray", win_display, &gray_thres, 255);
    createTrackbar("color", win_display, &detect_color, 2);  // 0 both / 1 red / 2 blue
    createTrackbar("conf", win_display, &conf_thres_pct, 100);
    createTrackbar("type", win_overlay, &overlay_type, 1);
    createTrackbar("lr", win_overlay, &overlay_lr, 1);
    createTrackbar("cd", win_overlay, &overlay_cd, 1);
    createTrackbar("ang", win_overlay, &overlay_ang, 1);
    createTrackbar("num", win_overlay, &overlay_num, 1);
    createTrackbar("conf", win_overlay, &overlay_conf, 1);
    createTrackbar("dist", win_overlay, &overlay_dist, 1);

    auto syncTrackbar = [&]() {
        scrubber.syncing = true;
        setTrackbarPos("frame", win_display, shown_index);
        scrubber.syncing = false;
    };

    auto grabAt = [&](int idx) {
        idx = std::clamp(idx, 0, last_index);
        cap.set(CAP_PROP_POS_FRAMES, idx);
        cap >> raw;
        if (raw.empty())
            return false;
        shown_index = idx;
        processArmors(raw, light_detector, armor_detector, bin, lights, armors);
        last_gray = gray_thres;
        last_color = detect_color;
        last_conf = conf_thres_pct;
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
        last_gray = gray_thres;
        last_color = detect_color;
        last_conf = conf_thres_pct;
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

        if (!raw.empty() && (gray_thres != last_gray || detect_color != last_color ||
                             conf_thres_pct != last_conf)) {
            processArmors(raw, light_detector, armor_detector, bin, lights, armors);
            last_gray = gray_thres;
            last_color = detect_color;
            last_conf = conf_thres_pct;
        }

        if (raw.empty() || current_frame.empty() || bin.empty())
            break;

        Mat vis_info = current_frame.clone();
        Mat vis_plain = current_frame.clone();
        for (auto& light : lights) {
            light.draw_light(vis_info);
            light.draw_light(vis_plain);
        }
        for (auto& armor : armors) {
            armor.draw_armor(vis_info);
            armor.draw_armor(vis_plain);
        }
        drawArmorOverlay(vis_info, armors);
        drawNumberThumbnails(vis_info, armors);

        const double mean_gray = current_gray.empty() ? 0.0 : mean(current_gray)[0];
        Mat display_canvas = makeDisplayCanvas(vis_info, is_show_fps ? fps : 0.0, mean_gray);
        Mat shown_display = scaleTo1080(display_canvas, &display_scale);
        imshow(win_display, shown_display);

        Mat bin_bgr;
        if (bin.channels() == 1)
            cvtColor(bin, bin_bgr, COLOR_GRAY2BGR);
        else
            bin_bgr = bin;
        if (bin_bgr.size() != vis_plain.size())
            resize(bin_bgr, bin_bgr, vis_plain.size());

        Mat combined;
        vconcat(vis_plain, bin_bgr, combined);
        combined_y_offset = vis_plain.rows;
        Mat shown_combined = scaleTo1080(combined, &combined_scale);
        imshow(win_combined, shown_combined);
        imshow(win_overlay, makeOverlayLegend());

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
