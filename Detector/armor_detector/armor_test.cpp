/**
 * @file armor_test.cpp
 * @brief 灯条 + 装甲板配对 / 数字 CNN / PnP 的播放调试（不改灯条检测源码）
 *
 * 注释契约：禁止随意删除本文件注释；改窗口布局、HUD、进度条或快捷键时必须改注释。
 * overlay 画的 type/lr/cd/ang/dAng/num/dist/nml/reproj 必须读 ArmorBoard 的同一属性，与配对/分类/PnP
 * 计算方式相同；禁止显示时另算中心距、长度比或从 rvec 另推法线。灯条框仍读 Light 字段。
 * 每个函数须写清作用、输入、输出；调用处也要注明。
 *
 * 四个窗口：
 *   displayimg          原图+灯条框+装甲板框及可选标注；顶栏 FPS/meanGray；只留 frame 进度条
 *                       左上角贴 number_img（默认数字 ROI 的 Otsu 二值，与 CNN 输入同一张）
 *                       nml 开时画品红箭头：远离相机的法线，默认略朝下（读 ArmorBoard::normal_cam）
 *                       reproj 开时画半透明色块：PnP 物点按 rvec/tvec 投回（ArmorBoard::reprojectQuad）
 *                       画面按 1920×1080 可用区域缩小（已扣标题栏和 1 条进度条）
 *   Frame with Armors   上半原图+框（无数字），下半灰度二值；上下拼接后约 2 倍帧高，
 *                       整体缩到不超过 1920×1080（已扣标题栏/底栏，与 displayimg 同一套上限）
 *   armor params        gray / hsv_f / color / conf：灰度阈值、HSV 占比过滤、配色、CNN 置信度
 *                       （conf 不含数字 6，数字 6 固定 60%；从大窗拆出，避免挤占画面）
 *   armor overlay       0/1 进度条：type / 灯条长度比 / 归一化中心距 / 板倾角 / 灯条倾角差 / 数字 / 距离 / 法线 / 重投影色块
 *
 * 快捷键：空格暂停，a/d 或方向键逐帧，1 开关帧率，q/Esc 退出。
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
#include "armor_detector.h"
#include "light_detector.h"
#include "Detector/detector_config.h"

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
constexpr int kMaxWinW = 1920;         // 屏幕宽上限；窗口（含边框）不得超过
constexpr int kMaxWinH = 1080;         // 屏幕高上限；组合大窗约 2 倍帧高，必须按这个缩
constexpr int kWinChromeW = 40;        // 左右边框，避免贴齐 1920 仍被窗框撑出屏幕
constexpr int kWinTitleH = 40;         // 标题栏
constexpr int kTaskbarH = 48;          // 屏幕底栏余量，大窗贴底会被挡住
constexpr int kTrackbarH = 46;         // GTK 一条进度条大约高度
constexpr int kDisplayTrackbars = 1;   // displayimg 只留 frame；gray/color/conf 在 armor params

int gray_thres = LightParams::kGrayThresDefault;
int hsv_filter = 1;       // 1=按 Light::hsv_ratio 丢灯，0=关；与 LightParams::use_hsv_filter 同源
int detect_color = 0;     // 0=红蓝都配，1=只红，2=只蓝；与 ArmorParams::detect_color 同源
int conf_thres_pct = 80;  // CNN 置信度门槛 ×100（须大于该值）；写入 classifier_threshold；数字 6 不读这项

// overlay 0/1：只改 displayimg 文字，不改检测。GTK 进度条下限是 0。
int overlay_type = 1;   // ArmorBoard::type
int overlay_lr = 1;     // ArmorBoard::light_ratio，与 isArmor 同一字段
int overlay_cd = 1;     // ArmorBoard::center_distance
int overlay_ang = 1;    // ArmorBoard::angle（两中心连线相对水平）
int overlay_dAng = 1;   // ArmorBoard::light_angle_diff，与配对同一字段
int overlay_num = 1;    // classification_result（已含数字和概率，不再另画 p）
int overlay_dist = 1;   // ArmorBoard::dist_m（PnP）
int overlay_nml = 1;    // ArmorBoard::draw_normal，读 normal_cam（远离相机）
int overlay_reproj = 1; // ArmorBoard::reprojectQuad：PnP 物点投回的半透明板面

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
 * 作用：只缩小不放大，让整窗落在 1920×1080 里。
 * 原理：WINDOW_AUTOSIZE 的窗高 = 图高 + 进度条；组合大窗上下拼接约 2 倍帧高，
 * 若不扣标题栏/底栏/进度条，图本身缩到 1080 后整窗仍会超出屏幕。
 * 输入：img 任意尺寸；scale_out 非空时写出缩放比；n_trackbars 为该窗进度条条数
 * （displayimg 为 1，Frame with Armors 为 0；gray/color/conf 不在大窗上）。
 * 输出：缩小后的图，或原图（扣完装饰后仍不超过上限）。
 */
Mat scaleTo1080(const Mat& img, double* scale_out, int n_trackbars) {
    const int max_w = std::max(1, kMaxWinW - kWinChromeW);
    const int max_h = std::max(1, kMaxWinH - kWinTitleH - kTaskbarH - n_trackbars * kTrackbarH);
    const double s = std::min({1.0,
                               static_cast<double>(max_w) / std::max(1, img.cols),
                               static_cast<double>(max_h) / std::max(1, img.rows)});
    if (scale_out)
        *scale_out = s;
    if (s >= 1.0)
        return img;
    Mat out;
    resize(img, out, Size(), s, s, INTER_AREA);  // 大图 → 1920×1080 可用区域，INTER_AREA 缩小更稳
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
 * 输入：raw 为 BGR；light_detector 用 gray_thres / hsv_filter；armor_detector 用 detect_color/conf。
 * 输出：bin=灯条二值；lights/armors 为检测结果。
 */
void processArmors(const Mat& raw, LightDetector& light_detector, ArmorDetector& armor_detector,
                   Mat& bin, std::vector<Light>& lights, std::vector<ArmorBoard>& armors) {
    current_frame = raw.clone();
    cvtColor(current_frame, current_gray, COLOR_BGR2GRAY);
    light_detector.setGrayThres(gray_thres);  // 进度条 int → 灯条检测器 [0,255]
    light_detector.setUseHsvFilter(hsv_filter != 0);  // hsv_f 0/1 → 是否用 hsv_ratio 丢灯
    lights = light_detector.get_lights(current_frame, &bin);  // BGR → 灯条 + 同源二值

    ArmorParams p = armor_detector.params();
    p.detect_color = detect_color;
    p.classifier_threshold = static_cast<float>(conf_thres_pct) / 100.f;
    armor_detector.setParams(p);
    // get_armors：灯条列表 + BGR 帧 → 配对/数字/PnP 后的装甲板
    armors = armor_detector.get_armors(lights, current_frame);
}

/**
 * 作用：把每块板的 PnP 重投影四边形画成半透明色块，核对物点+位姿是否贴上灯条框。
 * 输入：img 为 BGR；armors 为 get_armors 结果。输出：无。四角必须读 reprojectQuad，禁止另写物点。
 */
void drawPnpReprojectQuads(Mat& img, const std::vector<ArmorBoard>& armors) {
    const Scalar kFill[] = {
        Scalar(0, 220, 0),
        Scalar(0, 140, 255),
        Scalar(0, 255, 255),
        Scalar(255, 0, 255),
    };
    constexpr double kAlpha = 0.38;
    Mat wash = img.clone();
    std::vector<std::vector<Point>> outlines;
    std::vector<Scalar> colors;
    for (size_t i = 0; i < armors.size(); ++i) {
        const auto q = armors[i].reprojectQuad();  // rvec/tvec + 与 solvePnP 同一物点 → 像素四角
        if (q.size() != 4)
            continue;
        std::vector<Point> poly;
        poly.reserve(4);
        bool skip = false;
        for (const auto& p : q) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y)) {
                skip = true;
                break;
            }
            poly.emplace_back(cvRound(p.x), cvRound(p.y));
        }
        if (skip)
            continue;
        const Scalar& c = kFill[i % 4];
        fillConvexPoly(wash, poly, c, LINE_AA);
        outlines.push_back(std::move(poly));
        colors.push_back(c);
    }
    if (outlines.empty())
        return;
    addWeighted(wash, kAlpha, img, 1.0 - kAlpha, 0.0, img);
    // 轮廓 1px，与 draw_armor 灯条框同粗，错开的缝才看得清。
    for (size_t i = 0; i < outlines.size(); ++i)
        polylines(img, outlines[i], true, colors[i], 1, LINE_AA);
}

/**
 * 作用：按 overlay 开关在装甲板旁画 type/lr/cd/ang/dAng/num/dist，并可选画远离相机的法线。
 * 输入：img 为已画框的 BGR；armors 为 get_armors 的结果。
 * 输出：无返回值。数字必须读 ArmorBoard 字段，禁止另算中心距、长度比、灯条倾角差或法线。
 */
void drawArmorOverlay(Mat& img, const std::vector<ArmorBoard>& armors) {
    if (overlay_nml) {
        for (const auto& armor : armors)
            armor.draw_normal(img);  // 读 normal_cam，品红箭头远离相机
    }

    const int line_h = 14;
    for (const auto& armor : armors) {
        const int nlines = (overlay_type ? 1 : 0) + (overlay_lr ? 1 : 0) +
                           (overlay_cd ? 1 : 0) + (overlay_ang ? 1 : 0) +
                           (overlay_dAng ? 1 : 0) +
                           (overlay_num ? 1 : 0) +
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
        if (overlay_dAng) {
            // light_angle_diff：|Light::angle 左 - Light::angle 右|，与 isArmor 同一字段。
            putText(img, cv::format("dAng:%.1f", armor.light_angle_diff), Point(org.x, y),
                    FONT_HERSHEY_PLAIN, 0.95, text, kFontThick, LINE_AA);
            y += line_h;
        }
        if (overlay_num) {
            // classification_result 已是 "3:_92.0%"，与 confidence 同源，不再另画 p。
            const string num = armor.classification_result.empty()
                                   ? (armor.number > 0 ? std::to_string(armor.number) : "?")
                                   : armor.classification_result;
            putText(img, num, Point(org.x, y), FONT_HERSHEY_PLAIN, 0.95, text, kFontThick, LINE_AA);
            y += line_h;
        }
        if (overlay_dist) {
            const string d = armor.pnp_ok ? cv::format("d:%.2fm", armor.dist_m) : "d:fail";
            putText(img, d, Point(org.x, y), FONT_HERSHEY_PLAIN, 0.95, text, kFontThick, LINE_AA);
        }
    }
}

/**
 * 作用：把 number_img 贴到画面左上（大板 20×28，小板 26×28），核对与 CNN 输入是否同一张。
 * 输入：img 为 BGR 画布；armors 带 number_img。输出：无。禁止另做一遍二值。
 */
void drawNumberThumbnails(Mat& img, const std::vector<ArmorBoard>& armors) {
    int x = 8;
    const int y = 8;
    for (const auto& armor : armors) {
        if (armor.number_img.empty())
            continue;
        Mat vis;
        cvtColor(armor.number_img, vis, COLOR_GRAY2BGR);  // number_img 单通道 → BGR，不另做二值
        resize(vis, vis, Size(40, 56), 0, 0, INTER_NEAREST);  // 最近邻放大，二值边缘保持 0/255
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
    Mat panel(222, 520, CV_8UC3, Scalar(28, 28, 28));
    putText(panel, "0=hide  1=show   (displayimg only)", Point(8, 22),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(0, 200, 100), kFontThick, LINE_AA);
    putText(panel, "type=SMALL/LARGE from center_distance", Point(8, 44),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(0, 255, 255), kFontThick, LINE_AA);
    putText(panel, "lr=min(H)/max(H)  cd=centerDist/avg H", Point(8, 66),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(0, 255, 255), kFontThick, LINE_AA);
    putText(panel, "ang=center-line vs horizontal (deg)", Point(8, 88),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(0, 255, 255), kFontThick, LINE_AA);
    putText(panel, "dAng=|L.angle-R.angle|  drop if >max_dAng", Point(8, 110),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(0, 255, 255), kFontThick, LINE_AA);
    putText(panel, "num=CNN 1-8 with %  dist=PnP ||t|| meters", Point(8, 132),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(0, 255, 255), kFontThick, LINE_AA);
    putText(panel, "nml=magenta  away-from-cam normal (slightly down)", Point(8, 154),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(255, 0, 255), kFontThick, LINE_AA);
    putText(panel, "reproj=PnP object quad (rvec/tvec, same pts as solve)", Point(8, 176),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(0, 220, 0), kFontThick, LINE_AA);
    putText(panel, "H/angle=Light fields  same as matching", Point(8, 198),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(180, 180, 180), kFontThick, LINE_AA);
    return panel;
}

/**
 * 作用：给 armor params 窗写 gray/hsv_f/color/conf 的含义（进度条名字太短）。
 * 输入：无。输出：BGR 说明图，每帧 imshow，避免空窗被后端丢掉。
 * 原理：这几项从 displayimg 拆出，检测仍读同一全局变量，只是进度条换窗。
 */
Mat makeParamsLegend() {
    Mat panel(154, 520, CV_8UC3, Scalar(28, 28, 28));
    putText(panel, "gray=Light gray thres  (get_bin)", Point(8, 22),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(0, 255, 255), kFontThick, LINE_AA);
    putText(panel, "hsv_f 0=off 1=on  (HSV ratio gate, Light.hsv_ratio)", Point(8, 44),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(0, 255, 255), kFontThick, LINE_AA);
    putText(panel, "color 0=both 1=red 2=blue  (ArmorParams)", Point(8, 66),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(0, 255, 255), kFontThick, LINE_AA);
    putText(panel, "conf=CNN must >N%  (not digit 6)", Point(8, 88),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(0, 255, 255), kFontThick, LINE_AA);
    putText(panel, "digit6 gate=60% fixed  (classifier_threshold_digit6)", Point(8, 110),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(0, 255, 255), kFontThick, LINE_AA);
    putText(panel, "same fields as matching / classify", Point(8, 132),
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
 * 输入：可选 argv[1] 为 YAML；stdin 读 0–4 选视频。输出：0 成功，非 0 打不开视频或找不到模型时仍可跑几何。
 */
int main(int argc, char** argv) {
    const string names[] = {
        "2025sj.avi",
        "hero_blue.avi",
        "Infantry_blue.avi",
        "Infantry_red1.avi",
        "Infantry_red2.avi",
        "2026_10.mp4",
         "2026_TDT_4.mp4"
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
    if (i < 0 || i > names->length()-1) {
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
    const string cfg_preferred = (argc >= 2) ? argv[1] : string();
    const string cfg_path = findDetectorConfigPath(cfg_preferred);
    if (!cfg_path.empty()) {
        const bool light_ok = light_detector.loadConfig(cfg_path);
        const bool armor_ok = armor_detector.loadConfig(cfg_path);
        const bool pnp_ok = ArmorBoard::loadPnpConfig(cfg_path);
        if (light_ok && armor_ok && pnp_ok) {
            cout << "config: " << cfg_path << "\n";
            gray_thres = light_detector.grayThres();
            hsv_filter = light_detector.useHsvFilter() ? 1 : 0;
            detect_color = armor_detector.params().detect_color;
            conf_thres_pct = std::clamp(
                static_cast<int>(std::lround(armor_detector.params().classifier_threshold * 100.f)),
                0, 100);
        } else {
            cout << "config incomplete: " << cfg_path << "\n";
        }
    } else {
        cout << "config not loaded; using built-in defaults\n";
    }
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
    int last_hsv_filter = -1;
    int last_color = -1;
    int last_conf = -1;

    const string win_display = "displayimg";
    const string win_combined = "Frame with Armors";
    const string win_params = "armor params";
    const string win_overlay = "armor overlay";
    namedWindow(win_display);  // 只留 frame，保持 AUTOSIZE，尺寸由缩放后的图画布决定
    namedWindow(win_combined, WINDOW_NORMAL);  // 组合大窗：允许 resizeWindow 钉在 1920×1080 内
    namedWindow(win_params);  // gray/hsv_f/color/conf 从大窗拆到这里，避免进度条占画面高度
    namedWindow(win_overlay);
    createTrackbar("frame", win_display, nullptr, trackbar_max, onFrameTrackbar, &scrubber);
    createTrackbar("gray", win_params, &gray_thres, 255);  // 灯条 get_bin 灰度门槛
    createTrackbar("hsv_f", win_params, &hsv_filter, 1);  // 0=关 HSV 占比过滤，1=开
    createTrackbar("color", win_params, &detect_color, 2);  // 0 both / 1 red / 2 blue
    createTrackbar("conf", win_params, &conf_thres_pct, 100);  // CNN 须严格大于该百分比；数字 6 固定 60%
    createTrackbar("type", win_overlay, &overlay_type, 1);
    createTrackbar("lr", win_overlay, &overlay_lr, 1);
    createTrackbar("cd", win_overlay, &overlay_cd, 1);
    createTrackbar("ang", win_overlay, &overlay_ang, 1);
    createTrackbar("dAng", win_overlay, &overlay_dAng, 1);
    createTrackbar("num", win_overlay, &overlay_num, 1);
    createTrackbar("dist", win_overlay, &overlay_dist, 1);
    createTrackbar("nml", win_overlay, &overlay_nml, 1);  // 0/1：品红法线箭头，读 normal_cam
    createTrackbar("reproj", win_overlay, &overlay_reproj, 1);  // 0/1：PnP 重投影半透明板，读 reprojectQuad

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
        last_hsv_filter = hsv_filter;
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
        last_hsv_filter = hsv_filter;
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

        if (!raw.empty() && (gray_thres != last_gray || hsv_filter != last_hsv_filter ||
                             detect_color != last_color || conf_thres_pct != last_conf)) {
            processArmors(raw, light_detector, armor_detector, bin, lights, armors);
            last_gray = gray_thres;
            last_hsv_filter = hsv_filter;
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
        if (overlay_reproj) {
            // 先铺 PnP 色块，再画灯条端点框，错开的边就是重投影误差。
            drawPnpReprojectQuads(vis_info, armors);
            drawPnpReprojectQuads(vis_plain, armors);
        }
        for (auto& armor : armors) {
            armor.draw_armor(vis_info);
            armor.draw_armor(vis_plain);
        }
        drawArmorOverlay(vis_info, armors);  // 文字 + 法线箭头（nml 开时读 normal_cam）
        if (overlay_nml) {
            for (const auto& armor : armors)
                armor.draw_normal(vis_plain);  // 组合窗上半无文字，仍画同一条远离相机的法线
        }
        drawNumberThumbnails(vis_info, armors);

        const double mean_gray = current_gray.empty() ? 0.0 : mean(current_gray)[0];
        Mat display_canvas = makeDisplayCanvas(vis_info, is_show_fps ? fps : 0.0, mean_gray);
        // displayimg 只留 frame 一条进度条，扣掉后再缩到 1920×1080 可用区域
        Mat shown_display = scaleTo1080(display_canvas, &display_scale, kDisplayTrackbars);
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
        // 上下拼接约 2 倍帧高：无进度条，仍扣标题栏/底栏，缩到不超过 1920×1080
        Mat shown_combined = scaleTo1080(combined, &combined_scale, 0);
        imshow(win_combined, shown_combined);
        resizeWindow(win_combined, shown_combined.cols, shown_combined.rows);  // 钉死缩放后尺寸，GTK AUTOSIZE 有时不随图缩小
        imshow(win_params, makeParamsLegend());  // 无输入 → gray/hsv_f/color/conf 说明图
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
