/**
 * @file light_test.cpp
 * @brief 灰度灯条检测的多窗口播放调试
 *
 * 注释契约：禁止随意删除本文件注释；改窗口布局、HUD 内容、进度条或快捷键时必须改注释。
 * overlay 画的 H/W/ratio/ang/color/hsv 必须读 Light 的同一属性或函数，与检测筛选
 * 计算方式相同；禁止显示时另写 height/width 或另搞一套细长比 / HSV 占比。
 * 框内 HSV 三通道均值不是筛灯条量、不是 Light 字段：只用 pts 在未画框原图上现算，供调试。
 * 每个函数须写清作用、输入、输出；调用处也要注明调谁、送什么、得到什么。
 *
 * 三个窗口：
 *   displayimg          原图+旋转矩形及可选标注；画面外顶栏左=FPS/整帧灰度均值（带标签），
 *                       右=鼠标 HSV RGB Gray xy（带类型标签，字粗 1）；frame/gray 进度条
 *   Frame with Lights   上半原图+框（无数字），下半灰度二值；整体缩到不超过 1920x1080
 *   light overlay       0/1 进度条：是否在 displayimg 上画灯条高/宽/细长比/倾角/颜色/HSV占比
 *                       以及框内 H/S/V 均值（三者同一开关 hsv_m）
 *
 * 快捷键：空格暂停，a/d 或方向键逐帧，1 开关帧率，q/Esc 退出。
 */
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#include "Light.h"
#include "light_detector.h"

using namespace cv;
using namespace std;

// 当前帧保持「未画框」的原图，取像素 / 灰度均值都看它，避免框线污染。
Mat current_frame;
Mat current_gray;
Scalar hsv_value;       // 鼠标处 HSV，H∈[0,180]
Vec3b mouse_bgr;        // 鼠标处 BGR，显示时改成 R G B
int mouse_gray = 0;
Point mouse_pos;        // 相对原图（不是窗口像素）
bool show_pixel = false;

bool paused = false;

// 主窗口：顶栏 HUD + 画面；组合窗口：上半原图、下半二值。鼠标坐标要先除缩放再减偏移。
int display_hud_h = 0;
int combined_y_offset = 0;
double display_scale = 1.0;
double combined_scale = 1.0;

constexpr int kHudH = 36;              // 画面外顶栏高度（像素，缩放前）
constexpr int kFontThick = 1;          // 鼠标数值和灯条标注统一字粗 1
constexpr int kMaxWinW = 1920;         // 窗口内容宽上限
constexpr int kMaxWinH = 1080;         // 窗口内容高上限，含拼接后的组合窗
// 进度条初值跟检测器默认同源；。
int gray_thres = LightParams::kGrayThresDefault;

// light overlay 窗的 0/1 进度条：只改 displayimg 上的文字，不改检测。
// 默认全开，和原先四行都画一致。GTK 进度条下限是 0，所以用 int 而不是 bool。
int overlay_h = 1;      // 显示 Light::height，与检测同一字段
int overlay_w = 1;      // 显示 Light::width，与检测同一字段
int overlay_ratio = 1;  // 显示 Light::ratio，禁止另算 height/width
int overlay_ang = 1;    // 显示 Light::angle，与检测同一字段
int overlay_color = 1;  // 显示 Light::color（1 red / 2 blue）
int overlay_hsv = 1;    // 显示 Light::hsv_ratio，与检测同一字段
int overlay_hsv_mean = 1;  // 同时显示框内 H/S/V 三通道均值（调试用，不参与筛选）

/**
 * 作用：只在真正播出新帧时刷新 FPS，暂停时沿用上次，避免 waitKey 把读数打满。
 * 输入：fps 为要写入的变量；enabled 为 false 则不改 fps。
 * 输出：通过 fps 写回 1/Δt（秒⁻¹）。
 */
void updateFps(double& fps, bool enabled) {
    if (!enabled)
        return;
    static auto last = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    fps = 1.0 / std::chrono::duration<double>(now - last).count();
    last = now;
}

/**
 * 作用：把 displayimg 窗口鼠标坐标映回原图像素。
 * 输入：x/y 为窗口像素（含顶栏）。输出：相对未缩放原图的点（已减顶栏）。
 */
Point toDisplayCoord(int x, int y) {
    if (display_scale > 0.0) {
        x = static_cast<int>(x / display_scale + 0.5);
        y = static_cast<int>(y / display_scale + 0.5);
    }
    y -= display_hud_h;
    return Point(x, y);
}

/**
 * 作用：把组合窗鼠标坐标映回原图像素（上下两半共用一套 xy）。
 * 输入：x/y 为组合窗像素。输出：相对原图的点；落在下半时已减上半高度。
 */
Point toCombinedCoord(int x, int y) {
    if (combined_scale > 0.0) {
        x = static_cast<int>(x / combined_scale + 0.5);
        y = static_cast<int>(y / combined_scale + 0.5);
    }
    if (combined_y_offset > 0 && y >= combined_y_offset)
        y -= combined_y_offset;
    return Point(x, y);
}

/**
 * 作用：在未画框的 current_frame 上采样鼠标处 HSV/RGB/灰度，供顶栏显示。
 * 输入：x/y 为原图像素。输出：写入 mouse_* / hsv_value；越界则 show_pixel=false。
 */
void samplePixel(int x, int y) {
    mouse_pos = Point(x, y);
    show_pixel = false;
    if (current_frame.empty())
        return;
    if (x < 0 || y < 0 || x >= current_frame.cols || y >= current_frame.rows)
        return;

    mouse_bgr = current_frame.at<Vec3b>(y, x);
    Mat bgr_mat(1, 1, CV_8UC3, Scalar(mouse_bgr[0], mouse_bgr[1], mouse_bgr[2]));
    Mat hsv_mat;
    cvtColor(bgr_mat, hsv_mat, COLOR_BGR2HSV);  // 1x1 BGR → HSV，H∈[0,180]
    const Vec3b hsv = hsv_mat.at<Vec3b>(0, 0);
    hsv_value = Scalar(hsv[0], hsv[1], hsv[2]);

    // 灰度与 cvtColor(BGR2GRAY) 一致（BT.601），current_gray 空时手算兜底。
    if (!current_gray.empty() && y < current_gray.rows && x < current_gray.cols)
        mouse_gray = current_gray.at<uchar>(y, x);
    else
        mouse_gray = static_cast<int>(0.114f * mouse_bgr[0] + 0.587f * mouse_bgr[1] +
                                      0.299f * mouse_bgr[2] + 0.5f);
    show_pixel = true;
}

/**
 * 作用：displayimg 鼠标移动回调，把窗口坐标换成原图像素再采样。
 * 输入：OpenCV 鼠标事件；只用 EVENT_MOUSEMOVE。输出：更新顶栏像素信息。
 */
void onMouseDisplay(int event, int x, int y, int, void*) {
    if (event != EVENT_MOUSEMOVE)
        return;
    // toDisplayCoord：窗口 xy → 原图像素；samplePixel：该像素 → HSV/RGB/Gray。
    const Point p = toDisplayCoord(x, y);
    samplePixel(p.x, p.y);
}

/**
 * 作用：组合窗鼠标移动回调，上下半用同一套原图 xy 采样。
 * 输入：OpenCV 鼠标事件。输出：更新顶栏像素信息。
 */
void onMouseCombined(int event, int x, int y, int, void*) {
    if (event != EVENT_MOUSEMOVE)
        return;
    // toCombinedCoord：组合窗 xy → 原图像素（下半减上半高）。
    const Point p = toCombinedCoord(x, y);
    samplePixel(p.x, p.y);
}

/** 进度条拖动：GTK 不允许在回调里 setTrackbarPos，用 seek 标志在主循环里跳帧。 */
struct FrameScrubber {
    int want = 0;
    bool seek = false;
    bool syncing = false;  // 程序自己同步滑条时禁止再触发 seek，否则死循环
};

/**
 * 作用：frame 进度条回调。GTK 禁止在回调里 setTrackbarPos，只置 seek，主循环再跳帧。
 * 输入：pos 为目标帧号；userdata 为 FrameScrubber*。输出：写入 want/seek。
 */
void onFrameTrackbar(int pos, void* userdata) {
    auto* s = static_cast<FrameScrubber*>(userdata);
    if (s->syncing)
        return;
    s->want = pos;
    s->seek = true;
}

// 左/右方向键在 GTK、Qt、Windows 下 key 码不同，再加 a/d。
/**
 * 作用：判断是否为「上一帧」键。
 * 输入：waitKeyEx 原始键码。输出：true 表示左方向键或 a。
 */
bool isLeftKey(int key) {
    const int k = key & 0xFF;
    return key == 81 || key == 2 || key == 2424832 || key == 65361 || k == 'a';
}

/**
 * 作用：判断是否为「下一帧」键。
 * 输入：waitKeyEx 原始键码。输出：true 表示右方向键或 d。
 */
bool isRightKey(int key) {
    const int k = key & 0xFF;
    return key == 83 || key == 3 || key == 2555904 || key == 65363 || k == 'd';
}

/**
 * 作用：只缩小不放大，保证窗口内容不超过 1920×1080。
 * 输入：img 任意尺寸；scale_out 非空时写出缩放比（给鼠标反变换）。
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
    resize(img, out, Size(), s, s, INTER_AREA);  // 大图 → ≤1080p，INTER_AREA 缩小更稳
    return out;
}

/**
 * 作用：对干净原图跑灯条检测，并刷新采样用的 current_frame / current_gray。
 * 输入：raw 为 BGR 帧；detector 会写入 gray_thres；bin/lights 为输出槽。
 * 输出：bin=与检测同一张二值图；lights=通过几何+RGB+HSV 占比的灯条。
 */
void processLights(const Mat& raw, LightDetector& detector, Mat& bin,
                   std::vector<Light>& lights) {
    current_frame = raw.clone();
    cvtColor(current_frame, current_gray, COLOR_BGR2GRAY);  // BGR → 灰度，供顶栏 Gray/meanGray
    detector.setGrayThres(gray_thres);  // 进度条 int → 检测器 [0,255] 阈值
    // get_lights：BGR 帧 → lights；&bin 写出 get_bin 的二值图（阈值+闭运算）。
    lights = detector.get_lights(current_frame, &bin);
}

/**
 * 作用：求旋转矩形栅格内 HSV 三通道算术均值，仅供 overlay 调试，不写入 Light、不参与筛灯条。
 * 输入：hsv 为整帧 COLOR_BGR2HSV 结果（须与灯条 pts 同一分辨率）；pts 为 Light 四顶点
 *      （左上/右上/左下/右下，函数内改成绕序再填凸包）。
 * 输出：Scalar(H均值,S均值,V均值)，H∈[0,180] 算术平均（不做 0/180 环绕）；框栅格面积为 0 时全 0。
 */
Scalar meanHsvInRotatedRect(const Mat& hsv, const Point2f pts[4]) {
    // pts 不是绕序；fillConvexPoly 必须按凸包绕序，否则对角线交叉、均值范围不对。
    const Point ipts[4] = {pts[0], pts[1], pts[3], pts[2]};  // 左上→右上→右下→左下

    // zeros+fillConvexPoly：四顶点 → 与 hsv 同尺寸的旋转矩形栅格掩码。
    Mat poly = Mat::zeros(hsv.size(), CV_8UC1);
    fillConvexPoly(poly, ipts, 4, 255);
    if (countNonZero(poly) <= 0)
        return Scalar(0, 0, 0);

    // mean：掩码内 HSV 像素 → 三通道算术均值（与顶栏单像素 HSV 同一 COLOR_BGR2HSV）。
    return mean(hsv, poly);
}

/**
 * 作用：按 overlay 开关在主画面灯条旁画 H/W/ratio/ang/color/hsv 及框内 HSV 均值
 *      （关一行就上移，不留空；Hm/Sm/Vm 由同一开关控制）。
 * 输入：img 为已画框的 BGR 图（就地改）；lights 为 get_lights 的结果。
 * 输出：无返回值。筛灯条用的数字必须读 Light 字段（height/width/ratio/angle/color/hsv_ratio）；
 *      HSV 均值从 current_frame+pts 现算，避免读已画框的 img。
 */
void drawLightRotatedInfo(Mat& img, const std::vector<Light>& lights) {
    const int line_h = 12;

    // 均值看未画框原图：框线会污染 S/V。只在开关打开时转一次 HSV，每个灯条复用。
    Mat hsv_frame;
    if (overlay_hsv_mean && !current_frame.empty())
        cvtColor(current_frame, hsv_frame, COLOR_BGR2HSV);  // 干净 BGR → HSV，H∈[0,180]

    for (const auto& light : lights) {
        const int nlines = (overlay_h ? 1 : 0) + (overlay_w ? 1 : 0) +
                           (overlay_ratio ? 1 : 0) + (overlay_ang ? 1 : 0) +
                           (overlay_color ? 1 : 0) + (overlay_hsv ? 1 : 0) +
                           (overlay_hsv_mean ? 3 : 0);
        if (nlines == 0)
            continue;

        string color_text = "?";
        Scalar color_scalar(0, 255, 255);
        if (light.color == 1) {
            color_text = "red";
            color_scalar = Scalar(0, 0, 255);
        } else if (light.color == 2) {
            color_text = "blue";
            color_scalar = Scalar(255, 0, 0);
        }

        Point org(cvRound(light.center().x + 6), cvRound(light.center().y));  // center()：旋转矩形中心
        org.x = std::clamp(org.x, 2, std::max(2, img.cols - 110));
        org.y = std::clamp(org.y, 12, std::max(12, img.rows - line_h * nlines - 4));

        int y = org.y;
        if (overlay_h) {
            putText(img, cv::format("H:%.1f", light.height), Point(org.x, y),
                    FONT_HERSHEY_PLAIN, 0.9, Scalar(0, 255, 255), kFontThick, LINE_AA);
            y += line_h;
        }
        if (overlay_w) {
            putText(img, cv::format("W:%.1f", light.width), Point(org.x, y),
                    FONT_HERSHEY_PLAIN, 0.9, Scalar(0, 255, 255), kFontThick, LINE_AA);
            y += line_h;
        }
        if (overlay_ratio) {
            putText(img, cv::format("ratio:%.2f", light.ratio), Point(org.x, y),
                    FONT_HERSHEY_PLAIN, 0.9, Scalar(0, 255, 255), kFontThick, LINE_AA);
            y += line_h;
        }
        if (overlay_ang) {
            putText(img, cv::format("ang:%.1f", light.angle), Point(org.x, y),
                    FONT_HERSHEY_PLAIN, 0.9, Scalar(0, 255, 255), kFontThick, LINE_AA);
            y += line_h;
        }
        if (overlay_color) {
            putText(img, color_text, Point(org.x, y), FONT_HERSHEY_PLAIN, 0.9,
                    color_scalar, kFontThick, LINE_AA);
            y += line_h;
        }
        if (overlay_hsv) {
            putText(img, cv::format("hsv:%.2f", light.hsv_ratio), Point(org.x, y),
                    FONT_HERSHEY_PLAIN, 0.9, Scalar(0, 255, 255), kFontThick, LINE_AA);
            y += line_h;
        }
        if (overlay_hsv_mean) {
            // meanHsvInRotatedRect：旋转矩形 ∩ 整帧 HSV → H/S/V 均值；空图则保持 0。
            const Scalar hsv_m = hsv_frame.empty() ? Scalar(0, 0, 0)
                                                   : meanHsvInRotatedRect(hsv_frame, light.pts);
            putText(img, cv::format("Hm:%.1f", hsv_m[0]), Point(org.x, y),
                    FONT_HERSHEY_PLAIN, 0.9, Scalar(0, 255, 255), kFontThick, LINE_AA);
            y += line_h;
            putText(img, cv::format("Sm:%.1f", hsv_m[1]), Point(org.x, y),
                    FONT_HERSHEY_PLAIN, 0.9, Scalar(0, 255, 255), kFontThick, LINE_AA);
            y += line_h;
            putText(img, cv::format("Vm:%.1f", hsv_m[2]), Point(org.x, y),
                    FONT_HERSHEY_PLAIN, 0.9, Scalar(0, 255, 255), kFontThick, LINE_AA);
        }
    }
}

/**
 * 作用：给 overlay 窗生成说明图（进度条名字太短，用画面写 0/1 和几何定义）。
 * 输入：无。输出：116×440 的 BGR 图，每帧 imshow，避免空窗被后端丢掉。
 */
Mat makeOverlayLegend() {
    Mat panel(116, 440, CV_8UC3, Scalar(28, 28, 28));
    putText(panel, "0=hide  1=show   (displayimg only)", Point(8, 20),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(0, 200, 100), kFontThick, LINE_AA);
    putText(panel, "H=TL-BL  W=TL-TR  ratio=Light.ratio", Point(8, 42),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(0, 255, 255), kFontThick, LINE_AA);
    putText(panel, "ang=tilt deg(TR-BR)  color=red/blue", Point(8, 64),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(0, 255, 255), kFontThick, LINE_AA);
    putText(panel, "hsv=Light.hsv_ratio (HSV-in-rect / rect)", Point(8, 86),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(0, 255, 255), kFontThick, LINE_AA);
    putText(panel, "hsv_m=Hm/Sm/Vm mean HSV inside rect", Point(8, 108),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(0, 255, 255), kFontThick, LINE_AA);
    return panel;
}

/**
 * 作用：把检测画面接到带标签的顶栏下面（FPS/meanGray、鼠标 HSV RGB Gray xy）。
 * 输入：vis 为已画灯条的画面；fps、mean_gray 为顶栏左侧数字。
 * 输出：高 = vis.rows+kHudH 的画布；十字丝在画面内，数字在顶栏。
 */
Mat makeDisplayCanvas(const Mat& vis, double fps, double mean_gray) {
    Mat canvas(vis.rows + kHudH, vis.cols, CV_8UC3, Scalar(28, 28, 28));
    vis.copyTo(canvas(Rect(0, kHudH, vis.cols, vis.rows)));
    display_hud_h = kHudH;

    const string left = cv::format("FPS:%.1f  meanGray:%.1f", fps, mean_gray);
    putText(canvas, left, Point(8, 24), FONT_HERSHEY_SIMPLEX, 0.55,
            Scalar(0, 200, 100), kFontThick, LINE_AA);

    if (show_pixel) {
        const int r = mouse_bgr[2];
        const int g = mouse_bgr[1];
        const int b = mouse_bgr[0];
        const string right = cv::format("HSV:(%d,%d,%d)  RGB:(%d,%d,%d)  Gray:%d  xy:(%d,%d)",
                                        static_cast<int>(hsv_value[0]),
                                        static_cast<int>(hsv_value[1]),
                                        static_cast<int>(hsv_value[2]),
                                        r, g, b, mouse_gray, mouse_pos.x, mouse_pos.y);
        const Size tsz = getTextSize(right, FONT_HERSHEY_SIMPLEX, 0.5, kFontThick, nullptr);
        putText(canvas, right, Point(std::max(8, canvas.cols - tsz.width - 8), 24),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), kFontThick, LINE_AA);
    }

    if (show_pixel && mouse_pos.y >= 0 && mouse_pos.y < vis.rows &&
        mouse_pos.x >= 0 && mouse_pos.x < vis.cols) {
        const Point p(mouse_pos.x, mouse_pos.y + kHudH);
        line(canvas, Point(p.x - 5, p.y), Point(p.x + 5, p.y), Scalar(255, 255, 255), 1);
        line(canvas, Point(p.x, p.y - 5), Point(p.x, p.y + 5), Scalar(255, 255, 255), 1);
    }
    return canvas;
}

/**
 * 作用：打开视频，循环检测并显示 displayimg / 组合窗 / overlay。
 * 输入：从 stdin 读 0–4 选视频。输出：进程退出码 0 成功，非 0 打不开视频。
 */
int main() {
    // 相对 cmake-build-debug 时 ../../Video 指向 TDT/Video；多前缀兼容其它工作目录。
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
    // 部分 AVI 报 FRAME_COUNT=0；GTK createTrackbar 不允许 max=0。
    // 无索引时整段扫一遍（启动会慢），进度条范围才准。
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

    Mat raw;
    Mat bin;
    std::vector<Light> lights;
    LightDetector detector;
    bool is_show_fps = true;
    double fps = 0.0;
    FrameScrubber scrubber;
    int shown_index = 0;
    int last_gray = -1;  // 暂停时拖 gray 也要重跑检测

    const string win_display = "displayimg";
    const string win_combined = "Frame with Lights";
    const string win_overlay = "light overlay";
    namedWindow(win_display);
    namedWindow(win_combined);
    namedWindow(win_overlay);
    setMouseCallback(win_display, onMouseDisplay, nullptr);  // 窗口 xy → toDisplayCoord → samplePixel
    setMouseCallback(win_combined, onMouseCombined, nullptr);  // 组合窗 xy → toCombinedCoord → samplePixel
    createTrackbar("frame", win_display, nullptr, trackbar_max, onFrameTrackbar, &scrubber);  // 拖动 → seek 标志
    createTrackbar("gray", win_display, &gray_thres, 255);  // 直接改 gray_thres，无回调；主循环比对 last_gray
    // overlay 全是 0/1：只控制 displayimg 文字，不触发重检。max 必须 >=1（GTK）。
    createTrackbar("H", win_overlay, &overlay_h, 1);
    createTrackbar("W", win_overlay, &overlay_w, 1);
    createTrackbar("ratio", win_overlay, &overlay_ratio, 1);
    createTrackbar("ang", win_overlay, &overlay_ang, 1);
    createTrackbar("color", win_overlay, &overlay_color, 1);
    createTrackbar("hsv", win_overlay, &overlay_hsv, 1);
    createTrackbar("hsv_m", win_overlay, &overlay_hsv_mean, 1);  // 1=同时画 Hm/Sm/Vm，0=三个都藏

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
        processLights(raw, detector, bin, lights);  // raw(BGR) → lights + bin（get_bin 同源）
        last_gray = gray_thres;
        syncTrackbar();
        return true;
    };

    auto grabNext = [&]() {
        cap >> raw;
        if (raw.empty())
            return false;
        const int pos = static_cast<int>(cap.get(CAP_PROP_POS_FRAMES));
        shown_index = std::clamp(pos - 1, 0, last_index);
        processLights(raw, detector, bin, lights);  // 下一帧 BGR → lights + bin
        last_gray = gray_thres;
        syncTrackbar();
        return true;
    };

    while (true) {
        if (scrubber.seek) {
            scrubber.seek = false;
            paused = true;  // 拖进度条后停住，方便看那一帧
            grabAt(scrubber.want);
        } else if (!paused) {
            if (!grabNext())
                paused = true;  // 播到结尾当暂停，不退出
            else
                updateFps(fps, is_show_fps);  // 新帧 Δt → fps；暂停不走这里
        }

        // 暂停时改 gray 也立刻重检，否则二值窗看起来像没跟上。
        if (!raw.empty() && gray_thres != last_gray) {
            processLights(raw, detector, bin, lights);  // 同一帧、新 gray_thres → 重算 lights/bin
            last_gray = gray_thres;
        }

        if (raw.empty() || current_frame.empty() || bin.empty())
            break;

        Mat vis_info = current_frame.clone();
        Mat vis_plain = current_frame.clone();
        for (auto& light : lights) {
            light.draw_light(vis_info);   // Light 四顶点 → 画面上的黄/橙框
            light.draw_light(vis_plain);
        }
        drawLightRotatedInfo(vis_info, lights);  // 读 Light 字段画几何/颜色；可选再画框内 HSV 均值

        const double mean_gray = current_gray.empty() ? 0.0 : mean(current_gray)[0];
        Mat display_canvas = makeDisplayCanvas(vis_info, is_show_fps ? fps : 0.0, mean_gray);  // 画面+顶栏
        Mat shown_display = scaleTo1080(display_canvas, &display_scale);  // 画布 → ≤1080p
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
        Mat shown_combined = scaleTo1080(combined, &combined_scale);  // 上下拼接图 → ≤1080p
        imshow(win_combined, shown_combined);
        imshow(win_overlay, makeOverlayLegend());  // 无输入 → 0/1 说明图

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
