/**
 * @file light_test.cpp
 * @brief 灰度灯条检测的多窗口播放调试
 *
 * 注释契约：禁止随意删除本文件注释；改窗口布局、HUD 内容、进度条或快捷键时必须改注释。
 *
 * 三个窗口：
 *   displayimg          原图+旋转矩形及可选标注；画面外顶栏左=FPS/整帧灰度均值（带标签），
 *                       右=鼠标 HSV RGB Gray xy（带类型标签，字粗 1）；frame/gray 进度条
 *   Frame with Lights   上半原图+框（无数字），下半灰度二值；整体缩到不超过 1920x1080
 *   light overlay       0/1 进度条：是否在 displayimg 上画灯条高/宽/细长比/tan_xy/颜色
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
constexpr int kGrayThresDefault = 180; // 与 LightParams::gray_thres 默认一致

int gray_thres = kGrayThresDefault;

// light overlay 窗的 0/1 进度条：只改 displayimg 上的文字，不改检测。
// 默认全开，和原先四行都画一致。GTK 进度条下限是 0，所以用 int 而不是 bool。
int overlay_h = 1;      // 高：左上→左下（Light::height）
int overlay_w = 1;      // 宽：左上→右上（Light::width）
int overlay_ratio = 1;  // 细长比 height/width
int overlay_ang = 1;    // tan_xy（右上→右下 Δx/Δy）
int overlay_color = 1;  // red / blue / ?

/** 只在真正播到新帧时更新，暂停时沿用上次，避免 waitKey 循环把 FPS 打满。 */
void updateFps(double& fps, bool enabled) {
    if (!enabled)
        return;
    static auto last = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    fps = 1.0 / std::chrono::duration<double>(now - last).count();
    last = now;
}

/** 主窗口鼠标：先除 view 缩放，再减去顶栏，得到原图像素坐标。 */
Point toDisplayCoord(int x, int y) {
    if (display_scale > 0.0) {
        x = static_cast<int>(x / display_scale + 0.5);
        y = static_cast<int>(y / display_scale + 0.5);
    }
    y -= display_hud_h;
    return Point(x, y);
}

/** 组合窗鼠标：下半二值与上半原图同一套 xy，落在下半时减掉上半高度。 */
Point toCombinedCoord(int x, int y) {
    if (combined_scale > 0.0) {
        x = static_cast<int>(x / combined_scale + 0.5);
        y = static_cast<int>(y / combined_scale + 0.5);
    }
    if (combined_y_offset > 0 && y >= combined_y_offset)
        y -= combined_y_offset;
    return Point(x, y);
}

/** 在未画框的 current_frame 上取 HSV / RGB / 灰度；越界则 show_pixel=false。 */
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
    cvtColor(bgr_mat, hsv_mat, COLOR_BGR2HSV);
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

void onMouseDisplay(int event, int x, int y, int, void*) {
    if (event != EVENT_MOUSEMOVE)
        return;
    const Point p = toDisplayCoord(x, y);
    samplePixel(p.x, p.y);
}

void onMouseCombined(int event, int x, int y, int, void*) {
    if (event != EVENT_MOUSEMOVE)
        return;
    const Point p = toCombinedCoord(x, y);
    samplePixel(p.x, p.y);
}

/** 进度条拖动：GTK 不允许在回调里 setTrackbarPos，用 seek 标志在主循环里跳帧。 */
struct FrameScrubber {
    int want = 0;
    bool seek = false;
    bool syncing = false;  // 程序自己同步滑条时禁止再触发 seek，否则死循环
};

void onFrameTrackbar(int pos, void* userdata) {
    auto* s = static_cast<FrameScrubber*>(userdata);
    if (s->syncing)
        return;
    s->want = pos;
    s->seek = true;
}

// 左/右方向键在 GTK、Qt、Windows 下 key 码不同，再加 a/d。
bool isLeftKey(int key) {
    const int k = key & 0xFF;
    return key == 81 || key == 2 || key == 2424832 || key == 65361 || k == 'a';
}

bool isRightKey(int key) {
    const int k = key & 0xFF;
    return key == 83 || key == 3 || key == 2555904 || key == 65363 || k == 'd';
}

/** 只缩小不放大，保证内容不超过 1920x1080；scale_out 给鼠标坐标反变换用。 */
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

/** 用干净 raw 做检测；gray_thres 来自进度条。 */
void processLights(const Mat& raw, LightDetector& detector, Mat& bin,
                   std::vector<Light>& lights) {
    current_frame = raw.clone();
    cvtColor(current_frame, current_gray, COLOR_BGR2GRAY);
    detector.setGrayThres(gray_thres);
    lights = detector.get_lights(current_frame, &bin);
}

/**
 * 主窗口画面内按 overlay_* 开关逐行画（字粗 1，带标签；关掉某一行就上移，不留空行）：
 *   H     = Light::height（左上→左下）
 *   W     = Light::width（左上→右上）
 *   ratio = height / width
 *   tan   = Light::tan_xy（右上→右下 Δx/Δy）
 *   color = red / blue / ?
 */
void drawLightRotatedInfo(Mat& img, const std::vector<Light>& lights) {
    const int line_h = 12;
    for (const auto& light : lights) {
        const int nlines = (overlay_h ? 1 : 0) + (overlay_w ? 1 : 0) +
                           (overlay_ratio ? 1 : 0) + (overlay_ang ? 1 : 0) +
                           (overlay_color ? 1 : 0);
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

        Point org(cvRound(light.center().x + 6), cvRound(light.center().y));
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
            putText(img, cv::format("ratio:%.2f", light.hw_ratio()), Point(org.x, y),
                    FONT_HERSHEY_PLAIN, 0.9, Scalar(0, 255, 255), kFontThick, LINE_AA);
            y += line_h;
        }
        if (overlay_ang) {
            putText(img, cv::format("tan:%.2f", light.tan_xy), Point(org.x, y),
                    FONT_HERSHEY_PLAIN, 0.9, Scalar(0, 255, 255), kFontThick, LINE_AA);
            y += line_h;
        }
        if (overlay_color) {
            putText(img, color_text, Point(org.x, y), FONT_HERSHEY_PLAIN, 0.9,
                    color_scalar, kFontThick, LINE_AA);
        }
    }
}

/**
 * overlay 窗的说明图：进度条名字太短，用画面写清 0/1 含义和几何定义。
 * Hershey 无中文。每帧 imshow 一次，避免空窗被部分后端丢掉。
 */
Mat makeOverlayLegend() {
    Mat panel(72, 440, CV_8UC3, Scalar(28, 28, 28));
    putText(panel, "0=hide  1=show   (displayimg only)", Point(8, 20),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(0, 200, 100), kFontThick, LINE_AA);
    putText(panel, "H=TL-BL  W=TL-TR  ratio=H/W", Point(8, 42),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(0, 255, 255), kFontThick, LINE_AA);
    putText(panel, "tan=dx/dy(TR-BR)  color=red/blue", Point(8, 64),
            FONT_HERSHEY_SIMPLEX, 0.42, Scalar(0, 255, 255), kFontThick, LINE_AA);
    return panel;
}

/**
 * 顶栏画在画面外（带类型标签，Hershey 无中文所以用英文）：
 *   左 = FPS（帧率）+ meanGray（整帧灰度均值）
 *   右 = HSV(H∈[0,180]) RGB Gray xy（鼠标处像素，相对原图）
 * 十字丝画在画面内鼠标处，数字不叠在图上。
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
    setMouseCallback(win_display, onMouseDisplay, nullptr);
    setMouseCallback(win_combined, onMouseCombined, nullptr);
    createTrackbar("frame", win_display, nullptr, trackbar_max, onFrameTrackbar, &scrubber);
    createTrackbar("gray", win_display, &gray_thres, 255);
    // overlay 全是 0/1：只控制 displayimg 文字，不触发重检。max 必须 >=1（GTK）。
    createTrackbar("H", win_overlay, &overlay_h, 1);
    createTrackbar("W", win_overlay, &overlay_w, 1);
    createTrackbar("ratio", win_overlay, &overlay_ratio, 1);
    createTrackbar("tan", win_overlay, &overlay_ang, 1);
    createTrackbar("color", win_overlay, &overlay_color, 1);

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
        processLights(raw, detector, bin, lights);
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
        processLights(raw, detector, bin, lights);
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
                updateFps(fps, is_show_fps);
        }

        // 暂停时改 gray 也立刻重检，否则二值窗看起来像没跟上。
        if (!raw.empty() && gray_thres != last_gray) {
            processLights(raw, detector, bin, lights);
            last_gray = gray_thres;
        }

        if (raw.empty() || current_frame.empty() || bin.empty())
            break;

        Mat vis_info = current_frame.clone();
        Mat vis_plain = current_frame.clone();
        for (auto& light : lights) {
            light.draw_light(vis_info);
            light.draw_light(vis_plain);
        }
        drawLightRotatedInfo(vis_info, lights);

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
