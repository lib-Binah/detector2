/**
 * 注释契约：禁止删本文件注释；改选板规则、飞时迭代或绘制颜色时必须改注释。
 * 瞄准点：飞时内外推车心（kv·v）与当前板相位（kw·ω），再重力抬高；禁止改成飞时后重选正对板。
 * 每个函数须写清作用/输入/输出；调用处注明意图。
 */
#include "aimer.h"

#include "Detector/detector_config.h"

#include <algorithm>
#include <cmath>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace {

constexpr double kPi = 3.14159265358979323846;

/**
 * 作用：相机系 mm → 像素，K/D 与 PnP 同一套。
 * 输入：p_mm。输出：深度<=0 时 (-1,-1)。
 */
cv::Point2f projectCamMm(const cv::Point3d& p_mm) {
    if (p_mm.z <= 1e-3)
        return {-1.f, -1.f};
    const std::vector<cv::Point3f> pts = {cv::Point3f(static_cast<float>(p_mm.x),
                                                      static_cast<float>(p_mm.y),
                                                      static_cast<float>(p_mm.z))};
    const cv::Mat r0 = cv::Mat::zeros(3, 1, CV_64F);
    const cv::Mat t0 = cv::Mat::zeros(3, 1, CV_64F);
    std::vector<cv::Point2f> pix;
    cv::projectPoints(pts, r0, t0, ArmorBoard::cameraMatrix(), ArmorBoard::distCoeffs(), pix);
    if (pix.empty())
        return {-1.f, -1.f};
    return pix[0];
}

/**
 * 作用：角折到 (-π, π]。输入：弧度。输出：主值。
 */
double limitRad(double a) {
    while (a > kPi)
        a -= 2.0 * kPi;
    while (a < -kPi)
        a += 2.0 * kPi;
    return a;
}

/**
 * 作用：板在飞时后相对视线的朝向误差（越小越正对相机）。
 * 输入：xyza 一项 (x,y,z,yaw) 米。输出：弧度；在相机后方则很大。
 */
double facingErr(const Eigen::Vector4d& p) {
    if (p[2] < 0.05)
        return 1e9;
    const double los = std::atan2(p[0], p[2]);
    return std::abs(limitRad(p[3] - los));
}

/**
 * 作用：锁定当前相位做 v、ω 超前；转出可打锥面则沿自旋方向切到下一块。
 * 原理：不在飞时后重选「正对相机」的板，否则小陀螺时瞄准点会钉在车头。
 * 输入：飞时后的 n 块板；phase 为本帧匹配相位；vyaw 为飞时用的自旋（已乘 kw，rad/s）。
 * 输出：要打的相位下标。
 */
int pickLeadPlate(const std::vector<Eigen::Vector4d>& xyza, int phase, double vyaw) {
    const int n = static_cast<int>(xyza.size());
    if (n <= 0)
        return 0;
    int id = ((phase % n) + n) % n;
    constexpr double kMaxFace = 70.0 * kPi / 180.0;  // 板法线与视线夹角超过则难打
    if (facingErr(xyza[static_cast<std::size_t>(id)]) < kMaxFace)
        return id;
    // angle = yaw + i·2π/n；ω>0 时 yaw 增大，正前下标减小，下一块是 id-1。
    const int dir = (vyaw >= 0.0) ? -1 : 1;
    const int nxt = (id + dir + n) % n;
    if (facingErr(xyza[static_cast<std::size_t>(nxt)]) <
        facingErr(xyza[static_cast<std::size_t>(id)]))
        return nxt;
    return id;
}

/**
 * 作用：EKF 板心米 → 弹道用的相机系 mm。输入：xyza。输出：mm。
 */
cv::Point3d xyzaToMm(const Eigen::Vector4d& p) {
    return {p[0] * 1000.0, p[1] * 1000.0, p[2] * 1000.0};
}

const char* stateName(VehicleTrackState s) {
    switch (s) {
        case VehicleTrackState::Detecting:
            return "detecting";
        case VehicleTrackState::Tracking:
            return "tracking";
        case VehicleTrackState::TempLost:
            return "temp_lost";
        default:
            return "lost";
    }
}

}  // namespace

/**
 * 作用：YAML aim → 弹速/重力/迭代/v·ω 提前系数 + EKF 门槛。
 * 输入：path。输出：打开成功为 true。
 */
bool Aimer::loadConfig(const std::string& path) {
    cv::FileStorage fs;
    if (!openDetectorYaml(path, fs))
        return false;
    const cv::FileNode n = fs["aim"];
    readYaml(n, "bullet_speed_mps", params_.ballistic.bullet_speed_mps);
    readYaml(n, "gravity_mps2", params_.ballistic.gravity_mps2);
    readYaml(n, "iterations", params_.ballistic.iterations);
    readYaml(n, "w_lead_scale", params_.w_lead_scale);  // 只缩放飞时 ω，缺键保持 1
    readYaml(n, "v_lead_scale", params_.v_lead_scale);  // 只缩放飞时 vx/vy/vz，缺键保持 1
    tracker_.loadConfig(path);  // 同一节点里的半径/Q/状态机
    return true;
}

/**
 * 作用：EKF 跟踪最近车；用 kv·v 与 kw·ω 把当前板推到飞时，再弹道抬高。
 * 输入：装甲、dt。输出：瞄准结果。
 */
AimTarget Aimer::update(const std::vector<ArmorBoard>& armors, double dt) {
    tracker_.step(armors, dt);  // 最近车 → 11 维后验（含 vx,vy,vz,ω）
    AimTarget tgt;
    tgt.track_state = tracker_.state();
    tgt.vehicle_id = tracker_.trackedId();
    if (!tracker_.hasTarget())
        return tgt;

    const auto plates_now = tracker_.armorXyza();
    if (plates_now.empty())
        return tgt;
    tgt.hub_mm = tracker_.hubMm();
    tgt.hub_px = projectCamMm(tgt.hub_mm);
    tgt.r_m = tracker_.ekf().x[8];
    tgt.vx_mps = tracker_.ekf().x[1];
    tgt.vy_mps = tracker_.ekf().x[3];
    tgt.vz_mps = tracker_.ekf().x[5];
    tgt.vyaw_rad_s = tracker_.ekf().x[7];
    tgt.nis = tracker_.ekf().last_nis;
    tgt.phase_id = tracker_.lastPhaseId();
    const ArmorType plate_type = tracker_.trackedType();
    for (const auto& p : plates_now) {
        const cv::Point3d mm = xyzaToMm(p);
        tgt.plate_px.push_back(projectCamMm(mm));
        // 板心+yaw → 与 PnP 同尺寸四角，贴在车四周。
        tgt.plate_quads.push_back(ArmorBoard::projectArmorQuad(mm, p[3], plate_type));
    }

    int phase = tgt.phase_id;
    cv::Point3d shot = xyzaToMm(plates_now[static_cast<std::size_t>(
        std::clamp(phase, 0, std::max(0, static_cast<int>(plates_now.size()) - 1)))]);
    BallisticSolution ball = solveBallistic(shot, params_.ballistic);
    double t_fly = ball.ok ? ball.fly_time_s : 0.0;
    const int n_iter = std::max(1, params_.ballistic.iterations);
    for (int k = 0; k < n_iter && t_fly > 1e-4; ++k) {
        // 飞时内外推：车心 += kv·v·t，yaw += kw·ω·t（只改这份拷贝，EKF 不动）。
        Eigen::VectorXd x0 = tracker_.ekf().x;
        x0[1] *= params_.v_lead_scale;  // vxc
        x0[3] *= params_.v_lead_scale;  // vyc
        x0[5] *= params_.v_lead_scale;  // vzc
        x0[7] *= params_.w_lead_scale;  // vyaw
        const Eigen::VectorXd x = tracker_.extrapolate(x0, t_fly);
        const auto future = tracker_.armorXyza(&x);
        if (future.empty())
            break;
        phase = pickLeadPlate(future, tracker_.lastPhaseId(), x0[7]);
        shot = xyzaToMm(future[static_cast<std::size_t>(phase)]);
        ball = solveBallistic(shot, params_.ballistic);
        if (!ball.ok)
            break;
        t_fly = ball.fly_time_s;
    }
    if (!ball.ok)
        return tgt;

    tgt.ok = true;
    tgt.phase_id = phase;
    tgt.dist_m = cv::norm(shot) / 1000.0;
    tgt.fly_time_s = ball.fly_time_s;
    tgt.aim_cam_mm = ball.aim_cam_mm;
    tgt.aim_px = projectCamMm(ball.aim_cam_mm);
    tgt.ballistic = ball;
    return tgt;
}

/**
 * 作用：青圈车心、四周预测板色块、橙十字弹道点、车心速度文字；读 AimTarget，不另算 v/ω。
 * 板矩形读 plate_quads（与 PnP 同物点）；k 画在 v/ω 旁，与飞时系数同一字段。
 * 输入：BGR 图、update 结果、开关。输出：无。
 */
void Aimer::draw(cv::Mat& img, const AimTarget& tgt, const AimDrawFlags& flags) const {
    if (!tgt.ok && tgt.track_state == VehicleTrackState::Lost)
        return;
    if (flags.hub && tgt.hub_px.x >= 0.f)
        cv::circle(img, tgt.hub_px, 6, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
    if (flags.vel && tgt.hub_px.x >= 0.f) {
        // 线速度：旋转中心 3D 速率；xz 为水平面（相机 X 右 Z 前）。角速度读 EKF vyaw；kv/kw 为飞时系数。
        const double v_lin = std::hypot(tgt.vx_mps, tgt.vy_mps, tgt.vz_mps);
        const double v_xz = std::hypot(tgt.vx_mps, tgt.vz_mps);
        const double w_deg = tgt.vyaw_rad_s * 180.0 / kPi;
        const cv::Point c(cvRound(tgt.hub_px.x), cvRound(tgt.hub_px.y));
        const cv::Scalar cyan(255, 255, 0);
        cv::putText(img,
                    cv::format("v:%.2fm/s xz:%.2f*kv=%.2f", v_lin, v_xz, params_.v_lead_scale),
                    cv::Point(c.x + 10, c.y + 16), cv::FONT_HERSHEY_PLAIN, 0.95, cyan, 1, cv::LINE_AA);
        cv::putText(img,
                    cv::format("w:%.2frad/s (%.0fdeg/s)*kw=%.2f", tgt.vyaw_rad_s, w_deg,
                               params_.w_lead_scale),
                    cv::Point(c.x + 10, c.y + 30), cv::FONT_HERSHEY_PLAIN, 0.95, cyan, 1,
                    cv::LINE_AA);
    }
    if (flags.plates && !tgt.plate_quads.empty()) {
        // 相位 0..3：绿 / 橙 / 黄 / 品红，半透明盖在预测板面上，与检测金框区分。
        const cv::Scalar kFill[] = {
            cv::Scalar(0, 220, 0),
            cv::Scalar(0, 140, 255),
            cv::Scalar(0, 255, 255),
            cv::Scalar(255, 0, 255),
        };
        constexpr double kAlpha = 0.38;
        cv::Mat wash = img.clone();
        for (std::size_t i = 0; i < tgt.plate_quads.size(); ++i) {
            const auto& q = tgt.plate_quads[i];
            if (q.size() != 4)
                continue;
            std::vector<cv::Point> poly;
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
            const cv::Scalar& c = kFill[i % 4];
            cv::fillConvexPoly(wash, poly, c, cv::LINE_AA);
        }
        cv::addWeighted(wash, kAlpha, img, 1.0 - kAlpha, 0.0, img);
        for (std::size_t i = 0; i < tgt.plate_quads.size(); ++i) {
            const auto& q = tgt.plate_quads[i];
            if (q.size() != 4)
                continue;
            std::vector<cv::Point> poly;
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
            const cv::Scalar& c = kFill[i % 4];
            const bool aimed = tgt.ok && static_cast<int>(i) == tgt.phase_id;
            cv::polylines(img, poly, true, c, aimed ? 3 : 2, cv::LINE_AA);
        }
    } else if (flags.plates) {
        for (const auto& px : tgt.plate_px) {
            if (px.x >= 0.f)
                cv::circle(img, px, 7, cv::Scalar(0, 220, 0), 1, cv::LINE_AA);
        }
    }
    if (flags.aim && tgt.ok && tgt.aim_px.x >= 0.f) {
        const cv::Point c(cvRound(tgt.aim_px.x), cvRound(tgt.aim_px.y));
        const cv::Scalar orange(0, 140, 255);
        cv::drawMarker(img, c, orange, cv::MARKER_CROSS, 22, 2, cv::LINE_AA);
        cv::circle(img, c, 10, orange, 1, cv::LINE_AA);
        if (flags.hud) {
            const std::string hud =
                cv::format("aim id:%d %s t:%.0fms", tgt.vehicle_id, stateName(tgt.track_state),
                           tgt.fly_time_s * 1000.0);
            cv::putText(img, hud, cv::Point(c.x + 14, c.y - 8), cv::FONT_HERSHEY_PLAIN, 0.95, orange,
                        1, cv::LINE_AA);
        }
    } else if (flags.hud && tgt.hub_px.x >= 0.f) {
        const cv::Point c(cvRound(tgt.hub_px.x), cvRound(tgt.hub_px.y));
        const std::string hud = cv::format("ekf id:%d %s", tgt.vehicle_id, stateName(tgt.track_state));
        cv::putText(img, hud, cv::Point(c.x + 10, c.y - 8), cv::FONT_HERSHEY_PLAIN, 0.95,
                    cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
    }
}
