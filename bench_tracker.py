import cv2
import mediapipe as mp
import numpy as np
import argparse
import csv
import os
from datetime import datetime

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# ── helpers ───────────────────────────────────────────────────────────────────

def compute_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def ema(prev, new, alpha=0.2):
    if prev is None:
        return new
    return alpha * new + (1 - alpha) * prev


def elbow_flare_angle(shoulder, elbow, hip):
    """
    Angle between the upper-arm vector (shoulder→elbow) and the torso vector
    (shoulder→hip). Ideal bench tuck: ~45–75°. >85° = excessive flare.
    """
    upper_arm = np.array(elbow) - np.array(shoulder)
    torso_vec = np.array(hip) - np.array(shoulder)
    cosine = np.dot(upper_arm, torso_vec) / (
        np.linalg.norm(upper_arm) * np.linalg.norm(torso_vec) + 1e-9)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


# ── feedback ──────────────────────────────────────────────────────────────────

def bench_feedback(elbow_angle, flare_angle, wrist_elbow_offset_px):
    """
    elbow_angle           : shoulder-elbow-wrist angle (depth of press)
    flare_angle           : upper-arm vs torso angle (elbow tuck)
    wrist_elbow_offset_px : |wrist_x - elbow_x| in pixels (wrist stacking)
    """
    cues = []

    # press depth — elbow should reach ~90° at bottom
    if elbow_angle > 110:
        cues.append("Lower the bar further")
    elif elbow_angle < 60:
        cues.append("Don't go too deep")
    else:
        cues.append("Good press depth")

    # elbow flare — ideally 45–75° relative to torso
    if flare_angle > 85:
        cues.append("Tuck elbows in more")
    elif flare_angle < 30:
        cues.append("Elbows too tucked")
    else:
        cues.append("Good elbow angle")

    # wrist stacking — wrist should track over elbow
    if wrist_elbow_offset_px > 40:
        cues.append("Stack wrists over elbows")

    return cues


# ── rep counter ───────────────────────────────────────────────────────────────

class BenchRepCounter:
    """
    Rep = elbow bends below DOWN_THRESH then extends above UP_THRESH.
    """
    DOWN_THRESH = 100
    UP_THRESH   = 150

    def __init__(self):
        self.count = 0
        self.in_press = False

    def update(self, elbow_angle):
        if not self.in_press and elbow_angle < self.DOWN_THRESH:
            self.in_press = True
        elif self.in_press and elbow_angle > self.UP_THRESH:
            self.in_press = False
            self.count += 1
        return self.count


# ── overlay ───────────────────────────────────────────────────────────────────

def draw_text_box(frame, lines, origin, font_scale=0.65, thickness=2):
    x, y = origin
    line_h = int(font_scale * 30)
    pad = 8
    max_w = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
                for l in lines)
    box_h = line_h * len(lines) + pad * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - pad, y - line_h - pad),
                  (x + max_w + pad, y + box_h - pad), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    for i, line in enumerate(lines):
        color = (0, 255, 0) if "Good" in line else (0, 200, 255)
        cv2.putText(frame, line, (x, y + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


# ── main ──────────────────────────────────────────────────────────────────────

def run(source, save_csv=False, out_video=None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if out_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video, fourcc, fps, (w, h))

    log_rows = []
    rep_counter = BenchRepCounter()

    smooth_elbow = None
    smooth_flare = None
    frame_idx = 0

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if isinstance(source, int):
                frame = cv2.flip(frame, 1)

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                def pt(landmark):
                    l = lm[landmark.value]
                    return [l.x * w, l.y * h]

                l_sh  = pt(mp_pose.PoseLandmark.LEFT_SHOULDER)
                l_el  = pt(mp_pose.PoseLandmark.LEFT_ELBOW)
                l_wr  = pt(mp_pose.PoseLandmark.LEFT_WRIST)
                l_hip = pt(mp_pose.PoseLandmark.LEFT_HIP)

                r_sh  = pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)
                r_el  = pt(mp_pose.PoseLandmark.RIGHT_ELBOW)
                r_wr  = pt(mp_pose.PoseLandmark.RIGHT_WRIST)
                r_hip = pt(mp_pose.PoseLandmark.RIGHT_HIP)

                raw_elbow = (compute_angle(l_sh, l_el, l_wr) +
                             compute_angle(r_sh, r_el, r_wr)) / 2

                raw_flare = (elbow_flare_angle(l_sh, l_el, l_hip) +
                             elbow_flare_angle(r_sh, r_el, r_hip)) / 2

                avg_wr_x = (l_wr[0] + r_wr[0]) / 2
                avg_el_x = (l_el[0] + r_el[0]) / 2
                wrist_offset = abs(avg_wr_x - avg_el_x)

                smooth_elbow = ema(smooth_elbow, raw_elbow)
                smooth_flare = ema(smooth_flare, raw_flare)

                reps = rep_counter.update(smooth_elbow)
                cues = bench_feedback(smooth_elbow, smooth_flare, wrist_offset)

                mp_draw.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(200, 100, 255), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(100, 200, 255), thickness=2))

                for pt_pos, label in [
                    (l_el, f"elbow {int(smooth_elbow)}°"),
                    (l_sh, f"flare {int(smooth_flare)}°"),
                ]:
                    cv2.putText(frame, label,
                                (int(pt_pos[0]) + 5, int(pt_pos[1]) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                hud_lines = [f"Reps: {reps}",
                             f"Elbow: {int(smooth_elbow)}",
                             f"Flare: {int(smooth_flare)}"] + cues
                draw_text_box(frame, hud_lines, (20, 30))

                if save_csv:
                    log_rows.append({
                        "frame":        frame_idx,
                        "raw_elbow":    round(raw_elbow, 2),
                        "smooth_elbow": round(smooth_elbow, 2),
                        "raw_flare":    round(raw_flare, 2),
                        "smooth_flare": round(smooth_flare, 2),
                        "wrist_offset": round(wrist_offset, 2),
                        "reps":         reps,
                        "feedback":     "|".join(cues)
                    })

            if writer:
                writer.write(frame)

            cv2.imshow("Bench Press Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    if save_csv and log_rows:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"bench_log_{ts}.csv"
        with open(csv_path, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer_csv.writeheader()
            writer_csv.writerows(log_rows)
        print(f"[INFO] Saved angle log to {csv_path}")

    print(f"[DONE] Total reps counted: {rep_counter.count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time bench press form analyzer")
    parser.add_argument("--source", default="0",
                        help="0 for webcam, or path to video file")
    parser.add_argument("--save-csv", action="store_true",
                        help="Save per-frame angle log to CSV")
    parser.add_argument("--out-video", default=None,
                        help="Path to save annotated output video (e.g. out.mp4)")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    run(source, save_csv=args.save_csv, out_video=args.out_video)
