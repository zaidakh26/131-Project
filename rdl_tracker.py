import cv2
import mediapipe as mp
import numpy as np
import argparse
import csv
from datetime import datetime

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


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


def torso_angle_from_vertical(shoulder, hip):
    """
    Angle of the torso line (shoulder->hip) relative to vertical.
    0° = perfectly upright, 90° = horizontal.
    """
    dy = hip[1] - shoulder[1]   
    dx = hip[0] - shoulder[0]
    return np.degrees(np.arctan2(abs(dx), abs(dy)))




def rdl_feedback(hip_angle, torso_angle, knee_angle):
    """
    hip_angle    : shoulder-hip-knee angle (hinge depth)
    torso_angle  : degrees from vertical (back straightness proxy)
    knee_angle   : hip-knee-ankle (should stay slightly bent, not deeply bent)
    """
    cues = []

    if hip_angle > 130:
        cues.append("Hinge more at hips")
    elif hip_angle < 30:
        cues.append("Don't hinge too deep")
    else:
        cues.append("Good hip hinge depth")

    # back alignment — torso should be roughly parallel to floor at bottom
    # at top, torso is upright (~0°). We flag if back is rounded (hard to detect
    # from 2D side view, so we use torso angle as a proxy for forward lean)
    if torso_angle > 70:
        cues.append("Back nearly parallel — keep neutral spine")
    elif torso_angle < 10 and hip_angle < 80:
        cues.append("Hinge further forward")

    # knee bend — RDL = slight bend only, not a squat
    if knee_angle < 130:
        cues.append("Too much knee bend — keep legs straighter")
    else:
        cues.append("Good knee position")

    return cues




class RDLRepCounter:
    """
    Rep = hip hinge down (hip_angle drops below HINGE_THRESH)
    then back up (hip_angle rises above STAND_THRESH).
    """
    HINGE_THRESH = 100
    STAND_THRESH = 150

    def __init__(self):
        self.count = 0
        self.in_hinge = False

    def update(self, hip_angle):
        if not self.in_hinge and hip_angle < self.HINGE_THRESH:
            self.in_hinge = True
        elif self.in_hinge and hip_angle > self.STAND_THRESH:
            self.in_hinge = False
            self.count += 1
        return self.count



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
    rep_counter = RDLRepCounter()

    smooth_hip    = None
    smooth_knee   = None
    smooth_torso  = None

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
                l_hip = pt(mp_pose.PoseLandmark.LEFT_HIP)
                l_kn  = pt(mp_pose.PoseLandmark.LEFT_KNEE)
                l_an  = pt(mp_pose.PoseLandmark.LEFT_ANKLE)

                r_sh  = pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)
                r_hip = pt(mp_pose.PoseLandmark.RIGHT_HIP)
                r_kn  = pt(mp_pose.PoseLandmark.RIGHT_KNEE)
                r_an  = pt(mp_pose.PoseLandmark.RIGHT_ANKLE)

                raw_hip   = (compute_angle(l_sh, l_hip, l_kn) +
                             compute_angle(r_sh, r_hip, r_kn)) / 2

                raw_knee  = (compute_angle(l_hip, l_kn, l_an) +
                             compute_angle(r_hip, r_kn, r_an)) / 2

                # use average of left/right torso angle
                raw_torso = (torso_angle_from_vertical(l_sh, l_hip) +
                             torso_angle_from_vertical(r_sh, r_hip)) / 2

                smooth_hip   = ema(smooth_hip,   raw_hip)
                smooth_knee  = ema(smooth_knee,  raw_knee)
                smooth_torso = ema(smooth_torso, raw_torso)

                reps = rep_counter.update(smooth_hip)
                cues = rdl_feedback(smooth_hip, smooth_torso, smooth_knee)

                mp_draw.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(66, 245, 200), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(66, 117, 245), thickness=2))

                # joint angle annotations
                for pt_pos, label in [
                    (l_hip, f"hip {int(smooth_hip)}°"),
                    (l_kn,  f"knee {int(smooth_knee)}°"),
                    (l_sh,  f"torso {int(smooth_torso)}°"),
                ]:
                    cv2.putText(frame, label,
                                (int(pt_pos[0]) + 5, int(pt_pos[1]) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                hud_lines = [f"Reps: {reps}",
                             f"Hip:   {int(smooth_hip)}",
                             f"Knee:  {int(smooth_knee)}",
                             f"Torso: {int(smooth_torso)}"] + cues
                draw_text_box(frame, hud_lines, (20, 30))

                if save_csv:
                    log_rows.append({
                        "frame": frame_idx,
                        "raw_hip": round(raw_hip, 2),
                        "smooth_hip": round(smooth_hip, 2),
                        "raw_knee": round(raw_knee, 2),
                        "smooth_knee": round(smooth_knee, 2),
                        "torso_angle": round(smooth_torso, 2),
                        "reps": reps,
                        "feedback": "|".join(cues)
                    })

            if writer:
                writer.write(frame)

            cv2.imshow("RDL Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    if save_csv and log_rows:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"rdl_log_{ts}.csv"
        with open(csv_path, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer_csv.writeheader()
            writer_csv.writerows(log_rows)
        print(f"[INFO] Saved angle log to {csv_path}")

    print(f"[DONE] Total reps counted: {rep_counter.count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time RDL form analyzer")
    parser.add_argument("--source", default="0",
                        help="0 for webcam, or path to video file")
    parser.add_argument("--save-csv", action="store_true",
                        help="Save per-frame angle log to CSV")
    parser.add_argument("--out-video", default=None,
                        help="Path to save annotated output video (e.g. out.mp4)")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    run(source, save_csv=args.save_csv, out_video=args.out_video)
