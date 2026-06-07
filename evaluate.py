"""
evaluate.py — offline evaluation of squat form detection

Usage
-----
    # Evaluate against Kaggle squat pose dataset:
    python evaluate.py --dataset /path/to/squat_features_augmented.csv

    # Evaluate against labeled video clips (original mode):
    python evaluate.py --labels labels.csv

Output: per-class precision, recall, F1, plus a summary CSV.
"""

import numpy as np
import argparse
import csv
import os
from collections import defaultdict

# ── Kaggle dataset label mapping ──────────────────────────────────────────────
# Labels 4 (heels off) and 5 (asymmetric) are skipped — not detected by our system

KAGGLE_LABEL_MAP = {
    0: "good_depth",
    1: "go_deeper",
    2: "torso_lean",
    3: "knee_cave",
}

# ── prediction logic (mirrors squat_tracker.py thresholds) ───────────────────

def predict_squat(knee_angle, torso_lean, knee_lateral):
    if torso_lean > 108:
        return "torso_lean"
    elif knee_lateral > 0.10:
        return "knee_cave"
    elif knee_angle > 60:
        return "go_deeper"
    else:
        return "good_depth"

# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(true_labels, pred_labels):
    classes = sorted(set(true_labels) | set(pred_labels))
    results = {}
    for cls in classes:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        results[cls] = {"precision": precision, "recall": recall, "f1": f1,
                        "tp": tp, "fp": fp, "fn": fn}
    overall_acc = sum(1 for t, p in zip(true_labels, pred_labels) if t == p) / len(true_labels)
    return results, overall_acc

def print_metrics(metrics, overall_acc, true_labels):
    print("\n── Per-class metrics ─────────────────────────────────────────")
    print(f"{'Class':<25} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print("-" * 65)
    for cls, m in sorted(metrics.items()):
        support = sum(1 for t in true_labels if t == cls)
        print(f"{cls:<25} {m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>8.3f} {support:>8}")
    print("-" * 65)
    print(f"{'Overall accuracy':<25} {overall_acc:>10.3f}")

# ── dataset mode (Kaggle CSV) ─────────────────────────────────────────────────

def evaluate_dataset(csv_path, out_csv):
    true_labels = []
    pred_labels = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_int = int(row["label"])
            if label_int not in KAGGLE_LABEL_MAP:
                continue  # skip heels_off (4) and asymmetric (5)

            true_lbl   = KAGGLE_LABEL_MAP[label_int]
            knee_angle  = (float(row["left_knee_angle"]) + float(row["right_knee_angle"])) / 2
            torso_lean  = float(row["torso_lean"])
            knee_lat    = (float(row["left_knee_lateral"]) + float(row["right_knee_lateral"])) / 2

            pred = predict_squat(knee_angle, torso_lean, knee_lat)
            true_labels.append(true_lbl)
            pred_labels.append(pred)

    if not true_labels:
        print("[ERROR] No rows could be evaluated.")
        return

    print(f"  Evaluated {len(true_labels)} frames from Kaggle dataset")
    metrics, overall_acc = compute_metrics(true_labels, pred_labels)
    print_metrics(metrics, overall_acc, true_labels)

    with open(out_csv, "w", newline="") as f:
        f.write("Class,Precision,Recall,F1,Support\n")
        for cls, m in sorted(metrics.items()):
            support = sum(1 for t in true_labels if t == cls)
            f.write(f"{cls},{m['precision']:.3f},{m['recall']:.3f},{m['f1']:.3f},{support}\n")
        f.write(f"\nOverall accuracy,{overall_acc:.3f}\n")

    print(f"\n[INFO] Results saved to {out_csv}")

# ── video clip mode (original) ────────────────────────────────────────────────

def evaluate_clips(labels_path, out_csv):
    import cv2
    import mediapipe as mp

    mp_pose = mp.solutions.pose

    def compute_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def ema(prev, new, alpha=0.2):
        return new if prev is None else alpha * new + (1 - alpha) * prev

    def elbow_flare_angle(shoulder, elbow, hip):
        upper_arm = np.array(elbow) - np.array(shoulder)
        torso_vec = np.array(hip)   - np.array(shoulder)
        cosine = np.dot(upper_arm, torso_vec) / (
            np.linalg.norm(upper_arm) * np.linalg.norm(torso_vec) + 1e-9)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def analyze_bench_clip(video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARN] Cannot open {video_path}")
            return None

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        elbow_vals, flare_vals = [], []
        n_frames = 0

        with mp_pose.Pose(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                if not results.pose_landmarks:
                    continue
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

                # Only analyze frames where the person is lying down —
                # instructional videos mix standing/talking with bench reps,
                # which pollutes angle measurements. When lying, shoulder y ≈ hip y.
                avg_sh_y  = (l_sh[1] + r_sh[1]) / 2
                avg_hip_y = (l_hip[1] + r_hip[1]) / 2
                if abs(avg_sh_y - avg_hip_y) > 0.25 * h:
                    continue

                raw_elbow = (compute_angle(l_sh, l_el, l_wr) +
                             compute_angle(r_sh, r_el, r_wr)) / 2
                raw_flare = (elbow_flare_angle(l_sh, l_el, l_hip) +
                             elbow_flare_angle(r_sh, r_el, r_hip)) / 2

                elbow_vals.append(raw_elbow)
                flare_vals.append(raw_flare)
                n_frames += 1

        cap.release()

        if not elbow_vals:
            return None

        min_elbow = min(elbow_vals)
        avg_flare = np.mean(flare_vals)

        # Thresholds mirror bench_tracker.py's bench_feedback() —
        # not tuned to test clips.
        if avg_flare > 85:
            return "elbow_flare"
        elif min_elbow > 110:
            return "too_shallow"
        else:
            return "good"

    def analyze_clip(video_path, exercise):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARN] Cannot open {video_path}")
            return None

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        smooth_knee = smooth_hip = None
        issue_votes = defaultdict(int)
        knee_vals, hip_vals = [], []
        n_frames = 0

        with mp_pose.Pose(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                if not results.pose_landmarks:
                    continue

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

                raw_knee = (compute_angle(l_hip, l_kn, l_an) +
                            compute_angle(r_hip, r_kn, r_an)) / 2
                raw_hip  = (compute_angle(l_sh, l_hip, l_kn) +
                            compute_angle(r_sh, r_hip, r_kn)) / 2
                avg_kn_x = (l_kn[0] + r_kn[0]) / 2
                avg_an_x = (l_an[0] + r_an[0]) / 2

                smooth_knee = ema(smooth_knee, raw_knee)
                smooth_hip  = ema(smooth_hip,  raw_hip)

                if exercise == "squat":
                    knee_vals.append(smooth_knee)
                    hip_vals.append(smooth_hip)
                    issue_votes["knee_cave"] += (1 if abs(avg_kn_x - avg_an_x) > 30 else 0)
                elif exercise == "rdl":
                    knee_vals.append(smooth_knee)
                    hip_vals.append(smooth_hip)

                n_frames += 1

        cap.release()

        if n_frames == 0:
            return None

        if exercise == "squat" and knee_vals:
            # Use minimum knee angle (deepest point) rather than majority vote —
            # standing frames dominate otherwise and always vote "go_deeper".
            min_knee = min(knee_vals)
            knee_cave_votes = issue_votes["knee_cave"]
            # Only check torso lean on near-standing frames (knee > 140°) to avoid
            # false positives at the bottom of a deep squat where torso naturally leans.
            standing_hips = [h for k, h in zip(knee_vals, hip_vals) if k > 140]
            torso_leaning = len(standing_hips) > 0 and min(standing_hips) < 50
            if knee_cave_votes > len(knee_vals) * 0.3:
                return "knee_cave"
            elif torso_leaning:
                return "torso_lean"
            elif min_knee > 100:
                return "go_deeper"
            else:
                return "good_depth"

        if exercise == "rdl" and knee_vals:
            # Use deepest hinge point rather than majority vote —
            # most frames are at the upright standing position which
            # would always vote "go_deeper" with per-frame logic.
            min_hip  = min(hip_vals)
            min_knee = min(knee_vals)
            if min_knee < 110:
                return "too_much_knee_bend"
            elif min_hip > 100:
                return "go_deeper"
            else:
                return "good"

        if exercise == "bench":
            return analyze_bench_clip(video_path)

        return max(issue_votes, key=issue_votes.get)

    rows = []
    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    true_labels = []
    pred_labels = []
    result_rows = []

    for row in rows:
        video    = row["video_path"].strip()
        exer     = row["exercise"].strip().lower()
        true_lbl = row["issue"].strip().lower()

        print(f"  Analyzing {os.path.basename(video)} ({exer}) ...", end=" ", flush=True)
        pred = analyze_clip(video, exer)
        if pred is None:
            print("SKIP (no landmarks)")
            continue

        correct = "✓" if pred == true_lbl else "✗"
        print(f"pred={pred} {correct}")

        true_labels.append(true_lbl)
        pred_labels.append(pred)
        result_rows.append({"video": video, "exercise": exer,
                            "true": true_lbl, "pred": pred,
                            "correct": pred == true_lbl})

    if not true_labels:
        print("[ERROR] No clips could be analyzed.")
        return

    metrics, overall_acc = compute_metrics(true_labels, pred_labels)
    print_metrics(metrics, overall_acc, true_labels)

    with open(out_csv, "w", newline="") as f:
        fieldnames = ["video", "exercise", "true", "pred", "correct"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)
        f.write("\n\nClass,Precision,Recall,F1\n")
        for cls, m in sorted(metrics.items()):
            f.write(f"{cls},{m['precision']:.3f},{m['recall']:.3f},{m['f1']:.3f}\n")
        f.write(f"\nOverall accuracy,{overall_acc:.3f}\n")

    print(f"\n[INFO] Results saved to {out_csv}")

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate squat form detection precision/recall")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="Path to squat_features_augmented.csv (Kaggle dataset)")
    group.add_argument("--labels",  help="CSV with columns: video_path, exercise, issue")
    parser.add_argument("--out-csv", default="eval_results.csv",
                        help="Where to save metrics (default: eval_results.csv)")
    args = parser.parse_args()

    if args.dataset:
        evaluate_dataset(args.dataset, args.out_csv)
    else:
        evaluate_clips(args.labels, args.out_csv)

if __name__ == "__main__":
    main()
