import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b #knee to hip
    bc = c - b #knee to ankle angle 

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            def get_point(name):
                lm = landmarks[name.value]
                return [lm.x * w, lm.y * h]

            left_hip = get_point(mp_pose.PoseLandmark.LEFT_HIP)
            left_knee = get_point(mp_pose.PoseLandmark.LEFT_KNEE)
            left_ankle = get_point(mp_pose.PoseLandmark.LEFT_ANKLE)

            right_hip = get_point(mp_pose.PoseLandmark.RIGHT_HIP)
            right_knee = get_point(mp_pose.PoseLandmark.RIGHT_KNEE)
            right_ankle = get_point(mp_pose.PoseLandmark.RIGHT_ANKLE)

            left_knee_angle = angle(left_hip, left_knee, left_ankle)
            right_knee_angle = angle(right_hip, right_knee, right_ankle)

            avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

            if avg_knee_angle < 100:
                feedback = "Good"
            else:
                feedback = "Go lower"

            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            cv2.putText(frame, f"Knee angle: {int(avg_knee_angle)}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2)

            cv2.putText(frame, feedback,
                        (30, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2)

        cv2.imshow("Squat Tracker", frame)


cap.release()
cv2.destroyAllWindows()