import cv2 
import mediapipe as mp
import torch
import numpy as np
import time
import os
from datetime import datetime

# Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# YOLOv5 for object detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
allowed_objects = ['person']

# Create reports folder if not exists
if not os.path.exists("reports"):
    os.makedirs("reports")

# Webcam
cap = cv2.VideoCapture(0)

# Log file
log_file_path = os.path.join("reports", f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
log_file = open(log_file_path, "w")

cheat_count = 0
MAX_CHEATS = 3

with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_mesh = face_mesh.process(rgb)

        # YOLOv5 detection
        results = model(frame)
        labels = results.pandas().xyxy[0]['name'].tolist()
        cheating = "No"
        color = (0, 255, 0)

        for label in labels:
            if label not in allowed_objects:
                cheating = f"Yes ({label})"
                color = (0, 0, 255)
                cheat_count += 1
                break

        # Enhanced detection
        blink = "No"
        mouth = "Closed"
        head_pose = "Center"
        eyes = "Yes"

        if result_mesh.multi_face_landmarks:
            for landmarks in result_mesh.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(255,0,255), thickness=1)
                )

                # Blink
                left_eye_ratio = landmarks.landmark[159].y - landmarks.landmark[145].y
                blink = "Yes" if left_eye_ratio < 0.01 else "No"

                # Mouth
                mouth_ratio = landmarks.landmark[13].y - landmarks.landmark[14].y
                mouth = "Open" if mouth_ratio > 0.03 else "Closed"

                # Head pose
                nose = landmarks.landmark[1]
                if nose.x < 0.4:
                    head_pose = "Left"
                elif nose.x > 0.6:
                    head_pose = "Right"
                elif nose.y < 0.4:
                    head_pose = "Up"
                elif nose.y > 0.6:
                    head_pose = "Down"
                else:
                    head_pose = "Center"

        # Draw info on screen
        cv2.putText(frame, f"Head {head_pose}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Mouth {mouth}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Center", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(frame, f"{'Blink' if blink == 'Yes' else ''}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        cv2.putText(frame, f"{labels}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
        cv2.putText(frame, f"Cheating: {cheating}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Logging
        log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] Blink: {blink}, Eyes: {eyes}, Mouth: {mouth}, Head Pose: {head_pose}, Cheating: {cheating}\n"
        log_file.write(log_msg)
        print(log_msg.strip())

        # Show window
        cv2.imshow("Frame", frame)

        if cheat_count >= MAX_CHEATS:
            log_file.write("\nExam Terminated due to repeated cheating.\n")
            print("❌ Exam Terminated due to repeated cheating.")
            break

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

log_file.close()
cap.release()
cv2.destroyAllWindows()
