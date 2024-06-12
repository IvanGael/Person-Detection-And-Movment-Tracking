# pip install opencv-python mediapipe ultralytics

import cv2
import mediapipe as mp
from ultralytics import YOLO

# Load YOLOv8 model from Ultralytics hub
model = YOLO('yolov8n')  # 'yolov8n' stands for YOLOv8 Nano, you can choose other versions like 'yolov8s', 'yolov8m', etc.

# Initialize MediaPipe Face Detection and Landmark Detection
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Variables to track movement
prev_x, prev_y = None, None
threshold = 20  # Movement threshold

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect objects in the frame using YOLOv8
    results = model(rgb_frame)

    # Iterate over the detected results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Extract bounding box coordinates
            cls = box.cls[0]  # Extract class

            # Check if the detected object is a person (class id 0 for YOLO)
            if int(cls) == 0:
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (204,0,102), 2)

                # Calculate the center of the bounding box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Draw the center point
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                # Check for movement
                if prev_x is not None and prev_y is not None:
                    dx = abs(center_x - prev_x)
                    dy = abs(center_y - prev_y)
                    if dx > threshold or dy > threshold:
                        cv2.putText(frame, f'Mvt ({dx},{dy})', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                # Update previous coordinates
                prev_x, prev_y = center_x, center_y

                # Detect landmarks on the face within the bounding box
                bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                face_rgb = rgb_frame[int(y1):int(y2), int(x1):int(x2)]
                face_frame = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)

                # Convert the face frame to RGB for landmark detection
                face_results = face_detection.process(face_frame)

                if face_results.detections:
                    for detection in face_results.detections:
                        keypoints = detection.location_data.relative_keypoints
                        for keypoint in keypoints:
                            x = int(keypoint.x * bbox[2] + bbox[0])
                            y = int(keypoint.y * bbox[3] + bbox[1])
                            cv2.circle(frame, (x, y), 2, (231, 255, 151, 1.0), -1)

    # Display the frame
    cv2.imshow('Movement Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

