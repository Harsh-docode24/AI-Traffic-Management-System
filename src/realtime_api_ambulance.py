import cv2
import time
from ultralytics import YOLO

# --- Configuration ---
VIDEO_SOURCE = 0 
AMBULANCE_MODEL_PATH = "best.pt" 
GENERAL_MODEL = YOLO("yolov8n.pt")
CONFIDENCE_THRESHOLD = 0.5
LOW_THRESHOLD = 5
HIGH_THRESHOLD = 10
AMBULANCE_PRIORITY_SECONDS = 20

# ---------------------

# 1. Load Models
try:
    ambulance_model = YOLO(AMBULANCE_MODEL_PATH)
except Exception as e:
    print(f"Error loading ambulance model: {e}")
    print("Make sure 'best.pt' is in the same folder as the script.")
    exit()

# 2. Open Video Source
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

ambulance_detected_time = None

# 3. Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    final_frame = frame.copy()
    ambulance_present = False

    # --- Ambulance Detection ---
    ambulance_results = ambulance_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    for result in ambulance_results:
        if len(result.boxes) > 0:
            ambulance_present = True
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # --- THIS IS THE FIX ---
                # Use ambulance_model.names to get the class name
                cls_name = ambulance_model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                
                cv2.rectangle(final_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(final_frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- Ambulance Priority Override ---
    if ambulance_present:
        if ambulance_detected_time is None:
            ambulance_detected_time = time.time()

        elapsed_time = time.time() - ambulance_detected_time
        
        if elapsed_time < AMBULANCE_PRIORITY_SECONDS:
            status = "AMBULANCE DETECTED"
            recommendation = "IMMEDIATE GREEN LIGHT"
            color = (0, 0, 255)
            timer_text = f"Priority Timer: {int(AMBULANCE_PRIORITY_SECONDS - elapsed_time)}s"
            cv2.putText(final_frame, timer_text, (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            ambulance_detected_time = None
    
    # --- Normal Traffic Logic ---
    if not ambulance_present or ambulance_detected_time is None:
        ambulance_detected_time = None
        general_results = GENERAL_MODEL.track(frame, persist=True, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        vehicle_count = 0
        if general_results[0].boxes is not None and general_results[0].boxes.id is not None:
            class_ids = general_results[0].boxes.cls.int().cpu().tolist()
            vehicle_count = sum(1 for cid in class_ids if cid in [2, 3, 5, 7])

        if vehicle_count <= LOW_THRESHOLD:
            status, recommendation, color = "Light Traffic", "Green Time: 20s", (0, 255, 0)
        elif vehicle_count <= HIGH_THRESHOLD:
            status, recommendation, color = "Moderate Traffic", "Green Time: 45s", (0, 255, 255)
        else:
            status, recommendation, color = "Heavy Traffic", "Green Time: 60s", (0, 0, 255)
        
        final_frame = general_results[0].plot(img=final_frame)
        cv2.putText(final_frame, f"Total Vehicle Count: {vehicle_count}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # --- Display Information ---
    cv2.putText(final_frame, f"Status: {status}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(final_frame, f"Recommendation: {recommendation}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Final Traffic Management System", final_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4. Cleanup
cap.release()
cv2.destroyAllWindows()
