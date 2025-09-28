import cv2
import numpy as np
from ultralytics import YOLO

# --- This print statement confirms you are running the new version ---
print("--- Running script with Recommendation Engine ---")

# --- Configuration ---
VIDEO_SOURCE =  "C:/Users/sehra/OneDrive/Documents/SERIOUS STUFF   (RAM)/TRAFFIC ANALYSER (LIGHTS)/venv/sample.mp4"   

OUTPUT_FILENAME = "output_recommendation.mp4"

# --- Recommendation Engine Thresholds ---
# You can tune these values based on your video
LOW_THRESHOLD = 5
HIGH_THRESHOLD = 10

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# ---------------------

# 1. Open video and get dimensions
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise IOError(f"Cannot open video file {VIDEO_SOURCE}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 2. Define the full-screen zone and video writer
ZONE_POLYGON = np.array([[0, 0], [frame_width, 0], [frame_width, frame_height], [0, frame_height]], np.int32)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, fps, (frame_width, frame_height))

# 3. Main Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)
    
    final_frame = frame.copy() 
    
    current_frame_vehicle_ids = set()
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        
        for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
            # COCO class IDs for vehicles: car=2, motorcycle=3, bus=5, truck=7
            if cls_id in [2, 3, 5, 7]:
                current_frame_vehicle_ids.add(track_id)

    vehicle_count = len(current_frame_vehicle_ids)
    
    # --- RECOMMENDATION LOGIC ---
    status = "Unknown"
    recommendation = "No recommendation"
    color = (255, 255, 255) # White

    if vehicle_count <= LOW_THRESHOLD:
        status = "Light Traffic"
        recommendation = "Set Green Time: 20s"
        color = (0, 255, 0) # Green
    elif vehicle_count <= HIGH_THRESHOLD:
        status = "Moderate Traffic"
        recommendation = "Set Green Time: 45s"
        color = (0, 255, 255) # Yellow
    else:
        status = "Heavy Traffic"
        recommendation = "Set Green Time: 60s"
        color = (0, 0, 255) # Red
        
    # Draw the YOLO annotations first
    final_frame = results[0].plot(img=final_frame)

    # Draw our custom text overlays on top
    cv2.putText(final_frame, f"Total Vehicle Count: {vehicle_count}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(final_frame, f"Status: {status}", (50, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(final_frame, f"Recommendation: {recommendation}", (50, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    out.write(final_frame)
    cv2.imshow("Traffic Recommendation Engine", final_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4. Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()