import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# --- Configuration ---
VIDEO_SOURCE = "C:/Users/sehra/OneDrive/Documents/SERIOUS STUFF   (RAM)/TRAFFIC ANALYSER (LIGHTS)/venv/sample.mp4"   
OUTPUT_FILENAME = "output_queue_video.mp4"

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# ---------------------

# 1. Open video and set up writer
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise IOError(f"Cannot open video file {VIDEO_SOURCE}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, fps, (frame_width, frame_height))

# Define the polygon covering the whole frame
QUEUE_ZONE_POLYGON = np.array([
    [0, 0],                          # Top-left
    [frame_width, 0],                # Top-right
    [frame_width, frame_height],     # Bottom-right
    [0, frame_height]                # Bottom-left
], np.int32)

# Vehicle classes in COCO dataset
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# 2. Main Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)
    
    final_frame = frame.copy() # Start with a clean frame
    
    # Draw the queue zone overlay
    overlay = frame.copy()
    cv2.fillPoly(overlay, [QUEUE_ZONE_POLYGON], (0, 255, 0))  # Green overlay
    final_frame = cv2.addWeighted(overlay, 0.3, final_frame, 0.7, 0)

    current_frame_queue_ids = set()
    
    # Make sure we have tracking results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # class IDs for filtering
        
        for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
            x, y, w, h = box
            center_point = (int(x), int(y))

            # Check if detection is inside polygon
            is_inside = cv2.pointPolygonTest(QUEUE_ZONE_POLYGON, center_point, False)

            # ✅ Only count vehicles
            if is_inside > 0 and cls_id in VEHICLE_CLASSES:
                current_frame_queue_ids.add(track_id)

                # Debug print (optional, remove later)
                print(f"Counted: Track {track_id}, Class ID {cls_id}")

    # Display queue length
    queue_length = len(current_frame_queue_ids)
    text = f"Current Queue Length: {queue_length}"
    cv2.putText(final_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # black text
    
    # Add the YOLO annotations to our frame
    final_frame = results[0].plot(img=final_frame)

    out.write(final_frame)
    cv2.imshow("Queue Measurement", final_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 3. Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
