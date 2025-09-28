import cv2
from ultralytics import YOLO

# --- Configuration ---
# To use a webcam, set VIDEO_SOURCE = 0
# To use a video file, set VIDEO_SOURCE = "path/to/your/video.mp4"
# To use an IP camera, set VIDEO_SOURCE = "rtsp://..."
VIDEO_SOURCE = 0 

# --- Recommendation Engine Thresholds ---
LOW_THRESHOLD = 5
HIGH_THRESHOLD = 10

# Load the YOLOv8 model
MODEL = YOLO("yolov8n.pt")

# ---------------------

# 1. Open video source
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: Could not open video source at {VIDEO_SOURCE}")
    exit()

# 2. Main Loop
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        print("Stream ended or frame could not be read. Exiting...")
        break

    # Run YOLOv8 tracking on the frame
    results = MODEL.track(frame, persist=True)
    
    final_frame = frame.copy()

    # --- Analysis Logic ---
    current_frame_vehicle_ids = set()
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        
        for track_id, cls_id in zip(track_ids, class_ids):
            # COCO class IDs for vehicles: car=2, motorcycle=3, bus=5, truck=7
            if cls_id in [2, 3, 5, 7]:
                current_frame_vehicle_ids.add(track_id)

    vehicle_count = len(current_frame_vehicle_ids)
    
    # --- Recommendation Logic ---
    if vehicle_count <= LOW_THRESHOLD:
        status, recommendation, color = "Light Traffic", "Set Green Time: 20s", (0, 255, 0) # Green
    elif vehicle_count <= HIGH_THRESHOLD:
        status, recommendation, color = "Moderate Traffic", "Set Green Time: 45s", (0, 255, 255) # Yellow
    else:
        status, recommendation, color = "Heavy Traffic", "Set Green Time: 60s", (0, 0, 255) # Red
        
    # --- Drawing ---
    # Draw YOLO annotations first
    final_frame = results[0].plot(img=final_frame)

    # Draw our custom text overlays on top
    cv2.putText(final_frame, f"Total Vehicle Count: {vehicle_count}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(final_frame, f"Status: {status}", (50, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(final_frame, f"Recommendation: {recommendation}", (50, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Display the final frame in a window
    cv2.imshow("Real-Time Traffic Analysis", final_frame)

    # --- Quit Condition ---
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 3. Cleanup
cap.release()
cv2.destroyAllWindows()
print("Analysis stopped and windows closed.")