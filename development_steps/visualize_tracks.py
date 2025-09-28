import cv2
from ultralytics import YOLO
from collections import defaultdict

# --- Configuration ---
VIDEO_SOURCE = "C:/Users/sehra/OneDrive/Documents/SERIOUS STUFF   (RAM)/TRAFFIC ANALYSER (LIGHTS)/venv/sample.mp4 "   # <--- Change this to your video's filename
OUTPUT_FILENAME = "output_tracked_video.mp4"

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# A dictionary to store the tracking history (the path) of each object
# The key will be the track ID, and the value will be a list of center points
track_history = defaultdict(lambda: [])

# ---------------------

# 1. Open the video file
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise IOError(f"Cannot open video file {VIDEO_SOURCE}")

# 2. Set up video writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, fps, (frame_width, frame_height))

# 3. The Main Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 tracking on the frame
    results = model.track(frame, persist=True)
    
    # Get the bounding boxes and track IDs
    try:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
    except AttributeError:
        track_ids = []

    # Get the annotated frame from YOLO which has the boxes drawn on it
    annotated_frame = results[0].plot()

    # Plot the tracks
    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        center_x, center_y = int(x), int(y)
        
        # Append the new center point to this object's tracking history
        track = track_history[track_id]
        track.append((center_x, center_y))
        
        # Keep the track history to a certain length to avoid long trails
        if len(track) > 30:  # You can adjust this number
            track.pop(0)

        # Draw the tracking lines (the trail)
        if len(track) > 1:
            for i in range(1, len(track)):
                # Draw a line between the last two points
                cv2.line(annotated_frame, track[i - 1], track[i], (0, 255, 0), 2)
        
        # Display the Track ID next to the bounding box
        id_text = f"ID: {track_id}"
        cv2.putText(annotated_frame, id_text, (center_x, center_y - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    out.write(annotated_frame)
    cv2.imshow("Vehicle Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4. Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processing complete! Output saved as {OUTPUT_FILENAME}")