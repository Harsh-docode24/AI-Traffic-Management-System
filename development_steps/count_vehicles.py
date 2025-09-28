import cv2
from ultralytics import YOLO
import numpy as np

# --- Configuration ---
VIDEO_SOURCE ="C:/Users/sehra/OneDrive/Documents/SERIOUS STUFF   (RAM)/TRAFFIC ANALYSER (LIGHTS)/venv/sample.mp4 "   # <--- Change this to your video's filename
OUTPUT_FILENAME = "output_counted_video.mp4"


# IMPORTANT: You must adjust this line's Y-coordinate for your specific video.
# To find a good Y-value, open your video and guess a pixel height for your line.
LINE_Y_COORDINATE = 360 

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# A dictionary to store the last known position of each tracked object
track_history = {}

# A set to store the IDs of vehicles that have already been counted
counted_ids = set()

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

vehicle_counter = 0

# 3. The Main Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # *** THE KEY CHANGE: Use model.track() instead of model.predict() ***
    results = model.track(frame, persist=True)
    
    # Get the bounding boxes and track IDs
    try:
        boxes = results[0].boxes.xywh.cpu() # Bounding boxes in (x, y, width, height) format
        track_ids = results[0].boxes.id.int().cpu().tolist() # Vehicle IDs
    except AttributeError:
        # If no objects are tracked, boxes and track_ids will be None
        track_ids = []

    # Draw the counting line on the frame
    cv2.line(frame, (0, LINE_Y_COORDINATE), (frame_width, LINE_Y_COORDINATE), (0, 255, 0), 2)

    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        center_y = int(y) # The Y-coordinate of the center of the box

        # Check if the vehicle is new to the tracker
        if track_id not in track_history:
            track_history[track_id] = center_y
            continue

        # Get the previous position and update the history
        prev_y = track_history[track_id]
        track_history[track_id] = center_y

        # THE COUNTING LOGIC: Check for a crossing
        # If a vehicle was above the line and is now below it, count it!
        if prev_y < LINE_Y_COORDINATE and center_y >= LINE_Y_COORDINATE and track_id not in counted_ids:
            vehicle_counter += 1
            counted_ids.add(track_id)
            # Draw a circle on the vehicle as it's counted
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    # Display the running count on the frame
    cv2.putText(frame, f"Vehicles Counted: {vehicle_counter}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Get the annotated frame with bounding boxes
    annotated_frame = results[0].plot()

    # Combine our custom drawings (line, text) with the YOLO annotations
    # We create a mask for our drawings and apply it to the annotated frame
    final_frame = cv2.addWeighted(annotated_frame, 0.7, frame, 0.3, 0)
    final_frame[:100, :] = annotated_frame[:100, :] # Keep top part for YOLO's own labels
    final_frame = cv2.line(final_frame, (0, LINE_Y_COORDINATE), (frame_width, LINE_Y_COORDINATE), (0, 255, 0), 2)
    final_frame = cv2.putText(final_frame, f"Vehicles Counted: {vehicle_counter}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    out.write(final_frame)
    cv2.imshow("Vehicle Counting", final_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4. Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Final vehicle count: {vehicle_counter}")