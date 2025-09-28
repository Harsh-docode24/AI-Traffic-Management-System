import cv2
from ultralytics import YOLO

# --- Configuration ---
VIDEO_SOURCE = "C:/Users/sehra/OneDrive/Documents/SERIOUS STUFF   (RAM)/TRAFFIC ANALYSER (LIGHTS)/venv/sample.mp4 "  # <--- Change this to your video's filename
OUTPUT_FILENAME = "output_video.mp4"

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# ---------------------

# 1. Open the video file for reading
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise IOError(f"Cannot open video file {VIDEO_SOURCE}")

# 2. Get video properties (width, height, frames per second)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 3. Set up the video writer to save the output
#    'mp4v' is a common codec for MP4 files.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, fps, (frame_width, frame_height))

# 4. The Main Loop: Process each frame
while cap.isOpened():
    # Read one frame from the video
    ret, frame = cap.read()

    if not ret:
        # If 'ret' is False, it means we've reached the end of the video.
        print("End of video reached.")
        break

    # Run YOLO detection on the current frame
    results = model.predict(frame)
    annotated_frame = results[0].plot()

    # Write the annotated frame to our output video file
    out.write(annotated_frame)
    
    # (Optional) Display the processing in real-time
    cv2.imshow("Processing Video", annotated_frame)
    
    # Allow the user to quit by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Processing stopped by user.")
        break

# 5. Cleanup: Release the video objects
print(f"Processing complete! Output saved as {OUTPUT_FILENAME}")
cap.release()
out.release()
cv2.destroyAllWindows()