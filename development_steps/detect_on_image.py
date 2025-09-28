from ultralytics import YOLO
import cv2

# --- Configuration ---
# Load the pre-trained YOLOv8 model. 'yolov8n.pt' is the smallest and fastest version.
# The '.pt' file will be downloaded automatically the first time you run this.
model = YOLO("yolov8n.pt") 

# Define the image you want to analyze
IMAGE_FILE = "first_frame.jpg"

# ---------------------

# Run the prediction
# This line tells YOLO to find all the objects in our image.
results = model.predict(IMAGE_FILE)

# The 'results' object contains all the information about the detections.
# We are interested in the first result, since we only processed one image.
result = results[0]

# The ultralytics library has a handy '.plot()' method that automatically
# draws all the detected bounding boxes and labels on the image for us.
annotated_frame = result.plot()

# Display the annotated image in a window
cv2.imshow("YOLOv8 Detection", annotated_frame)

# Wait for a key press and then close the window
cv2.waitKey(0) 
cv2.destroyAllWindows()

# Optional: Save the annotated image to a file
cv2.imwrite("detection_result.jpg", annotated_frame)
print("Detection complete! Result saved as detection_result.jpg")