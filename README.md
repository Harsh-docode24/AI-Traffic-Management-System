# AI-Powered Traffic Management System

This project is an intelligent traffic management system that uses real-time object detection to analyze traffic flow, recommend optimal signal timings, and prioritize emergency vehicles.

![GIF of the project in action] ## Features
- **Real-Time Vehicle Counting**: Uses YOLOv8 to detect and track vehicles, providing an accurate, real-time count of traffic density.
- **Dynamic Signal Recommendation**: A rule-based engine analyzes the vehicle count and suggests optimal green light durations to reduce congestion.
- **Ambulance Priority Override**: Utilizes a custom-trained YOLOv8 model to specifically detect ambulances and trigger an immediate green light, clearing a path for emergency response.

## Technology Stack
- **Python**
- **OpenCV**
- **PyTorch**
- **Ultralytics (YOLOv8)**
- **Roboflow** (for dataset management)
- **Google Colab** (for model training)

## Setup and Usage
1. Clone the repository: `git clone https://github.com/Harsh-docode24/AI-Traffic-Management-System.git`
2. Navigate to the project directory: `cd AI-Traffic-Management-System`
3. Create and activate a virtual environment.
4. Install dependencies: `pip install -r requirements.txt`
5. **Download the custom model (`best.pt`)** from the [v1.0 Release page](https://github.com/Harsh-docode24/AI-Traffic-Management-System/releases/tag/v1.0.0). Place it in the main project folder.
6. Run the application: `python final_system.py`
