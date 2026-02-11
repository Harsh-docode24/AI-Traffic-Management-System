# 🚦 A Dual-Model Deep Learning Framework for Dynamic Traffic Control and Emergency Vehicle Prioritization

An AI-driven intelligent traffic optimization system that integrates dual deep learning models with deterministic signal control logic to improve congestion management and emergency response efficiency.

---

# 1. Project Overview

This project presents a dual-model deep learning framework designed to optimize urban traffic flow and prioritize emergency vehicles in real time.

The system combines:

• A vehicle detection model for dynamic traffic density estimation  
• A specialized ambulance detection model for emergency override control  
• A deterministic rule engine for adaptive signal timing  

The framework demonstrates how AI-based perception modules can integrate with structured decision logic to simulate intelligent traffic management systems.

---

# 2. Problem Statement

Traditional traffic signal systems operate on static timing cycles that do not adapt to real-time traffic density or emergency vehicle presence.

This system addresses:

- Dynamic congestion management
- Real-time traffic density estimation
- Emergency vehicle prioritization
- Deterministic and explainable signal decisions

---

# 3. System Architecture

The framework follows a modular pipeline:

1. Video Frame Acquisition  
2. Vehicle Detection (YOLOv8)  
3. Traffic Density Estimation  
4. Rule-Based Signal Timing Engine  
5. Ambulance Detection Module  
6. Emergency Override Logic  
7. Annotated Output Rendering  

Each module operates independently and contributes structured output to the final control logic.

---

# 4. Core Components

## 4.1 Vehicle Detection Module

- YOLOv8-based real-time object detection  
- Frame-by-frame vehicle counting  
- Bounding box visualization  
- Congestion level estimation  

Vehicle count acts as the primary congestion metric.

---

## 4.2 Traffic Density Classification

Vehicle counts are categorized into predefined thresholds:

- Low Density  
- Medium Density  
- High Density  

Signal timing recommendations are derived from these thresholds.

---

## 4.3 Dynamic Signal Timing Engine

Green Signal Duration is calculated as:

Base Time + Density Adjustment

This deterministic logic ensures:

- Efficient signal cycling during low traffic  
- Extended green duration during high congestion  

---

## 4.4 Ambulance Detection & Emergency Override

A custom-trained YOLOv8 model (`best.pt`) detects ambulances.

When an ambulance is detected:

- Immediate green signal activation  
- Override of current timing logic  
- Automatic restoration after clearance  

This ensures faster emergency response.

---

# 5. Decision Logic (Explainable & Deterministic)

Signal decisions are derived from:

- Real-time vehicle count  
- Predefined congestion thresholds  
- Emergency detection flag  

No probabilistic heuristics are used.

All signal decisions are rule-based, reproducible, and explainable.

---

# 6. Technology Stack

- Python 3.9+
- OpenCV (Video Processing & Visualization)
- PyTorch (Deep Learning Backend)
- Ultralytics YOLOv8 (Object Detection)
- Roboflow (Dataset Management)
- Google Colab (Model Training)

---

# 7. Project Structure

```
AI-Traffic-Management-System/
│
├── src/
│   └── realtime_api_ambulance.py
│
├── docs/
│   ├── dita/
│   │   ├── concept_system_overview.xml
│   │   ├── task_installation.xml
│   │   └── reference_signal_logic.xml
│   │
│   ├── AI_Documentation_Workflow.md
│   └── docs_style_guide.md
│
├── Research Paper/
│   └── manuscript.pdf
│
├── README.md
└── .gitignore
```


# 8. Installation

## 1. Clone Repository

git clone https://github.com/Harsh-docode24/AI-Traffic-Management-System.git

cd AI-Traffic-Management-System

---

## 2. Create Virtual Environment

python -m venv venv

Windows:
venv\Scripts\activate

Linux/Mac:
source venv/bin/activate

---

## 3. Install Dependencies

pip install -r requirements.txt

---

## 4. Download Trained Model

Download `best.pt` from the Releases section  
Place it in the project root directory.

---

## 5. Run Application

python src/realtime_api_ambulance.py

---

# 9. Research Contribution

This project demonstrates:

- Dual-model deep learning integration  
- Real-time AI-based traffic perception  
- Structured signal control logic  
- Emergency override simulation  
- Modular system architecture  

The full research manuscript is available in the `Research Paper` directory.

---

# 10. Known Limitations

- Simulation-based signal control (no hardware deployment)
- Single-camera perspective
- Pixel-space congestion estimation
- Performance dependent on detection accuracy

---

# 11. Future Improvements

- Multi-intersection coordination
- IoT-based real traffic signal integration
- Multi-camera fusion
- Reinforcement learning–based signal optimization
- Smart city analytics dashboard

---

# 12. Documentation Approach

This repository follows structured documentation principles:

- Modular separation of perception and decision logic
- Deterministic explanation of algorithmic flow
- Reproducible installation instructions
- DITA-inspired content organization

Structured XML documentation modules are included in `/docs/dita`.

---

# 13. Author

Harsh Sehrawat

---

# Why This Project Stands Out

- Dual deep learning model architecture  
- Custom-trained ambulance detection model  
- Deterministic signal control logic  
- Real-time congestion analysis  
- Modular and research-backed framework  
- Structured documentation integration  

