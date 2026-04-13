Fire Detector using YOLO

A simple computer vision project to detect fire using YOLO and Python.

Overview

This project detects fire from images or video using a YOLO-based model.
It can be used as a basic system for fire monitoring and early warning.

Project Structure
fire-detector/
│── datasets/               # Dataset for training (not uploaded)
│── runs/                   # Training results (ignored)
│── fire.mp4                # Sample input video
│── output_fire_detect.mp4  # Output result video
│── firetest.py             # Fire detection script
│── infer.py                # Inference script
│── trainfire.py            # Training script
⚙️ Requirements
Python 3.x
OpenCV
PyTorch
Ultralytics (YOLO)

Install dependencies:

pip install -r requirements.txt
Usage

Run detection:

python firetest.py

Run inference:

python infer.py

Train model:

python trainfire.py
Example
Input: fire.mp4
Output: output_fire_detect.mp4
 Notes
Dataset folder is not included in the repository
Model files (.pt) are excluded due to size limitations
runs/ folder is ignored
Future Improvements
Add real-time webcam detection
Improve model accuracy with larger dataset
Add alert/notification system
