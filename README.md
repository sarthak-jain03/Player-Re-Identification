# Player Re-Identification in Sports Footage

A real-time computer vision system to detect, track, and assign consistent IDs to football players using a single video feed.

---

## Description

This project implements a complete pipeline for **player re-identification** in a sports video using **YOLOv11** for object detection and **DeepSORT** for tracking. It simulates a real-world broadcast scenario where players move in and out of the frame, and the model ensures that each player retains a consistent identity.

Each player is annotated with:
- A unique player ID
- A fixed-size colored square anchored to the top-center of their bounding box

The final output is an annotated `.mp4` video with accurate real-time re-identification.

---

##  Getting Started

###  Environment Requirements

- OS: Windows 10, Windows 11, or Ubuntu 22.04
- Python: **3.8 or higher**
- RAM: Minimum **4GB** (8GB recommended)
- GPU (optional but recommended): NVIDIA CUDA-compatible GPU for faster YOLO inference

###  Dependencies

Install via `requirements.txt`:

```bash
pip install -r requirements.txt
```
Or manually install:
```bash

pip install opencv-python
pip install ultralytics
pip install deep-sort-realtime

```

Dependencies used:

- `opencv-python` – Video processing and drawing

- `ultralytics` – YOLOv11 model integration

- `deep-sort-realtime` – Player tracking and re-identification

- `numpy` – (comes preinstalled with OpenCV)


## Installing
- Clone the repository or copy the project folder.

- Place your video in the videos/ folder.

- Since the yolo model is very large and cannot be pushed to github, it will automatically download the model from my google drive link and it will directly be placed into the model/ directory.
https://drive.google.com/file/d/1vk_YNbDv270ue4AVDzJaTrAljRNscvJu/view?usp=sharing


## Executing program
Run the program using:
```bash
python main.py
```
This will:

- Detect and track players

- Assign consistent IDs

- Output annotated video to: `output/reid_output.mp4`

## Help
If the output video is blank:

- Ensure the model path in main.py is correct.

- Make sure the input video exists and is not corrupted.

## Author
- **Sarthak Jain**
- github: `https://github.com/sarthak-jain03`


## License
This project is licensed under the [MIT License](LICENSE.md) – see the LICENSE.md file for details.
