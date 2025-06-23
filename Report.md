# Player Re-Identification – Brief Report

## Project Objective

To design and implement a system that detects, tracks, and re-identifies football players in real-time using a single video feed, ensuring consistent identification even when players move across frames or leave and re-enter the scene.

---

## Approach & Methodology

### 1. **Detection**
- We used the **YOLOv11** model (You Only Look Once) for real-time object detection.
- YOLO was chosen for its speed and accuracy in detecting players (class `0`: person).

### 2. **Tracking & Re-Identification**
- Integrated **DeepSORT** (Simple Online and Realtime Tracking with a Deep Association Metric) for:
  - Multi-object tracking
  - Consistent identity assignment
- It uses a Kalman Filter and appearance features to associate detections across frames.

### 3. **Annotation & Visualization**
- Players are annotated with:
  - Unique ID displayed above the player
  - Fixed-size square box anchored to the player’s bounding box

---

##  Techniques Tried & Their Outcomes

| Technique                  | Purpose                          | Outcome                                                  |
|---------------------------|----------------------------------|----------------------------------------------------------|
| YOLOv11                   | Person detection                 | Accurate player detection at real-time speeds            |
| DeepSORT                  | Tracking and Re-identification   | Maintained consistent IDs with minimal ID switching      |
| Manual Cropping + Embedding | (Initially tried) Player cropping for re-ID | Too slow and unnecessary due to DeepSORT's built-in features |




##  Challenges Faced

-  **Large model file not pushable to GitHub**  
  → Solution: Hosted on Google Drive and auto-downloaded using `gdown`.

-  **Inconsistent ID assignment** (when players left and re-entered)  
  → Solution: Tuned DeepSORT parameters (`max_age`, `n_init`) for stability.

-  **Player box not aligned correctly**  
  → Fixed by anchoring the ID square to the top-center of the bounding box.

-  **Box scaling issue (fixed size not realistic on all resolutions)**  
  → Used fixed pixel values to simulate real-world size for 720p.

---

##  Final Outcome

-  Real-time video processing with consistent player IDs
-  Output video: `output/reid_output.mp4`
-  Robust pipeline combining YOLOv11 + DeepSORT
-  GitHub-friendly with automatic model loading
