import os
import cv2
import gdown
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

model_path = "model/yolo.pt"
file_id = "1vk_YNbDv270ue4AVDzJaTrAljRNscvJu"
gdrive_url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(model_path):
    os.makedirs("model", exist_ok=True)
    print("Downloading YOLO model from Google Drive...")
    gdown.download(gdrive_url, model_path, quiet=False)


yolo = YOLO(model_path)
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.4)


# Open video
video_path = "videos/15sec_input_720p.mp4"
cap = cv2.VideoCapture(video_path)
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)


import random

id_colors = {}

def get_color_for_id(track_id):
    if track_id not in id_colors:
        random.seed(track_id)
        id_colors[track_id] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    return id_colors[track_id]


# Video writer

out = cv2.VideoWriter("output/reid_output.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (frame_width, frame_height))

frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = []
    results = yolo(frame)[0]

    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = r
        if int(cls) == 0:  # Assuming class 0 = person
            bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]  # x, y, w, h
            detections.append(([bbox[0], bbox[1], bbox[2], bbox[3]], conf, 'player'))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, w, h = track.to_ltrb()

        color = get_color_for_id(track_id)

        # Draw box with unique color
        cv2.rectangle(frame, (int(l), int(t)), (int(l + w), int(t + h)), color, 1)

        # ID label with matching color
        cv2.putText(frame, f"ID {track_id}", (int(l), int(t) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    out.write(frame)
    frame_id += 1
    print(f"[Frame {frame_id}] Tracked {len(tracks)} players")

cap.release()
out.release()
cv2.destroyAllWindows()
