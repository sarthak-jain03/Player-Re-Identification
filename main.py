import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize model and tracker
yolo = YOLO("model/yolo.pt")  # Replace with your YOLOv11 model file
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.4)

# Open video
video_path = "videos/15sec_input_720p.mp4"
cap = cv2.VideoCapture(video_path)
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

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
        cv2.rectangle(frame, (int(l), int(t)), (int(l + w), int(t + h)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (int(l), int(t) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out.write(frame)
    frame_id += 1
    print(f"[Frame {frame_id}] Tracked {len(tracks)} players")

cap.release()
out.release()
cv2.destroyAllWindows()
