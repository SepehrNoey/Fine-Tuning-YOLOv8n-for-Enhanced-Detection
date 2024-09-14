import cv2
from ultralytics import YOLO
from ultralytics.solutions import object_counter

modelName = "yolov8n"
videoName = "10"
resVideoName = modelName + f"_video{videoName}"

model = YOLO(f'{modelName}.pt')
cap = cv2.VideoCapture(f'{videoName}.mp4')
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

region_points = [(0, 0), (w, 0), (w, h), (0, h)]

classes_to_count = [0, 1, 2, 7]  # person, bicycle, car, truck

video_writer = cv2.VideoWriter(f"{resVideoName}.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

counter = object_counter.ObjectCounter(
    view_img=False,
    reg_pts=region_points,
    names=model.names,
    draw_tracks=True,
    line_thickness=2,
    view_in_counts=False,
    view_out_counts=False
)

# Initialize counts for each class
class_counts = {cls: 0 for cls in classes_to_count}

def draw_class_labels(frame, counts, class_names, colors):
    """Draw class counts at the top right corner of the frame."""
    x, y = frame.shape[1] - 200, 10  # Position for labels
    label_height = 40
    background_color = (0, 0, 0)  # Black background for labels
    text_color = (255, 255, 255)  # White text color

    for class_id in classes_to_count:
        label = f"{class_names[class_id]}: {counts[class_id]}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_width, label_height = label_size
        cv2.rectangle(frame, (x - 10, y - 10), (x + label_width + 10, y + label_height + 10), background_color, -1)
        cv2.putText(frame, label, (x, y + label_height), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        y += label_height + 20  # Move down for the next label

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    tracks = model.predict(im0, show=False, classes=classes_to_count, conf=0.3, verbose=False)

    # resetting counts
    class_counts = {cls: 0 for cls in classes_to_count}

    # for drawing confidence on the boxes
    for track in tracks:
        for box in track.boxes:
            class_id = int(box.cls)
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf

            # draw confidence too
            cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{model.names[class_id]} {confidence.item():.2f}"
            cv2.putText(im0, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # updating counts
            class_counts[class_id] += 1

    # draw class labels
    draw_class_labels(im0, class_counts, model.names, [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)])

    # Process counts
    im0 = counter.start_counting(im0, tracks)

    # Write the frame to the output video
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
