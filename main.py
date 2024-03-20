from ultralytics import YOLO
import cv2
import os

# Load your model
model = YOLO('yolov8m.pt')

# Open the video
cap = cv2.VideoCapture("people.mp4")

results = model.track(source="people.mp4", show=True, conf=0.3, save=True, tracker="bytetrack.yaml", classes=0)

if(not os.path.exists("data")):
    os.makedirs("data")
path = 'data/{id}/'
i = 0

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    boxes = results[i].boxes
    for box, id in zip(boxes.xyxy, boxes.id):
        id = int(id.tolist())
        x1, y1, x2, y2 = box
        # Crop the object using the bounding box coordinates
        ultralytics_crop_object = frame[int(y1):int(y2), int(x1):int(x2)]
        # Save the cropped object as an image
        print(box.tolist())
        print(id)
        if(not os.path.exists(path.format(id=id))):
            os.makedirs(path.format(id=id))
        # get the number of files in the directory
        files = os.listdir(path.format(id=id))
        ii = len(files)
        cv2.imwrite(os.path.join(path.format(id=id), f'id_{id}_{ii}.jpg'), ultralytics_crop_object)
    people_count = len(results[i])
    cv2.putText(frame, f"People Count: {people_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow('Frame', frame)
    i += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()