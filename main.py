from ultralytics import YOLO
import cv2
import os
import shutil

class PeopleTracker:
    def __init__(self, model_path, video_path, output_dir='data'):
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.output_dir = output_dir
        try:
            shutil.rmtree(self.output_dir)
        except:
            pass
        os.makedirs(self.output_dir)
        self.cap = cv2.VideoCapture(video_path)
        self.results = self.model.track(source=video_path, show=True, conf=0.3, save=True, tracker="bytetrack.yaml", classes=0)

    def process_frames(self):
        frame_i = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            boxes = self.results[frame_i].boxes
            for box, id in zip(boxes.xyxy, boxes.id):
                id = int(id.tolist())
                x1, y1, x2, y2 = box
                # Crop the object using the bounding box coordinates
                ultralytics_crop_object = frame[int(y1):int(y2), int(x1):int(x2)]
                # Save the cropped object as an image
                print(box.tolist())
                print(id)
                path = os.path.join(self.output_dir, str(id))
                if not os.path.exists(path):
                    os.makedirs(path)
                # get the number of files in the directory
                files = os.listdir(path)
                id_file_len = len(files)
                cv2.imwrite(os.path.join(path, f'id_{id}_{id_file_len}.jpg'), ultralytics_crop_object)
            people_count = len(self.results[frame_i])
            cv2.putText(frame, f"People Count: {people_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Show the frame
            cv2.imshow('Frame', frame)
            frame_i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        self.process_frames()

# Usage
model_path = 'yolov8m.pt'
video_path = 'people.mp4'
tracker = PeopleTracker(model_path, video_path)
tracker.run()
