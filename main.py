from ultralytics import YOLO
import cv2
import os
import shutil
import torch
import math

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
        torch.cuda.set_device(0)
        self.results = self.model.track(source=video_path, show=True, conf=0.3, tracker="bytetrack.yaml", classes=0)

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
                xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
                xc, yc = round(float(xc), 2), round(float(yc), 2)
                # Save the cropped object as an image
                # print(box.tolist())
                # print(id)
                path = os.path.join(self.output_dir, str(id))
                if not os.path.exists(path):
                    os.makedirs(path)
                # get the number of files in the directory
                files = os.listdir(path)
                id_file_len = len(files)
                cv2.imwrite(os.path.join(path, f'id_{id}_{id_file_len:03}_{xc}_{yc}.jpg'), ultralytics_crop_object)
            people_count = len(self.results[frame_i])
            cv2.putText(frame, f"People Count: {people_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Show the frame
            cv2.imshow('Frame', frame)
            frame_i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def classify_direction(self):
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        # run files named id_*.jpg through the classifier
        ids = os.listdir(self.output_dir)
        # get first file to classify direction
        for id in ids:
            files = os.listdir(os.path.join(self.output_dir, id))
            file = files[0]
            mv_files = []
            curx, cury = file.split('_')[-2:]
            curx, cury = float(curx), float(cury.split('.jpg')[0])
            mv_files.append(file)
            # if next file has movement in 3 in any direction, classify as moving in that direction
            for file in files[1:]:
                nextx, nexty = file.split('_')[-2:]
                nextx, nexty = float(nextx), float(nexty.split('.jpg')[0])
                if abs(nextx - curx) > 10 or abs(nexty - cury) > 10:
                    # classify as moving
                    # get the direction
                    dx = nextx - curx
                    dy = nexty - cury
                    angle = math.atan2(dy, dx)
                    angle = math.degrees(angle)
                    direction = directions[(int((angle - 22.5) % 360 / 45) + 3) % 8]
                    print(file, nextx, curx, nexty, cury, angle, direction)
                    mv_files.append(file)
                    # create a folder for the direction
                    if not os.path.exists(os.path.join(self.output_dir, id, direction)):
                        os.makedirs(os.path.join(self.output_dir, id, direction))
                    # move the files to the direction folder
                    for mv_file in mv_files:
                        shutil.move(os.path.join(self.output_dir, id, mv_file), os.path.join(self.output_dir, id, direction, mv_file))
                    mv_files = []
                    curx, cury = nextx, nexty
                else:
                    mv_files.append(file)
            if len(mv_files) > 0:
                os.makedirs(os.path.join(self.output_dir, id, "undefined"))
                for mv_file in mv_files:
                    shutil.move(os.path.join(self.output_dir, id, mv_file), os.path.join(self.output_dir, id, "undefined", mv_file))
            
    def run(self):
        self.process_frames()
        self.classify_direction()

# Usage
model_path = 'yolov8m.pt'
video_path = 'people.mp4'
tracker = PeopleTracker(model_path, video_path)
tracker.run()
