import cv2
import torch
from ultralytics import YOLO
import csv
import os
import configparser

class Tracker:
    def __init__(self, model_path='yolov8x.pt'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path).to(self.device)

        # self.model.fuse = False  # Disable fusion for FP16

        # self.model.fuse() # Fuse layers in FP32

        # Temporarily switch to FP32 for fusion
        with torch.amp.autocast(device_type=self.device, enabled=False):
            self.model.fuse()
        
        if self.device == 'cuda':
            self.model = self.model.half()
            print("Using FP16 precision")

    def track_sequence(self, sequence_path, output_path='output.mp4', prediction_file='tracking_predictions.csv'):
        ini_path = os.path.join(sequence_path, 'seqinfo.ini')
        config = configparser.ConfigParser()
        config.read(ini_path)
        seq_info = config['Sequence']
        imDir = seq_info.get('imDir')
        frameRate = int(seq_info.get('frameRate'))
        seqLength = int(seq_info.get('seqLength'))
        imWidth = int(seq_info.get('imWidth'))
        imHeight = int(seq_info.get('imHeight'))
        imExt = seq_info.get('imExt')  # e.g., '.jpg'
        
        # Define the image folder path
        img_folder = os.path.join(sequence_path, imDir)
        image_files = sorted([f for f in os.listdir(img_folder) if f.endswith(imExt)])
        if not image_files:
            print("No images found in", img_folder)
            return

        # Setup VideoWriter using metadata from seqinfo.ini
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, frameRate, (imWidth, imHeight))
        
        predictions = []  # List to hold prediction rows
        frame_number = 1

        for img_name in image_files:
            img_path = os.path.join(img_folder, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Warning: Could not read image {img_path}")
                continue

            # Run tracking on the frame
            results = self.model.track(
                frame,
                persist=True,
                classes=0,    # Only detect persons
                conf=0.5,
                iou=0.5,
                verbose=False,
                device=self.device
            )

            annotated_frame = results[0].plot()
            out.write(annotated_frame)

            for box in results[0].boxes:
                # If no track id is found, skip this detection.
                if not hasattr(box, "id") or box.id is None:
                    continue
                tracked_id = int(box.id)
                # Get bounding box coordinates (in [x1, y1, x2, y2] format)
                bbox = box.xyxy.cpu().numpy()[0]
                x1, y1, x2, y2 = bbox.tolist()
                w = x2 - x1
                h = y2 - y1
                # Use detection confidence from the box
                confidence = float(box.conf.cpu().numpy()[0])
                predictions.append([frame_number, tracked_id, x1, y1, w, h, confidence])
            
            frame_number += 1
            # print(f"Processed frame {frame_number}")
        out.release()

        with open(prediction_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['frame', 'tracked_id', 'x', 'y', 'w', 'h', 'confidence'])
            for row in predictions:
                writer.writerow(row)
        print(f"Predictions saved to {prediction_file}")

if __name__ == '__main__':
    tracker = Tracker(model_path='yolov8x.pt')
    tracker.track_sequence(
        sequence_path=r"data\tracking\train\05",
        output_path=r"data\tracking\train\05.mp4",
        prediction_file=r"data\tracking\train\05.csv"
    )