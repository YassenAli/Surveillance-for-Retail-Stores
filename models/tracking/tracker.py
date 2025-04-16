import cv2
import torch
from ultralytics import YOLO
import csv
import os 
import configparser  
import numpy 
import pandas as pd
from ast import literal_eval
import time  # To manage timing (if needed)

class Tracker:
    def __init__(self, model_path='yolo12x.pt'):
        # Load model and configure device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path).to(self.device)
        
        with torch.amp.autocast(device_type=self.device, enabled=False):
            self.model.fuse()

        if self.device == 'cuda':
            self.model = self.model.half()
            print("Using FP16 precision on", self.device)
        else:
            print("Using FP32 precision on", self.device)

    def track_sequence(self, sequence_path, output_path='output.mp4'):
        """
        Perform tracking on a sequence of images based on information provided in a 'seqinfo.ini' file.
        The method will:
        - Read sequence configuration from 'seqinfo.ini'
        - Process each image in the sequence, perform tracking, and write annotated frames to a video file.
        - Save tracking predictions in MOTChallenge format.
        """
        # Construct path to sequence configuration file
        ini_path = os.path.join(sequence_path, 'seqinfo.ini')
        config = configparser.ConfigParser()
        config.read(ini_path)
        seq_info = config['Sequence']
        imDir = seq_info.get('imDir')
        frameRate = int(seq_info.get('frameRate'))
        seqLength = int(seq_info.get('seqLength'))
        imWidth = int(seq_info.get('imWidth'))
        imHeight = int(seq_info.get('imHeight'))
        imExt = seq_info.get('imExt')

        # Build full path to the image folder and get a sorted list of image files
        img_folder = os.path.join(sequence_path, imDir)
        image_files = sorted([f for f in os.listdir(img_folder) if f.endswith(imExt)])
        if not image_files:
            print("No images found in", img_folder)
            return

        # Setup VideoWriter to generate an output video using metadata from seqinfo.ini
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, frameRate, (imWidth, imHeight))

        # Setup for writing prediction results in a text file in MOTChallenge format
        sequence_name = os.path.basename(os.path.normpath(sequence_path))
        prediction_dir = os.path.join('tracker_results', 'data')
        os.makedirs(prediction_dir, exist_ok=True)
        prediction_file = os.path.join(prediction_dir, f'{sequence_name}.txt')

        predictions = []  # List to hold prediction rows for each detection
        frame_number = 1  # Initialize frame counter

        for img_name in image_files:
            img_path = os.path.join(img_folder, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Warning: Could not read image {img_path}")
                continue

            results = self.model.track(
                frame,
                persist=True,
                classes=0,
                conf=0.5,  
                iou=0.55,  
                verbose=False,
                device=self.device
            )

            # Get the annotated frame from results and write to the video file
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

            # Loop over each detected box and format the output in the MOTChallenge style
            for box in results[0].boxes:
                # Skip detections without a track ID
                if not hasattr(box, "id") or box.id is None:
                    continue
                tracked_id = int(box.id)
                bbox = box.xyxy.cpu().numpy()[0]  # Get bounding box coordinates
                x1, y1, x2, y2 = bbox.tolist()
                w = x2 - x1
                h = y2 - y1
                confidence = float(box.conf.cpu().numpy()[0])
                # Append detection details: frame, tracked_id, bbox coordinates, confidence, and placeholders for additional data
                predictions.append([frame_number, tracked_id, x1, y1, w, h, confidence, -1, -1, -1])
            
            frame_number += 1
        out.release()

        # Write all tracking predictions to a text file
        with open(prediction_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(predictions)
        print(f"Predictions saved to {prediction_file}")

    def create_submission(self, prediction_file, sample_submission_path="sample_submission.csv", output_file="submission.csv"):
        """
        Convert tracking predictions to the submission format for the competition.
        The method performs the following:
        - Reads the sample submission file.
        - Loads the tracking predictions.
        - Groups predictions frame-by-frame.
        - Updates the submission DataFrame with the corresponding tracking detections.
        - Saves the final submission file.
        """
        # Load the sample submission file
        submission_df = pd.read_csv(sample_submission_path)
        submission_df['objects'] = submission_df['objects'].apply(literal_eval)

        # Load tracking predictions, providing column names for clarity
        tracking_data = pd.read_csv(
            prediction_file,
            header=None,
            names=['frame', 'tracked_id', 'x', 'y', 'w', 'h', 'confidence', '_1', '_2', '_3']
        )

        # Group predictions by frame number for easy lookup during submission creation
        frame_groups = tracking_data.groupby('frame')
        frame_objects = {
            frame: [
                {
                    'tracked_id': row['tracked_id'],
                    'x': round(row['x']),
                    'y': round(row['y']),
                    'w': round(row['w']),
                    'h': round(row['h']),
                    'confidence': row['confidence']
                }
                for _, row in group.iterrows()
            ]
            for frame, group in frame_groups
        }

        for idx, row in submission_df.iterrows():
            if row['objective'] == 'tracking':
                frame_num = row['frame']
                submission_df.at[idx, 'objects'] = frame_objects.get(frame_num, [])

        # Save the final submission CSV file
        submission_df.to_csv(output_file, index=False)
        print(f"Submission file created: {output_file}")


if __name__ == '__main__':
    tracker = Tracker(model_path=r'..\..\dataset\best.pt')
    tracker.track_sequence(
        sequence_path=r"..\..\data\tracking\test\01",
        output_path=r"tracker_results\data\test.mp4"
    )
    
    tracker.create_submission(
        prediction_file=r"tracker_results\data\05.txt",
        sample_submission_path="submission.csv",
        output_file="final_submission.csv"
    )