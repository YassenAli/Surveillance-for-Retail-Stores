import cv2
import torch
from ultralytics import YOLO

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

    def track_objects(self, video_path, output_path='output.mp4'):
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.track(
                frame,
                persist=True,
                classes=0,
                conf=0.5,
                iou=0.5,
                verbose=False,
                device=self.device
            )

            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            cv2.imshow('Tracking', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    tracker = Tracker(model_path='yolov8x.pt')
    tracker.track_objects(
        video_path=r"Y:\Fawry Competition\surveillance-for-retail-stores\Surveillance-for-Retail-Stores\models\tracking\football-video.mp4",
        output_path="../../tracking_results.mp4"
    )