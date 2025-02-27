from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

class Tracker:
    def __init__(self):

        self.tracker = DeepSort(max_age=30, n_init=4, embedder="mobilenet")  #mobilenet is a model for extracting features

    def update(self, frame, detections):
        
        deepSort_detections = []
        for detected in detections:
            bbox = detected['bbox']
            width = bbox[2] - bbox[0]  
            height = bbox[3] - bbox[1]  
            deepSort_detections.append(([bbox[0], bbox[1], width, height], detected['confidence'], 0))  

        tracks = self.tracker.update_tracks(deepSort_detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlwh()  # Convert to [x, y, w, h] format
            x, y, w, h = map(int, bbox)

            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return frame