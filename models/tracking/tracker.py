from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2




class Tracker :
    def __init__(self): 
        # max age -> how many frames to keep searching for person before consider it as lost
        # n_init -> how many frames the person must be detected to be considered as a valid detection and start tracking it
        self.tracker = DeepSort(max_age=30 , n_init=4)  
        
    def update(self ,frame , detections):
        deepSort_detections =[]
        for detected in detections :
            bbox = detected['bbox']
            confidence = detected['confidence']
            person_type = detected['person_type']
            deepSort_detections.append((bbox , confidence , person_type))

        tracks = self.tracker.update_tracks(deepSort_detections , frame= frame  ) 

        for track in tracks:
            if not track.is_confirmed():
                continue

            trackID = track.track_id
            bbox = track.to_tlwh()
            x , y , w  , h = map(int , bbox)

            # cv2.rectangle(frame, (x, y), ( w, h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {trackID}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return frame


        