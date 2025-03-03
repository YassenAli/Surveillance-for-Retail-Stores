import sys
import numpy as np
sys.setrecursionlimit(10000)

from deepface import DeepFace
# from deepface.commons import functions

class DeepFaceRecognizer:
    def __init__(self, db_path, model_name="Facenet", distance_metric="cosine", threshold=0.4, enforce_detection=False):
        self.db_path = db_path
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.threshold = threshold
        self.enforce_detection = enforce_detection

    def recognize(self, test_img_path):
        try:
            # Validate input image first
            # if not functions.detect_face(test_img_path):
            #     return "doesn't_exist", None
            # DeepFace.find will search the db_path for similar faces using the chosen model and metric.
            results = DeepFace.find(
                img_path=test_img_path,
                db_path=self.db_path,
                model_name=self.model_name,
                distance_metric=self.distance_metric,
                enforce_detection=self.enforce_detection
            )
            
            if not results or results[0].empty:
                return "doesn't_exist", None
            
            df = results[0]
            best_match = df.iloc[0]
            best_distance = best_match["distance"]
            if best_distance < self.threshold:
                return best_match["identity"], best_distance
            else:
                return "doesn't_exist", best_distance
        except Exception as e:
            print("Error in recognize:", str(e))
            return "doesn't_exist", None

# --- Testing Code ---
if __name__ == '__main__':
    db_path = r"data\face_identification\train"  # Adjust this path to your training images database
    test_img_path = r"data\face_identification\test\11684.jpg"  # Adjust to your test image path
    
    recognizer = DeepFaceRecognizer(
        db_path=db_path,
        model_name="Facenet",
        distance_metric="cosine",
        threshold=0.4,
        enforce_detection=False
    )
    
    identity, distance = recognizer.recognize(test_img_path)
    print("Recognized Identity:", identity)
    print("Distance:", distance)
