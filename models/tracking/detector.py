# YOLO
from ultralytics import YOLO
import cv2


class Detector:
    def __init__(self, model_path='yolov8n.pt', device='cpu'):
        try:

            # self.model = YOLO(model_path, device=device)
            self.model = YOLO(model_path)
            self.device = device
            self.model.model.to(device)
            print(f"YOLO model loaded successfully on {device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def detect(self, images):
        try:
            if not isinstance(images, list):
                images = [images]
            
            results = self.model(images)

            detections = [] # list of dictionaries with detection info
            for result in results:
                for box in result.boxes:
                    coordinates = box.xyxy.cpu().numpy()[0]
                    x1, y1, x2, y2 = coordinates.tolist()

                    conf = float(box.conf.cpu().numpy()[0])
                    cls_id = int(box.cls.cpu().numpy()[0]) if hasattr(box, "cls") else -1

                    # Placeholder for person type (staff/non_staff); to be updated later by face recognition model
                    person_type = 'unknown'

                    if cls_id == 0 and conf > 0.5:
                        person_type = 'person'
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'person_type': person_type
                        }
                        detections.append(detection)

            return detections
    
        except Exception as e:
            print(f"Error detecting: {e}")
            return []

    def draw_boxes(self, image, detections):
        try:
            for detection in detections:
                x1, y1, x2, y2 = map(int, detection['bbox'])
                conf = detection['confidence']
                label_text = f"Conf:{conf:.2f}"

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
                
            return image

        except Exception as e:
            print(f"Error drawing boxes: {e}")
            return image
        
    # def __del__(self):
    #     del self.model
    #     print("YOLO model deleted successfully.")

# Test
# if __name__ == '__main__':
#     detector = Detector('yolov8n.pt', 'cpu')
#     image_path = 'test.jpg'
#     image = cv2.imread(image_path) # if not working, try with absolute path
#     detections = detector.detect(image)
#     image = detector.draw_boxes(image, detections)
#     cv2.imshow('image', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows
