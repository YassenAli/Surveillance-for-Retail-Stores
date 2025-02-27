from models.tracking import detector
from models.tracking import tracker
import cv2


detector = detector.Detector('yolov8n.pt', 'cpu')
# image_path = 'test.jpg'
# image = cv2.imread(image_path) # if not working, try with absolute path
# detections = detector.detect(image)
# image = detector.draw_boxes(image, detections)
# cv2.imshow('image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows
# tracker = tracker.Tracker(detections)


videoPath = 'models/tracking/football-video.mp4'
video = cv2.VideoCapture(videoPath)

tracker = tracker.Tracker()

if not video.isOpened():
    print("Error opening video stream or file")
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    
    detections = detector.detect(frame)
    updated_frame = tracker.update(frame , detections)

    cv2.imshow("tracking people ", updated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
