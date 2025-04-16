backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn', 'retinaface', 
            'mediapipe', 'yolov8', 'yolov11s', 'yolov11n', 'yolov11m', 'yunet', 'centerface']

models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", 
            "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet", "Buffalo_L"]

metrics = ["cosine", "euclidean", "euclidean_l2"]

import numpy as np
import pandas as pd
import pickle
import cv2
from deepface import DeepFace
import ast
import os

dataset_path = "../../data/face_identification/"
trainset = pd.read_csv(f"{dataset_path}trainset.csv")
print(trainset.head())

# Dictionary to store embeddings
embeddings_dict = {}

failed = success = 0
model_name = models[6]
detector_backend = backends[7]

for index, row in trainset.iterrows():
    image_path = f"{dataset_path}{row['image_path']}"
    person_name = row["gt"]

    try:
        embedding_obj = DeepFace.represent(image_path, model_name=model_name, detector_backend=detector_backend)[0]["embedding"]
        embeddings_dict.setdefault(person_name, []).append(embedding_obj)
        
        success += 1
        if success % 500 == 0:
            print(f"Processed {success} images")
    except Exception as e:
        print(f"Error: {e}")
        failed += 1

with open("embeddings.pkl", "wb") as f:
    pickle.dump({
        "embeddings_dict": embeddings_dict,
        "model_name": model_name,
        "distance_metric": metrics[0],
        "threshold": 0.5
    }, f)

print("Embeddings saved to embeddings.pkl")
print(f"Processed successfully: {success}")
print(f"Failed to process: {failed}")

with open("embeddings.pkl", "rb") as f:
    known_embeddings = pickle.load(f)

def recognize_face(image_path, threshold=0.5):
    try:
        embedding_obj = DeepFace.represent(image_path, model_name=model_name, detector_backend=detector_backend)[0]["embedding"]
    except Exception as e:
        print(f"Error: {e}")
        return "doesn't_exist"

    best_match = None
    best_similarity = -1

    embeddings_dict = known_embeddings["embeddings_dict"]
    
    for person, known_embedding_list in embeddings_dict.items():
        mean_known_embedding = np.mean(known_embedding_list, axis=0)

        similarity = np.dot(embedding_obj, mean_known_embedding) / (np.linalg.norm(embedding_obj) * np.linalg.norm(mean_known_embedding))

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = person

    return best_match if best_similarity >= threshold else "doesn't_exist"


test_dataset_path = f"{dataset_path}test/"
eval_set = pd.read_csv(f"{dataset_path}eval_set.csv")

for index, row in eval_set.iterrows():
    result = recognize_face(f"{test_dataset_path}{row['image_path']}")
    eval_set.at[index, "gt"] = result
    
    if index % 500 == 0:
        print(f"{index} images evaluated")

eval_set.to_csv("eval_set.csv", index=False)

submission = pd.read_csv("../../data/submission_file.csv")
filled_eval_set = pd.read_csv("/kaggle/working/eval_set.csv")

# loop through submission file
for index, row in submission.iterrows():
    if row["ID"] < 429:
        continue
    image_dict_string = row["objects"]
    image_dict = ast.literal_eval(image_dict_string)
    image_string_name = image_dict["image"]
    filename = os.path.basename(image_string_name)
    
    # loop through eval_set
    for index2, row2 in filled_eval_set.iterrows():
        if row2["image_path"] == filename:
            ground_truth = row2["gt"]
            image_dict["gt"] = ground_truth
            submission.at[index, "objects"] = image_dict
            break

submission.to_csv("submission.csv", index=False)