import os
import cv2
import numpy as np
from sklearn.svm import SVC
import pickle

# Define the dataset path
DATASET_PATH = r"E:\Aarthi\python project\face recognition\dataset"

def load_faces(dataset_path=DATASET_PATH):
    X, y = [], []
    for label in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, label)
        if not os.path.isdir(person_path):
            continue
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_resized = cv2.resize(img, (100, 100))
            X.append(img_resized.flatten())
            y.append(label)
    return np.array(X), np.array(y)

def train_and_save_model():
    print("[INFO] Loading dataset...")
    X, y = load_faces()
    print(f"[INFO] Found {len(X)} images from {len(set(y))} classes.")

    print("[INFO] Training model...")
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y)

    with open("face_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("[INFO] Model saved to face_model.pkl")

if __name__ == "__main__":
    train_and_save_model()
