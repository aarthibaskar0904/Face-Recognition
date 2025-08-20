import cv2
import pickle
import numpy as np

# Load the trained model
with open("face_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load OpenCV Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Set confidence threshold for unknown detection
CONFIDENCE_THRESHOLD = 75.0  # Adjust this based on accuracy

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Starting webcam. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extract face and resize it to match the model input
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (100, 100)).flatten().reshape(1, -1)

        # Predict the class probabilities
        probs = model.predict_proba(face_resized)[0]
        predicted_label = model.classes_[np.argmax(probs)]
        confidence = np.max(probs) * 100  # Convert to percentage

        # Updated logic for unknown detection
        if confidence >= CONFIDENCE_THRESHOLD:
            label = f"{predicted_label} ({confidence:.2f}%)"
        else:
            label = "Unknown"

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the frame with the detection
    cv2.imshow("Live Face Recognition", frame)

    # Break loop on ESC key
    if cv2.waitKey(1) == 27:
        break

# Release webcam and destroy OpenCV windows
cap.release()
cv2.destroyAllWindows()
