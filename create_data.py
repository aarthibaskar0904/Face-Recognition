import cv2 #access camera 
import os  #used for directory orientation function


## to create face dataset with limit of 50 images
def create_face_dataset(
    person_name="Aarthi",
    base_path=r"E:\Aarthi\python project\face recognition\dataset",
    face_limit=50
):

## to create and save the face image in particular directory
    dataset_path = os.path.join(base_path, person_name)
    os.makedirs(dataset_path, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    count = 0
    print(f"[INFO] Starting data collection for '{person_name}'... Press ESC to stop early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (100, 100))

            img_path = os.path.join(dataset_path, f"{count}.jpg")
            cv2.imwrite(img_path, face_resized)

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"{count}/{face_limit}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("Collecting Faces - Aarthi", frame)

        if count >= face_limit or cv2.waitKey(1) == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved {count} face(s) to: {dataset_path}")

if __name__ == "__main__":
    create_face_dataset()
