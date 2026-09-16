import cv2
import numpy as np
import os
import serial
import time

# ESP32 Serial setup
try:
    ser = serial.Serial('COM9', 115200, timeout=1)
    time.sleep(2)
    print("ESP32 Connected Successfully!")
except Exception as e:
    print(f"Serial Error: {e}")
    ser = None

# Train LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    face_samples = []
    ids = []
    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        faces = face_cascade.detectMultiScale(img)
        for (x, y, w, h) in faces:
            face_samples.append(img[y:y+h, x:x+w])
            ids.append(1)
    return face_samples, ids

faces, ids = get_images_and_labels('dataset')
recognizer.train(faces, np.array(ids))
print("Dataset Training Complete!")

# Recognition and Relay Logic
cam = cv2.VideoCapture(0)
last_seen_time = 0
HOLD_TIME = 3.0
relay_state = False

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    current_time = time.time()
    face_detected = False

    for (x, y, w, h) in detected_faces:
        id_num, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
        # Confidence threshold (lower means better match in LBPH)
        if confidence < 70:
            face_detected = True
            last_seen_time = current_time
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"AUTHORIZED ({round(100 - confidence)}%)", 
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "UNKNOWN", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Relay latching logic (3-second hold)
    if face_detected or (current_time - last_seen_time < HOLD_TIME and last_seen_time > 0):
        if not relay_state:
            relay_state = True
            if ser: ser.write(b'1')
            print("RELAY: ON (Door Unlocked)")
    else:
        if relay_state:
            relay_state = False
            if ser: ser.write(b'0')
            print("RELAY: OFF (Door Locked)")

    cv2.imshow("Face Recognition Door Lock", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if ser:
    ser.write(b'0')
    ser.close()
cam.release()
cv2.destroyAllWindows()
