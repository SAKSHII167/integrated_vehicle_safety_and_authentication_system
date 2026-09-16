import cv2
import os

if not os.path.exists('dataset'):
    os.makedirs('dataset')

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 0
print("Photos capture ho rahi hain, chehra rotate karte rahein...")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        cv2.imwrite(f"dataset/user_{count}.jpg", frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"Captured: {count}/40", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Capturing Dataset", frame)

    if count >= 40 or cv2.waitKey(100) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
print("Photos captured successfully!")