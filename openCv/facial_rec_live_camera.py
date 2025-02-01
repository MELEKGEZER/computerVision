import tkinter as tk
import cv2
import os
import numpy as np
from datetime import datetime

# Haar Cascade yükle
cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Kayıtlı yüzleri ve etiketlerini yükle
known_faces = []
face_labels = []

# TrainingImages klasöründen yüz resimlerini ve etiketlerini oku
image_files = os.listdir("TrainingImages")
for image_name in image_files:
    # Her resmi oku
    current_image = cv2.imread(f'TrainingImages/{image_name}', cv2.IMREAD_GRAYSCALE)
    known_faces.append(current_image)
    # Dosya adından ismi çıkar
    face_labels.append(os.path.splitext(image_name)[0])


# Yüz eşleştirme fonksiyonu (basit histogram karşılaştırması)
def match_face(detected_face):
    detected_face = cv2.resize(detected_face, (100, 100))  # Boyutları normalize et
    min_diff = float("inf")
    label = "Unknown"

    for idx, known_face in enumerate(known_faces):
        # Kayıtlı yüzleri aynı boyuta getir
        known_face_resized = cv2.resize(known_face, (100, 100))
        # Histogram farkını hesapla
        diff = cv2.norm(detected_face, known_face_resized, cv2.NORM_L2)
        if diff < min_diff:
            min_diff = diff
            label = face_labels[idx]

    return label if min_diff < 200 else "Unknown"  # Belirli bir eşik değeri kullan


# Tespit edilen yüzleri belgeleyen fonksiyon
def document_detected_face(name, filename='records.csv'):
    capture_date = datetime.now().strftime("%Y-%m-%d")
    if not os.path.isfile(filename):
        with open(filename, 'w') as f:
            f.write('Name,Date,Time')

    with open(filename, 'r+') as file:
        lines = file.readlines()
        existing_names = [line.split(",")[0] for line in lines]

        if name not in existing_names:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            file.write(f'\n{name},{capture_date},{current_time}')


# Yüz algılama ve eşleştirme programı
def start_detection_program():
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = gray_frame[y:y + h, x:x + w]  # Algılanan yüz
                recognized_label = match_face(face_roi)  # Yüzü eşleştir

                # Yüzü çerçevele ve ismi göster
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, recognized_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                # Tespit edilen yüzü belgeleyin
                document_detected_face(recognized_label)

            # Çıktıyı göster
            cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Çıkmak için 'q' tuşuna basın
            break

    video_capture.release()
    cv2.destroyAllWindows()


# Tkinter arayüzü
root = tk.Tk()
root.title("Face Detection Program")

label = tk.Label(root, text="Click the button to start the face detection program")
label.pack(pady=10)

start_button = tk.Button(root, text="Start Detection", command=start_detection_program)
start_button.pack(pady=10)

def quit_app():
    root.quit()
    cv2.destroyAllWindows()


exit_button = tk.Button(root, text="Close", command=quit_app)
exit_button.pack(pady=10)

root.mainloop()
