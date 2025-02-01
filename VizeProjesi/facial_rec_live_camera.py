import tkinter as tk, numpy as np, cv2, os, face_recognition
from datetime import datetime

# Görselleri ve kişilerin isimlerini saklamak için boş listeler başlatılır.
known_faces = []
face_labels = []

# TrainingImages dizinindeki tüm görsellerin listesi alınır.
image_files = os.listdir("TrainingImages")

# Dizin içerisindeki görseller üzerinde dönülür.
for image_name in image_files:
    # Her bir görsel okunur ve known_faces listesine eklenir.
    current_image = cv2.imread(f'TrainingImages/{image_name}')
    known_faces.append(current_image)

    # Görselin uzantısı kaldırılarak kişinin ismi çıkarılır ve face_labels listesine eklenir.
    face_labels.append(os.path.splitext(image_name)[0])


# Görsellerden yüz kodları almak için bir fonksiyon tanımlanır.
def get_face_encodings(images):
    encoding_list = []
    for image in images:
        # Görsel RGB formatına dönüştürülür. RGB, Kırmızı Yeşil Mavi demektir.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Görselde bulunan ilk yüz için yüz kodlaması alınır.
        face_encoding = face_recognition.face_encodings(image)[0]
        encoding_list.append(face_encoding)
    return encoding_list


# Tanınan yüzü belgelemek için bir fonksiyon tanımlanır.
def document_recognised_face(name, filename='records.csv'):
    # Geçerli tarih YYYY-AA-GG formatında alınır.
    capture_date = datetime.now().strftime("%Y-%m-%d")

    # Belirtilen CSV dosyasının var olup olmadığı kontrol edilir.
    if not os.path.isfile(filename):
        # Dosya yoksa, dosya oluşturulur ve başlık yazılır.
        with open(filename, 'w') as f:
            f.write('Name,Date,Time')  # Dosya oluşturulur ve başlık yazılır.

    # CSV dosyası okuma ve yazma için açılır ('r+')
    with open(filename, 'r+') as file:
        # Dosyadaki tüm satırlar bir listeye okunur.
        lines = file.readlines()

        # Mevcut satırlardan isimler çıkarılır.
        existing_names = [line.split(",")[0] for line in lines]

        # Sağlanan ismin mevcut isimler arasında olup olmadığı kontrol edilir.
        if name not in existing_names:
            # Geçerli saat HH:DD:SS formatında alınır.
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")

            # Yeni kayıt, isim, tarih ve saat ile CSV dosyasına yazılır.
            file.write(f'\n{name},{capture_date},{current_time}')


# Tanınan görseller için yüz kodlamaları alınır.
known_face_encodings = get_face_encodings(known_faces)


# Yüz tanıma programını başlatacak bir fonksiyon tanımlanır.
def start_recognition_program():
    # Video yakalama için bir webcam açılır. Bilgisayarınızın webcam’ini kullanıyorsanız, 1’i 0 yapın.
    # Harici bir webcam kullanıyorsanız, 1 olarak bırakın.
    video_capture = cv2.VideoCapture(0)

    while True:
        # Webcam'den bir kare alınır.
        frame = video_capture.read()

        # Kare başarısız değilse (yani başarılı bir kare yakalanmışsa) kontrol edilir.
        if frame is not None:
            frame = frame[1]  # Kare, video_capture.read() fonksiyonunun döndürdüğü ikilinin ikinci elemanıdır.

            # Görsel boyutu küçültülür.
            resized_frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            # Mevcut karede yüzler tespit edilir.
            face_locations = face_recognition.face_locations(resized_frame)

            # Mevcut karede tespit edilen yüzler için yüz kodlamaları alınır.
            current_face_encodings = face_recognition.face_encodings(resized_frame, face_locations)

            # Mevcut karedeki tespit edilen yüzler üzerinde dönülür.
            for face_encoding, location in zip(current_face_encodings, face_locations):
                # Mevcut yüz kodlaması, tanınan yüz kodlamalarıyla karşılaştırılır.
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                # En iyi eşleşmenin indeksi bulunur. Yani en iyi benzerlik.
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    # Eğer eşleşme bulunursa, tanınan kişinin ismi alınır.
                    recognized_name = face_labels[best_match_index].upper()

                    # Yüzün konum koordinatları çıkarılır.
                    top, right, bottom, left = location
                    top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

                    # Tanınan yüz etrafında bir dikdörtgen çizilir.
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                    # Dolu bir dikdörtgen çizilir ve ismi yüzün üstünde gösterilir.
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, recognized_name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (255, 255, 255), 2)
                    document_recognised_face(recognized_name)

            # Tanınan yüzlerle birlikte görsel gösterilir.
            cv2.imshow("Webcam", frame)

        # Tuş basımı kontrol edilir
        key = cv2.waitKey(1) & 0xFF

        # 'q' tuşuna basılırsa programdan çıkılır.
        if key == ord('q'):
            break

    # Video kaydı serbest bırakılır ve tüm OpenCV pencereleri kapatılır.
    video_capture.release()
    cv2.destroyAllWindows()


# Ana uygulama penceresi oluşturulur.
root = tk.Tk()
root.title("Yüz Tanıma Programı")

# Bir etiket oluşturulur
label = tk.Label(root, text="Yüz tanıma programını başlatmak için butona tıklayın")
label.pack(pady=10)

# Programı başlatmak için bir buton oluşturulur
start_button = tk.Button(root, text="Tanımayı Başlat", command=start_recognition_program)
start_button.pack(pady=10)


# Uygulamadan çıkmak için bir fonksiyon. Webcam yayını kapatmak için 'q' tuşuna basılmalıdır.
def quit_app():
    root.quit()
    cv2.destroyAllWindows()


# Uygulamadan çıkmak için bir çıkış butonu oluşturulur.
exit_button = tk.Button(root, text="Kapat", command=quit_app)
exit_button.pack(pady=10)

# Tkinter olay döngüsü başlatılır.
root.mainloop()
