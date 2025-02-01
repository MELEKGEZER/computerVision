import cv2, numpy as np, face_recognition, os, tkinter as tk
from tkinter import filedialog

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


# Tanınan görseller için yüz kodlamaları alınır.
known_face_encodings = get_face_encodings(known_faces)


# Görsel seçimi ve tanıma işlemini yöneten bir fonksiyon tanımlanır.
def select_and_recognize_image():
    # Kullanıcının bir görsel seçmesi için dosya diyaloğu kullanılır.
    selected_file = filedialog.askopenfilename()
    if selected_file:
        # Seçilen görsel okunur.
        selected_image = cv2.imread(selected_file)

        # Görsel RGB formatına dönüştürülür.
        selected_image_rgb = cv2.cvtColor(selected_image, cv2.COLOR_BGR2RGB)

        # Seçilen görselin yüz kodlamaları alınır.
        selected_face_encodings = face_recognition.face_encodings(selected_image_rgb)

        match_found = False  # Eşleşme bulunup bulunmadığını takip etmek için bayrak.

        if not selected_face_encodings:
            print("Seçilen görselde yüz bulunamadı.")
        else:
            # Seçilen görselde tespit edilen yüzler üzerinde dönülür.
            for face_encoding in selected_face_encodings:
                # Mevcut yüz kodlaması, tanınan yüz kodlamalarıyla karşılaştırılır.
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                # En iyi eşleşmenin indeksi bulunur. Yani en iyi benzerlik.
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    # Eğer eşleşme bulunursa, tanınan kişinin ismi alınır.
                    recognized_name = face_labels[best_match_index].upper()

                    # Tanınan yüz etrafında yeşil bir dikdörtgen çizilir.
                    top, right, bottom, left = face_recognition.face_locations(selected_image_rgb)[0]
                    cv2.rectangle(selected_image, (left, top), (right, bottom), (0, 255, 0), 2,)

                    # İsmi yüzün altında gösterilir.
                    cv2.putText(selected_image, recognized_name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                (0, 255, 0), 2)

                    match_found = True  # Eşleşme bulundu bayrağı.
                    break  # Eşleşme bulunduğunda döngüden çıkılır.

            if not match_found:
                # Eğer eşleşme bulunmazsa, kırmızı bir dikdörtgen çizilir ve "No match" (Eşleşme yok) yazılır.
                top, right, bottom, left = face_recognition.face_locations(selected_image_rgb)[0]
                cv2.rectangle(selected_image, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(selected_image, "Eşleşme Yok", (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 0, 255), 2)

            # Dikdörtgen ve isim ile birlikte görsel gösterilir.
            cv2.imshow("Tanınan Görsel", selected_image)
            known_faces.clear()  # Fazlalık gereksiz kodlamaların programın yavaşlamasına neden olmasını engellemek için.
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# Ana uygulama penceresi oluşturulur.
root = tk.Tk()
root.title("Yüz Tanıma Programı")

# Tanıma için bir görsel seçmek için bir buton oluşturulur.
select_button = tk.Button(root, text="Tanıma için Görsel Seç", command=select_and_recognize_image)
select_button.pack(pady=10)


# Uygulamadan çıkmak için bir fonksiyon tanımlanır.
def quit_app():
    root.quit()


# Uygulamadan çıkmak için bir çıkış butonu oluşturulur.
quit_button = tk.Button(root, text="Çık", command=quit_app)
quit_button.pack(pady=10)

# Tkinter olay döngüsü başlatılır.
root.mainloop()
