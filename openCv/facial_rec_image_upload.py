import cv2, os, tkinter as tk
from tkinter import filedialog

# Haar Cascade XML dosyasını yükle
cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Initialize empty lists to store images and people's names.
known_faces = []
face_labels = []

# Get a list of all images in the TrainingImages directory.
image_files = os.listdir("TrainingImages")

# Loop through the images in the directory.
for image_name in image_files:
    # Read each image and add it to the known_faces list.
    current_image = cv2.imread(f'TrainingImages/{image_name}', cv2.IMREAD_GRAYSCALE)
    known_faces.append(current_image)

    # Extract the person's name by removing the file extension and add it to the face_labels list.
    face_labels.append(os.path.splitext(image_name)[0])


# Function to handle image selection and recognition
def select_and_recognize_image():
    # Use a file dialog to let the user select an image.
    selected_file = filedialog.askopenfilename()
    if selected_file:
        # Read the selected image.
        selected_image = cv2.imread(selected_file)
        gray_image = cv2.cvtColor(selected_image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        match_found = False  # Flag to track if a match is found.

        if len(faces) == 0:
            print("No faces found in the selected image.")
        else:
            for (x, y, w, h) in faces:
                # Crop the face region of interest (ROI)
                face_roi = gray_image[y:y + h, x:x + w]
                face_roi_resized = cv2.resize(face_roi, (100, 100))

                min_diff = float("inf")
                recognized_name = "No match"

                # Compare the detected face with known faces
                for idx, known_face in enumerate(known_faces):
                    known_face_resized = cv2.resize(known_face, (100, 100))
                    diff = cv2.norm(face_roi_resized, known_face_resized, cv2.NORM_L2)

                    if diff < min_diff:
                        min_diff = diff
                        recognized_name = face_labels[idx]

                # Determine if a match is found based on a threshold
                if min_diff < 200:
                    match_found = True
                    color = (0, 255, 0)  # Green for recognized faces
                else:
                    color = (0, 0, 255)  # Red for unrecognized faces

                # Draw a rectangle around the face and label it
                cv2.rectangle(selected_image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(selected_image, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Show the image with detected faces and labels
        cv2.imshow("Recognized Image", selected_image)
        known_faces.clear()  # Clear the list to free memory
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Create the main application window.
root = tk.Tk()
root.title("Face Recognition Program")

# Create a button to select an image for recognition.
select_button = tk.Button(root, text="Select Image for Recognition", command=select_and_recognize_image)
select_button.pack(pady=10)


# Function to quit the application.
def quit_app():
    root.quit()


# Create a quit button to exit the application.
quit_button = tk.Button(root, text="Quit", command=quit_app)
quit_button.pack(pady=10)

# Start the Tkinter event loop.
root.mainloop()
