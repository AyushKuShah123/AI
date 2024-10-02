import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

# Create a directory for saving unknown faces
if not os.path.exists('unknown_faces'):
    os.makedirs('unknown_faces')

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load known faces and their names
known_face_encodings = []
known_face_names = []

# Add known faces
known_image_1 = face_recognition.load_image_file("/home/aayush/Desktop/Python_AI/AI_new/Human_detection/known_faces/2101955_Aayush.jpg")
known_encoding_1 = face_recognition.face_encodings(known_image_1)[0]
known_face_encodings.append(known_encoding_1)
known_face_names.append("Aayush")

# known_image_2 = face_recognition.load_image_file("/home/aayush/Desktop/Python_AI/AI_new/Human_detection/known_faces/vishal.jpeg")
# known_encoding_2 = face_recognition.face_encodings(known_image_2)[0]
# known_face_encodings.append(known_encoding_2)
# known_face_names.append("Vishal")

# known_image_3 = face_recognition.load_image_file("/home/aayush/Desktop/Python_AI/AI_new/Human_detection/known_faces/abhishek.jpeg")
# known_encoding_3 = face_recognition.face_encodings(known_image_1)[0]
# known_face_encodings.append(known_encoding_1)
# known_face_names.append("Abhishek")

# known_image_4 = face_recognition.load_image_file("/home/aayush/Desktop/Python_AI/AI_new/Human_detection/known_faces/damaged_potato.jpeg")
# known_encoding_4 = face_recognition.face_encodings(known_image_2)[0]
# known_face_encodings.append(known_encoding_2)
# known_face_names.append("bad")


# Open the webcam
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("http://192.168.1.4:8040/video")

# Reduce the resolution to lighten the processing load
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Set width to 320px
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Set height to 240px

# Constants for distance calculation
KNOWN_WIDTH = 14.0  # Average width of a human face in cm
FOCAL_LENGTH = 600.0  # You need to calibrate this value for your camera

# Frame count for processing every nth frame
frame_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Convert frame to grayscale (required for Haar Cascade detection)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process every nth frame to save CPU
    if frame_count % 5 == 0:
        # Get face encodings for any detected faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (x, y, w, h), face_encoding in zip(faces, face_encodings):
            # Compare detected face with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            color = (0, 0, 255)  # Red color for unknown faces

            # Find the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                color = (255, 0, 0)  # Blue color for known faces
            else:
                # Save the full frame when an unknown person is detected
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unknown_face_filename = f"unknown_faces/unknown_{timestamp}.jpg"
                cv2.imwrite(unknown_face_filename, frame)
                print(f"Unknown face captured and saved as {unknown_face_filename}")

            # Calculate distance to the detected face
            face_width = w  # Width of the detected face in pixels
            distance = (KNOWN_WIDTH * FOCAL_LENGTH) / face_width
            print(f"Distance to {name}: {distance:.2f} cm")

            # Scale face coordinates back to original frame size
            x *= 2
            y *= 2
            w *= 2
            h *= 2

            # Draw a rectangle around the face and label it
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{name} {distance:.2f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    frame_count += 1  # Increment frame counter

    # Display the resulting frame
    cv2.imshow('Face Detection and Recognition with Distance', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()