from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import tkinter as tk
import threading

# Load pre-trained face detection and emotion detection models
face_classifier = cv2.CascadeClassifier(r'/media/kali/A083-B3E6/Coding Projects/Face Emotion Detection/haarcascade_frontalface_default.xml')
try:
    classifier = load_model(r'/media/kali/A083-B3E6/Coding Projects/Face Emotion Detection/model.h5')
except Exception as e:
    print(f'Error loading the model: {str(e)}')
    exit()

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Lock to prevent threading issues
lock = threading.Lock()

# OpenCV video capture
cap = cv2.VideoCapture(0)

# Tkinter GUI for real-time emotion display
root = tk.Tk()
root.title("Real-time Emotion Detection")

label = tk.Label(root, text='', font=('Helvetica', 16), pady=10, padx=20)
label.pack()

# Function to update the Tkinter label with detected emotion
def update_label(detected_emotion):
    with lock:
        label.config(text=f'Emotion: {detected_emotion}')

# Function to process frames and update the GUI
def process_frames():
    while True:
        _, frame = cap.read()
        detected_emotions = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Change color to red (BGR format)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                label_pred = emotion_labels[prediction.argmax()]
                detected_emotions.append(label_pred)

                # Write the emotion on the box
                cv2.putText(frame, label_pred, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Update Tkinter label with the detected emotions
        if detected_emotions:
            update_label(', '.join(detected_emotions))
        else:
            update_label('No Faces')

        # Display the video feed with padding
        cv2.imshow('Emotion Detector', cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255)))

        # Allow the user to exit the loop by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start the frame processing thread
frame_thread = threading.Thread(target=process_frames)
frame_thread.start()

# Function to handle closing the application
def close_app():
    cap.release()
    cv2.destroyAllWindows()
    root.event_generate('<Escape>')  # Simulate 'q' keypress event to exit the loop

# Button to close the application
close_button = tk.Button(root, text="Close", command=close_app)
close_button.pack()

# Tkinter main loop
root.mainloop()
