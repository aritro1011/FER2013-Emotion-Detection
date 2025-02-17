import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('emotion_detection_model.keras')  # Load the model
emotion_labels = ['Angry','Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the pre-trained face detection model (Haar cascade for frontal face)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert the frame to grayscale (face detection works better with grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Crop the face from the frame
        face = gray[y:y+h, x:x+w]
        
        # Resize the face to 48x48 (as expected by the model)
        face_resized = cv2.resize(face, (48, 48))
        
        # Normalize pixel values
        face_normalized = face_resized / 255.0
        
        # Reshape the face to (1, 48, 48, 1) to match the input shape of the model
        face_normalized = np.expand_dims(face_normalized, axis=-1)
        face_normalized = np.expand_dims(face_normalized, axis=0)
        
        # Predict the emotion
        emotion_prob = model.predict(face_normalized)
        emotion_index = np.argmax(emotion_prob)
        emotion = emotion_labels[emotion_index]
        
        # Display the emotion label on the frame
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame with face detection and emotion label
    cv2.imshow("Emotion Detection", frame)
    
    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()