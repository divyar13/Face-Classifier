import cv2
from keras.models import model_from_json
import numpy as np
# from keras_preprocessing.image import load_img
json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)


# Load the Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to extract features
def extract_features(image):
    # Resize to 48x48 pixels (model input size)
    resized_image = cv2.resize(image, (48, 48))
    feature = np.array(resized_image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0  # Normalize to range [0, 1]

# Open the webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    ret, im = webcam.read()
    if not ret:
        print("Failed to capture image")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (p, q, r, s) in faces:
        # Extract the face region
        face_image = gray[q:q+s, p:p+r]

        # Preprocess and predict
        try:
            features = extract_features(face_image)
            prediction = model.predict(features)
            emotion = labels[np.argmax(prediction)]  # Get emotion label

            # Draw rectangle and put emotion label
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            cv2.putText(im, emotion, (p, q-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        except Exception as e:
            print(f"Error processing face: {e}")

    # Display the video feed with emotion detection
    cv2.imshow('Emotion Detector', im)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
webcam.release()
cv2.destroyAllWindows()
