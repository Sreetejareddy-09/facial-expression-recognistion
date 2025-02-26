
# Facial Expression Recognition (FER) using TensorFlow and OpenCV

## Introduction
Facial Expression Recognition (FER) is a deep learning application that detects human emotions from facial images. This project uses TensorFlow and OpenCV to classify facial expressions into different categories.

## Prerequisites
Ensure you have the following dependencies installed:

```bash
pip install tensorflow opencv-python numpy matplotlib
```

## Model Training
1. **Load and preprocess the dataset** (e.g., FER2013 dataset).
2. **Define a CNN model** using TensorFlow/Keras.
3. **Train the model** and save it for later use.

Example CNN Model:

```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(7, activation='softmax')  # 7 emotion classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

## Real-time Facial Expression Detection
Use OpenCV to capture live video feed and predict emotions:

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained FER model
model = load_model("fer_model.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48)) / 255.0
        face = np.expand_dims(face, axis=0).reshape(1, 48, 48, 1)
        prediction = model.predict(face)
        emotion = np.argmax(prediction)
        cv2.putText(frame, str(emotion), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
    
    cv2.imshow("Facial Expression Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Conclusion
This project detects facial expressions in real-time using a deep learning model trained on the FER2013 dataset. The combination of TensorFlow and OpenCV enables robust emotion classification.
