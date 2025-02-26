import os
import cv2
import numpy as np

# Set folder paths
train_dir = r"E:\PROJECT\train"
test_dir = r"E:\PROJECT\test"

# Define function to load images
def load_images_from_folder(folder):
    images = []
    labels = []
    for emotion in os.listdir(folder):  # Each subfolder represents an emotion
        emotion_path = os.path.join(folder, emotion)
        if os.path.isdir(emotion_path):  # Ensure it's a folder
            for img_name in os.listdir(emotion_path):
                img_path = os.path.join(emotion_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
                if img is not None:
                    img = cv2.resize(img, (48, 48))  # Resize to 48x48
                    images.append(img)
                    labels.append(emotion)  # Label based on folder name
    return np.array(images), np.array(labels)

# Load train and test images
X_train, y_train = load_images_from_folder(train_dir)
X_test, y_test = load_images_from_folder(test_dir)

print(f"Training images: {X_train.shape}, Labels: {y_train.shape}")
print(f"Testing images: {X_test.shape}, Labels: {y_test.shape}")

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Reshape images to add a channel dimension (required for CNNs)
X_train = X_train.reshape(-1, 48, 48, 1) / 255.0  # Normalize (0-1)
X_test = X_test.reshape(-1, 48, 48, 1) / 255.0

# Encode labels as numbers (convert emotions to 0,1,2...)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Convert labels to categorical (one-hot encoding)
num_classes = len(set(y_train))  # Count unique emotions
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print(f"Processed Training Data: {X_train.shape}, Labels: {y_train.shape}")
print(f"Processed Testing Data: {X_test.shape}, Labels: {y_test.shape}")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define CNN Model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Output layer for multi-class classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model Summary
model.summary()
# Train the CNN model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=64)
# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")
model.save("fer_model.h5")
print("Model saved successfully!")
