import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# Define dataset path
dataset_path = "./dataset/"
labels = []
features = []

# Convert audio to spectrogram
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram_db

# Load dataset
for emotion in os.listdir(dataset_path):
    emotion_path = os.path.join(dataset_path, emotion)
    if os.path.isdir(emotion_path):
        for file in os.listdir(emotion_path):
            file_path = os.path.join(emotion_path, file)
            features.append(extract_features(file_path))
            labels.append(emotion)

# Convert to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Encode labels
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)

# Reshape features for CNN
features = features.reshape(features.shape[0], features.shape[1], features.shape[2], 1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

# Build CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(features.shape[1], features.shape[2], 1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(set(labels_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save model
model.save("voice_emotion_model.h5")

# Plot accuracy
plt.figure(figsize=(10, 5))
sns.lineplot(x=range(1, 21), y=history.history['accuracy'], label='Train Accuracy')
sns.lineplot(x=range(1, 21), y=history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training Progress")
plt.legend()
plt.show()
