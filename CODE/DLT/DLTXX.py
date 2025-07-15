import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === Step 1: Load CSV and set image folder ===
csv_path = "C:\\Users\\hp\\Documents\\DEV\\DLT\\Dataset.csv"
image_folder = "C:\\Users\\hp\\Documents\\DEV\\DLT\\images"

df = pd.read_csv(csv_path)

IMG_SIZE = (100, 100)
images = []
labels = []

# === Step 2: Load and preprocess images ===
for idx, row in df.iterrows():
    filename = str(row['id']).strip()
    label = str(row['label']).strip()
    img_path = os.path.join(image_folder, filename)

    if os.path.exists(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, IMG_SIZE)
            images.append(img)
            labels.append(label)

print(f"\n✅ Loaded {len(images)} images.")

if len(images) < 2:
    raise ValueError("❌ Not enough valid images for training.")

# === Step 3: Preprocess data ===
images = np.array(images, dtype='float32') / 255.0
images = images[..., np.newaxis]  # Add channel dimension
labels = np.array(labels)

le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_cat = to_categorical(labels_encoded)

# === Step 4: Split data ===
X_train, X_test, y_train, y_test = train_test_split(
    images, labels_cat, test_size=0.2, stratify=labels_cat, random_state=42
)

# === Step 5: Data augmentation ===
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(X_train)

# === Step 6: Build CNN ===
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === Step 7: Callbacks ===
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# === Step 8: Train ===
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=30,
    callbacks=[early_stop]
)

# === Step 9: Save model ===
model.save("custom_face_cnn_improved.h5")
print("✅ Model saved as custom_face_cnn_improved.h5")

# === Step 10: Prediction function ===
def predict_custom_face(img_path):
    if not os.path.exists(img_path):
        print(f"❌ File not found: {img_path}")
        return "Unknown"

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Failed to read image: {img_path}")
        return "Invalid"

    img = cv2.resize(img, IMG_SIZE)
    img = img.astype('float32') / 255.0
    img = img[np.newaxis, ..., np.newaxis]
    pred = model.predict(img)
    idx = np.argmax(pred)
    label = le.inverse_transform([idx])[0]
    confidence = np.max(pred)
    print(f"🔍 Predicted: {label} (Confidence: {confidence:.2f})")
    return label

# === Step 11: Show predictions for random multiple images ===
N = 5  # Number of images to display
valid_paths = []

# Build list of valid image paths
for _, row in df.iterrows():
    filename = str(row['id']).strip()
    full_path = os.path.join(image_folder, filename)
    if os.path.exists(full_path):
        valid_paths.append((filename, full_path))

# Shuffle for different output each run
random.shuffle(valid_paths)

if len(valid_paths) == 0:
    print("❌ No valid images found in folder for prediction.")
else:
    plt.figure(figsize=(15, 5))
    for i, (filename, path) in enumerate(valid_paths[:N]):
        label = predict_custom_face(path)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            plt.subplot(1, N, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(f"Predicted:\n{label}")
            plt.axis("off")

    plt.tight_layout()
    plt.show()
