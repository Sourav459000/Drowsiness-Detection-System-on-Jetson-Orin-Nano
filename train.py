import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

# Define paths
train_dir = '/home/user/Desktop/sproject/sourav/dataset/train'

# Image Data Augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255,            # Normalize pixel values
    validation_split=0.2        # Split for training and validation
)

# Data generators for training and validation
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),       # Resize images to 64x64
    batch_size=32,
    class_mode='binary',        # Binary classification
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary output
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# Save the model
model.save('drowsiness_detection_model.h5')

print("Model training completed and saved as 'drowsiness_detection_model.h5'.")
