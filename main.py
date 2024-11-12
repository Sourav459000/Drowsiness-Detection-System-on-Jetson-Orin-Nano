import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('/home/user/Desktop/sproject/sourav/drowsiness_detection_model.h5')

# Define labels for the classes
labels = {0: 'Drowsy (Closed Eyes)', 1: 'Alert (Open Eyes)'}

# Initialize the video capture (0 for default webcam)
cap = cv2.VideoCapture(0)

# Function to preprocess each frame
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (64, 64))  # Resize to match the model's input size
    frame_normalized = frame_resized / 255.0    # Normalize pixel values to [0, 1]
    frame_reshaped = np.expand_dims(frame_normalized, axis=0)  # Add batch dimension
    return frame_reshaped

# Start video capture
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Flip the frame for a mirror effect (optional)
    frame = cv2.flip(frame, 1)

    # Preprocess the frame for prediction
    preprocessed_frame = preprocess_frame(frame)
    
    # Perform prediction
    prediction = model.predict(preprocessed_frame)
    predicted_class = int(prediction[0][0] > 0.5)  # 0 for drowsy, 1 for alert

    # Overlay prediction on the frame
    label = labels[predicted_class]
    color = (0, 255, 0) if predicted_class == 1 else (0, 0, 255)  # Green for alert, red for drowsy
    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display the frame
    cv2.imshow('Drowsiness Detection', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
