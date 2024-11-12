# Drowsiness Detection System  
This project implements a real-time **Drowsiness Detection System** using a Convolutional Neural Network (CNN). It classifies video input into two categories: **Drowsy (Closed Eyes)** and **Alert (Open Eyes)**. The system is deployed on the **NVIDIA Jetson Orin Nano** using a TensorFlow `.h5` model for inference.

---

## Features  
- **Real-Time Detection**: Uses a webcam for live classification of drowsiness.  
- **Edge Deployment**: Optimized for NVIDIA Jetson Orin Nano.  
- **Lightweight Model**: Low-latency and accurate performance on edge devices.  
- **Easy Integration**: Can be used in vehicles or industrial monitoring systems.

---

## System Requirements  
### Hardware  
- **NVIDIA Jetson Orin Nano**  
- USB Camera/Webcam  

### Software  
- **Ubuntu 20.04** (recommended for Jetson)  
- NVIDIA JetPack SDK 6.0 (includes TensorFlow and CUDA support)  
- Python 3.6+  

---

## Installation  
1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/Sourav459000/drowsiness-detection.git
   cd drowsiness-detection
   ```

2. **Set Up Python Environment**:  
   Create a virtual environment (optional but recommended):  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```

4. **Download or Place the Trained Model**:  
   Ensure the trained model (`drowsiness_detection_model.h5`) is placed in the root folder of the repository:  
   ```bash
   mv /path/to/drowsiness_detection_model.h5 ./drowsiness_detection_model.h5
   ```

---

## Dataset and Training  
### Dataset Structure:
- The dataset should be structured as follows:  
  ```
  train/
    Closed_Eyes/
    Open_Eyes/
  ```

### Training:  
If needed, you can retrain the model by running the following command:  
```bash
python train.py
```

---

## Running the Application  
### Live Video Classification  
Run the script for real-time drowsiness detection:  
```bash
python main.py
```

### Key Bindings:  
- **'q'**: Exit the live classification application.

---

## Workflow  
1. **Preprocessing**: Captures video frames, resizes them, and normalizes pixel values.  
2. **Inference**: The `.h5` model classifies frames in real time.  
3. **Overlay**: Displays the classification result directly on the video feed.  

---

## Deployment on Jetson Orin Nano  
1. **Direct Model Deployment**:  
   The `.h5` model is directly loaded using TensorFlow/Keras without any conversion.  

2. **Run the Application**:  
   Use the following command on your Jetson Orin Nano to start the application:  
   ```bash
   python live_classification.py
   ```

---

## Future Enhancements  
- Add an alarm system for prolonged drowsy states.  
- Include detection for yawning or head tilts.  
- Optimize further for industrial or vehicular use cases.

---

## Contact  
For issues or contributions, reach out at:  
- **Email**: sourav280902@gmail.com, sourav.toshniwal45@gmail.com  
