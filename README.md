🌿 Crop Disease Detection System

This project is a Machine Learning / Deep Learning based web application that detects crop diseases from leaf images using a trained model.

📌 Project Description

The system allows users to upload an image of a plant leaf and predicts whether the crop is healthy or diseased. It helps farmers and users take early action to prevent crop damage.

🎯 Features
Upload crop/leaf image
Detect disease using trained model
Display prediction result
Simple and user-friendly interface
🧠 Technologies Used
Python
Flask
TensorFlow / Keras
OpenCV
HTML, CSS
📁 Project Structure
Crop-disease-detection/
│── app.py
│── plant_disease_model.keras
│── class_indices.json
│── requirements.txt
│── README.md
│
├── templates/
├── static/
├── uploads/
⚙️ Installation
Clone the repository:
git clone https://github.com/your-username/Crop-disease-detection.git
cd Crop-disease-detection
Create virtual environment:
python -m venv venv
Activate environment:
venv\Scripts\activate
Install dependencies:
pip install -r requirements.txt
▶️ Run the Application
python app.py

Open browser and go to:

http://127.0.0.1:5000/
📊 Model Details
Model Type: Convolutional Neural Network (CNN)
Framework: TensorFlow / Keras
Dataset: Plant leaf images
📷 Input
Leaf image (JPG/PNG)
📤 Output
Disease prediction (e.g., Healthy / Leaf Blight)
