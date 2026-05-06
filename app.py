from flask import Flask, request, render_template, send_from_directory
import os
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ================= UPLOAD FOLDER =================
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= LOAD MODEL =================
model = load_model("plant_disease_model.keras")

# ================= LOAD CLASS MAP =================
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

CLASS_LIST = [None] * len(class_indices)
for k, v in class_indices.items():
    CLASS_LIST[v] = k

# ================= SOLUTIONS =================
solutions = {
    "Apple___Apple_scab": "Apply fungicides like captan. Remove infected leaves.",
    "Apple___Black_rot": "Prune infected parts and apply fungicides.",
    "Apple___Cedar_apple_rust": "Use resistant varieties and fungicide sprays.",
    "Apple___healthy": "Healthy plant. Maintain care.",

    "Blueberry___healthy": "Healthy plant.",

    "Cherry_(including_sour)___Powdery_mildew": "Use sulfur fungicides and improve airflow.",
    "Cherry_(including_sour)___healthy": "Healthy plant.",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Use resistant seeds and fungicides.",
    "Corn_(maize)___Common_rust_": "Apply fungicides and grow resistant varieties.",
    "Corn_(maize)___Northern_Leaf_Blight": "Crop rotation and fungicide recommended.",
    "Corn_(maize)___healthy": "Healthy crop.",

    "Grape___Black_rot": "Remove infected fruits and apply fungicides.",
    "Grape___Esca_(Black_Measles)": "Remove infected vines.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Use fungicides and proper spacing.",
    "Grape___healthy": "Healthy plant.",

    "Orange___Haunglongbing_(Citrus_greening)": "No cure. Remove infected trees and control insects.",

    "Peach___Bacterial_spot": "Use copper sprays.",
    "Peach___healthy": "Healthy plant.",

    "Pepper,_bell___Bacterial_spot": "Use disease-free seeds and copper sprays.",
    "Pepper,_bell___healthy": "Healthy plant.",

    "Potato___Early_blight": "Apply fungicides and rotate crops.",
    "Potato___Late_blight": "Use fungicides immediately and remove infected plants.",
    "Potato___healthy": "Healthy crop.",

    "Raspberry___healthy": "Healthy plant.",
    "Soybean___healthy": "Healthy crop.",

    "Squash___Powdery_mildew": "Apply sulfur fungicides.",

    "Strawberry___Leaf_scorch": "Remove infected leaves and apply fungicides.",
    "Strawberry___healthy": "Healthy plant.",

    "Tomato___Bacterial_spot": "Use copper sprays and avoid overhead watering.",
    "Tomato___Early_blight": "Apply fungicides and remove infected leaves.",
    "Tomato___Late_blight": "Use fungicides and destroy infected plants.",
    "Tomato___Leaf_Mold": "Improve ventilation and apply fungicides.",
    "Tomato___Septoria_leaf_spot": "Remove infected leaves and apply fungicides.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Use neem oil or insecticide.",
    "Tomato___Target_Spot": "Apply fungicides.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies and remove infected plants.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants and disinfect tools.",
    "Tomato___healthy": "Healthy plant."
}

# ================= PREPROCESS =================
def preprocess(img_path):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    return x

# ================= PREDICTION =================
def predict(img_path):
    x = preprocess(img_path)
    preds = model.predict(x)

    pred_index = np.argmax(preds[0])
    confidence = np.max(preds) * 100

    label = CLASS_LIST[pred_index]
    crop, disease = label.split("___")

    solution = solutions.get(label, "No solution available")

    return crop, disease, round(confidence, 2), solution

# ================= ROUTES =================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def upload():
    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", prediction="No file selected")

    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    crop, disease, confidence, solution = predict(path)

    result = f"""
    <b>Crop:</b> {crop}<br>
    <b>Disease:</b> {disease}<br>
    <b>Confidence:</b> {confidence}%<br>
    <b>Solution:</b> {solution}
    """

    return render_template("index.html", prediction=result, image_file=filename)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)