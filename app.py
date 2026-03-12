"""
NeuroScan — Flask Backend (Sabit Tahmin Versiyonu)
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from PIL import Image
import io, json, os
import warnings
warnings.filterwarnings('ignore')

# TensorFlow'u deterministik modda başlat
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '42'

import tensorflow as tf
tf.random.set_seed(42)
np.random.seed(42)

app = Flask(__name__, static_folder='.')
CORS(app)

MODEL_PATH = "parkinson_cnn_model.h5"
IMG_SIZE   = (224, 224)

print("="*50)
print("  NeuroScan Flask API")
print("="*50)

# ── Modeli bir kere yükle, sabit tut ──
print("\nModel yükleniyor...")
model = tf.keras.models.load_model(MODEL_PATH)

# Modeli inference moduna al (Dropout + BN kapalı)
# Bunun için dummy bir prediction yapıyoruz
dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
_ = model(dummy, training=False).numpy()

print(f"  ✓ Model hazır: {MODEL_PATH}")

# Sınıf bilgisi
class_indices = {"healthy": 0, "parkinson": 1}
class_names   = ["healthy", "parkinson"]
if os.path.exists("model_results.json"):
    with open("model_results.json") as f:
        d = json.load(f)
    if "class_indices" in d:
        class_indices = d["class_indices"]
        class_names   = list(class_indices.keys())
print(f"  ✓ Sınıflar: {class_indices}")
print(f"\n  http://127.0.0.1:5000")
print("="*50 + "\n")

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route("/")
def index():
    return send_from_directory(".", "parkinson_detection_app.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Dosya bulunamadı"}), 400
    try:
        img_bytes = request.files["file"].read()
        x = preprocess(img_bytes)

        # training=False → Dropout ve BatchNorm tahmin modunda
        pred = model(x, training=False).numpy()

        if pred.shape[-1] == 1:
            pk_prob = float(pred[0][0])
            h_prob  = 1.0 - pk_prob
        else:
            pk_idx  = class_indices.get("parkinson", 1)
            h_idx   = class_indices.get("healthy", 0)
            pk_prob = float(pred[0][pk_idx])
            h_prob  = float(pred[0][h_idx])

        prediction = "parkinson" if pk_prob > 0.35 else "healthy"  # Düşük eşik → Recall artışı
        confidence = pk_prob if pk_prob > 0.5 else h_prob

        return jsonify({
            "prediction":     prediction,
            "confidence":     round(confidence * 100, 2),
            "healthy_prob":   round(h_prob  * 100, 2),
            "parkinson_prob": round(pk_prob * 100, 2),
            "model":          "MobileNetV2-NeuroScan"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/metrics")
def metrics():
    if os.path.exists("model_results.json"):
        with open("model_results.json") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Metrik dosyası bulunamadı"}), 404

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=False)