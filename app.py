"""
NeuroScan — Flask Backend v7
Kamera ayri thread'de surekli okuyor, Flask son kareyi paylasiyor
"""
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import numpy as np
from PIL import Image
import io, json, os, time, threading
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '42'

import tensorflow as tf
tf.random.set_seed(42)
np.random.seed(42)
import cv2

app = Flask(__name__, static_folder='.')
CORS(app)

MODEL_PATH = "parkinson_cnn_model.h5"
IMG_SIZE   = (224, 224)

print("="*50)
print("  NeuroScan Flask API v7")
print("="*50)

print("\nCNN Modeli yukleniyor...")
model = tf.keras.models.load_model(MODEL_PATH)
_ = model(np.zeros((1,224,224,3),dtype=np.float32), training=False).numpy()
print("  ✓ CNN Model hazir")

class_indices = {"healthy": 0, "parkinson": 1}
if os.path.exists("model_results.json"):
    with open("model_results.json") as f:
        d = json.load(f)
    if "class_indices" in d:
        class_indices = d["class_indices"]

# ── KAMERA THREAD ──
class CameraThread:
    def __init__(self):
        self.frame     = None
        self.lock      = threading.Lock()
        self.running   = False
        self.cap       = None

    def start(self):
        self.running = True
        t = threading.Thread(target=self._read_loop, daemon=True)
        t.start()
        print("  ✓ Kamera thread baslatildi")

    def _read_loop(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        time.sleep(1.0)
        print(f"  Kamera acildi: {self.cap.isOpened()}")
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                with self.lock:
                    self.frame = frame.copy()
            else:
                time.sleep(0.05)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

camera = CameraThread()
camera.start()
time.sleep(2)  # Kamera isinsin

# Titreme state
tremor_state = {"active":False,"positions":[],"timestamps":[],"result":None}
tremor_lock  = threading.Lock()

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def analyze_tremor(positions, timestamps):
    if len(positions) < 15:
        return None
    positions  = np.array(positions)
    timestamps = np.array(timestamps)
    amplitude  = float(np.std(positions[:,0]) + np.std(positions[:,1]))
    centered   = positions[:,0] - np.mean(positions[:,0])
    zc         = np.sum(np.diff(np.sign(centered)) != 0)
    duration   = max(timestamps[-1] - timestamps[0], 0.1)
    frequency  = float(zc / (2 * duration))
    if amplitude > 8 and frequency >= 3.5:
        sev="siddetli"; risk="yuksek"; score=min(95,60+amplitude*1.5+frequency*3); col="#f87171"
    elif amplitude > 4 and frequency >= 2.5:
        sev="hafif-orta"; risk="orta"; score=min(75,35+amplitude*2+frequency*2); col="#fb923c"
    elif amplitude > 2:
        sev="hafif"; risk="dusuk"; score=min(40,15+amplitude*3); col="#facc15"
    else:
        sev="normal"; risk="cok dusuk"; score=min(15,amplitude*3); col="#4ade80"
    return {"amplitude":round(amplitude,2),"frequency":round(frequency,2),
            "severity":sev,"risk":risk,"risk_score":round(score,1),
            "color":col,"duration":round(duration,1),"sample_count":len(positions)}

def detect_hand(frame):
    h, w = frame.shape[:2]
    hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower  = np.array([0,  30,  60], dtype=np.uint8)
    upper  = np.array([25, 255, 255], dtype=np.uint8)
    mask   = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    area    = cv2.contourArea(largest)
    if area < 1500 or area > 25000:
        return None, None
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    cv2.drawContours(frame, [largest], -1, (74,222,128), 2)
    cv2.circle(frame, (cx,cy), 8, (99,179,237), -1)
    cv2.circle(frame, (cx,cy), 12, (255,255,255), 2)
    return cx, cy

def generate_frames():
    while True:
        frame = camera.get_frame()

        if frame is None:
            # Kamera henuz hazir degil
            blank = np.zeros((240,320,3), dtype=np.uint8)
            cv2.putText(blank, "Kamera hazirlanıyor...", (30,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,100), 1)
            _, buf = cv2.imencode('.jpg', blank)
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
            time.sleep(0.1)
            continue

        h, w = frame.shape[:2]

        cx, cy = detect_hand(frame)
        hand_detected = cx is not None

        with tremor_lock:
            is_active = tremor_state["active"]

        if is_active and hand_detected:
            with tremor_lock:
                tremor_state["positions"].append([float(cx), float(cy)])
                tremor_state["timestamps"].append(time.time())
                if len(tremor_state["timestamps"]) > 1:
                    elapsed = tremor_state["timestamps"][-1] - tremor_state["timestamps"][0]
                    if elapsed >= 5.0:
                        result = analyze_tremor(tremor_state["positions"], tremor_state["timestamps"])
                        tremor_state["result"] = result
                        tremor_state["active"] = False
                        tremor_state["positions"] = []
                        tremor_state["timestamps"] = []

        # Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (w,36), (13,17,23), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        with tremor_lock:
            active  = tremor_state["active"]
            count   = len(tremor_state["timestamps"])
            elapsed = tremor_state["timestamps"][-1]-tremor_state["timestamps"][0] if count>1 else 0

        if active and count > 1:
            remaining = max(0, 5.0 - elapsed)
            progress  = min(int((elapsed/5.0)*(w-16)), w-16)
            cv2.putText(frame, f"KAYIT: {remaining:.1f}s", (8,24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (74,222,128), 2)
            cv2.rectangle(frame, (8,28), (8+progress,33), (74,222,128), -1)
        elif hand_detected:
            cv2.putText(frame, "El tespit edildi", (8,24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (99,179,237), 2)
        else:
            cv2.putText(frame, "Elinizi gosterin", (8,24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120,145,176), 1)

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        time.sleep(0.05)  # ~20 FPS

# ── ROUTES ──
@app.route("/")
def index():
    return send_from_directory(".", "parkinson_detection_app.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Dosya bulunamadi"}), 400
    try:
        x    = preprocess(request.files["file"].read())
        pred = model(x, training=False).numpy()
        if pred.shape[-1] == 1:
            pk_prob = float(pred[0][0]); h_prob = 1.0 - pk_prob
        else:
            pk_prob = float(pred[0][class_indices.get("parkinson",1)])
            h_prob  = float(pred[0][class_indices.get("healthy",0)])
        prediction = "parkinson" if pk_prob > 0.35 else "healthy"
        confidence = pk_prob if pk_prob > 0.35 else h_prob
        return jsonify({"prediction":prediction,"confidence":round(confidence*100,2),
                        "healthy_prob":round(h_prob*100,2),"parkinson_prob":round(pk_prob*100,2),
                        "model":"MobileNetV2-NeuroScan"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/tremor/start", methods=["POST"])
def tremor_start():
    with tremor_lock:
        tremor_state.update({"active":True,"positions":[],"timestamps":[],"result":None})
    return jsonify({"status":"started"})

@app.route("/tremor/stop", methods=["POST"])
def tremor_stop():
    with tremor_lock:
        tremor_state["active"] = False
        pos = tremor_state["positions"].copy()
        ts  = tremor_state["timestamps"].copy()
    result = analyze_tremor(pos, ts)
    with tremor_lock:
        tremor_state["result"] = result
    return jsonify({"status":"stopped","result":result})

@app.route("/tremor/result")
def tremor_result():
    with tremor_lock:
        active  = tremor_state["active"]
        result  = tremor_state["result"]
        count   = len(tremor_state["positions"])
        elapsed = tremor_state["timestamps"][-1]-tremor_state["timestamps"][0] if count>1 else 0
    return jsonify({"active":active,"result":result,"count":count,"elapsed":round(elapsed,1)})

@app.route("/metrics")
def metrics():
    if os.path.exists("model_results.json"):
        with open("model_results.json") as f:
            return jsonify(json.load(f))
    return jsonify({"error":"Metrik dosyasi bulunamadi"}), 404

@app.route("/health")
def health():
    return jsonify({"status":"ok","model":MODEL_PATH})

if __name__ == "__main__":
    print(f"\n  http://127.0.0.1:5000")
    print("="*50+"\n")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)