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
import mediapipe as mp
# MediaPipe 'solutions' bazı ortamlarda eksik olabildiği için fallback ekliyoruz
try:
    from mediapipe.solutions import hands as mp_hands
    from mediapipe.solutions import drawing_utils as mp_draw
    MP_AVAILABLE = True
except (ImportError, AttributeError):
    MP_AVAILABLE = False
    print("WARNING: MediaPipe 'solutions' not found. Falling back to HSV tracking.")

from scipy.signal import butter, lfilter
from scipy.fft import fft, fftfreq

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

# ── MEDIAPIPE HAZIRLIK ──
hands = None
if MP_AVAILABLE:
    try:
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
    except Exception as e:
        print(f"MediaPipe initialization error: {e}")
        MP_AVAILABLE = False

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
        valid_cap = None
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                valid_cap = cap
                print(f"  ✓ Kamera index {i} bulundu.")
                break
        
        self.cap = valid_cap if valid_cap is not None else cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(1.0)
        print(f"  Kamera durumu: {self.cap.isOpened()}")
        
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
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
tremor_state = {
    "active": False,
    "positions": [],
    "timestamps": [],
    "result": None,
    "medication": False # İlaç durumu
}
tremor_lock  = threading.Lock()

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def analyze_tremor(positions, timestamps):
    if len(positions) < 30: # 15 yerine en az 2 saniyelik veri
        return None
    
    positions  = np.array(positions)
    timestamps = np.array(timestamps)
    duration   = max(timestamps[-1] - timestamps[0], 0.1)
    fs         = len(positions) / duration  # Örnekleme hızı (FPS)
    
    # Sinyali merkezle (DC bileşenini çıkar)
    # Kritik: X ve Y eksenlerindeki hareketlerin bileşkesi üzerinden gidelim
    # Hareketin hızındaki değişimlere odaklanmak titremeyi daha iyi yakalar
    x = positions[:, 0] - np.mean(positions[:, 0])
    y = positions[:, 1] - np.mean(positions[:, 1])
    
    # ── BANDPASS FILTRE (2Hz - 12Hz) ──
    def bandpass_filter(data, lowcut, highcut, fs, order=3):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        # Nyquist sınırı kontrolü (örnekleme hızı düşükse yüksek kesimi daralt)
        if high >= 1.0: high = 0.95
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

    try:
        x_filt = bandpass_filter(x, 2.0, 12.0, fs)
        y_filt = bandpass_filter(y, 2.0, 12.0, fs)
    except:
        x_filt = x; y_filt = y

    # ── FFT (FREKANS ANALİZİ) ──
    N = len(x_filt)
    yf = fft(x_filt)
    xf = fftfreq(N, 1 / fs)
    
    # Pozitif frekansları al ve 2-12Hz arasını seç
    pos_mask = (xf > 0) & (xf < 12)
    xf_pos = xf[pos_mask]
    yf_pos = np.abs(yf[pos_mask])
    
    if len(xf_pos) > 0:
        peak_idx = np.argmax(yf_pos)
        frequency = float(xf_pos[peak_idx])
    else:
        frequency = 0.0

    # Amplitüd: Filtrelenmiş sinyalin standart sapması (yoğunluk)
    amplitude = float(np.std(x_filt) + np.std(y_filt))
    
    # Parkinson Kriterleri: 4-6 Hz arası ritmik titremeler tipiktir
    is_parkinson_range = 3.5 <= frequency <= 6.5
    
    if amplitude > 10 and is_parkinson_range:
        sev="şiddetli"; risk="yüksek"; score=min(98, 70 + amplitude*1.5 + (6-abs(frequency-5))*5); col="#f87171"
    elif amplitude > 6 and is_parkinson_range:
        sev="orta"; risk="orta-yüksek"; score=min(85, 50 + amplitude*2); col="#fb923c"
    elif amplitude > 3:
        if is_parkinson_range:
            sev="hafif"; risk="orta"; score=min(60, 30 + amplitude*3); col="#fbbf24"
        else:
            sev="hafif (atipik)"; risk="düşük"; score=min(35, 10 + amplitude*2); col="#facc15"
    else:
        sev="normal"; risk="çok düşük"; score=min(15, amplitude*4); col="#4ade80"
        
    return {"amplitude":round(amplitude,2),"frequency":round(frequency,2),
            "severity":sev,"risk":risk,"risk_score":round(score,1),
            "color":col,"duration":round(duration,1),"sample_count":len(positions),
            "is_parkinson_freq": is_parkinson_range}

def detect_hand(frame):
    # ── YÖNTEM 1: MEDIAPIPE (Varsayılan) ──
    if MP_AVAILABLE and hands:
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                itip = hand_lms.landmark[8] # İşaret parmağı ucu
                h, w, _ = frame.shape
                cx, cy = int(itip.x * w), int(itip.y * h)
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                cv2.circle(frame, (cx,cy), 10, (99,179,237), 2)
                return cx, cy
        except:
            pass

    # ── YÖNTEM 2: HSV RENK TAKİBİ (Fallback) ──
    hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower  = np.array([0,  30,  60], dtype=np.uint8)
    upper  = np.array([25, 255, 255], dtype=np.uint8)
    mask   = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 1000:
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.drawContours(frame, [largest], -1, (74,222,128), 2)
                cv2.circle(frame, (cx,cy), 8, (99,179,237), -1)
                return cx, cy
    return None, None

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
    data = request.json or {}
    med_status = data.get("medication", False)
    with tremor_lock:
        tremor_state.update({
            "active": True,
            "positions": [],
            "timestamps": [],
            "result": None,
            "medication": med_status
        })
    return jsonify({"status":"started", "medication": med_status})

@app.route("/tremor/stop", methods=["POST"])
def tremor_stop():
    with tremor_lock:
        tremor_state["active"] = False
        pos = tremor_state["positions"].copy()
        ts  = tremor_state["timestamps"].copy()
        med = tremor_state["medication"]
    result = analyze_tremor(pos, ts)
    if result:
        result["medication"] = med
    with tremor_lock:
        tremor_state["result"] = result
    return jsonify({"status":"stopped","result":result})

@app.route("/tremor/result")
def tremor_result():
    with tremor_lock:
        active  = tremor_state["active"]
        result  = tremor_state["result"]
        med     = tremor_state["medication"]
        count   = len(tremor_state["positions"])
        elapsed = tremor_state["timestamps"][-1]-tremor_state["timestamps"][0] if count>1 else 0
    return jsonify({
        "active": active,
        "result": result,
        "medication": med,
        "count": count,
        "elapsed": round(elapsed, 1)
    })

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