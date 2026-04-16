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
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# MediaPipe Tasks API Hazırlığı
# hand_landmarker.task dosyası proje kök dizininde olmalıdır
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
try:
    detector = vision.HandLandmarker.create_from_options(options)
    MP_AVAILABLE = True
    print("  [OK] MediaPipe Tasks API hazir")
except Exception as e:
    MP_AVAILABLE = False
    print(f"WARNING: MediaPipe initialization error: {e}")

from scipy.signal import butter, lfilter, find_peaks
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
print("  [OK] CNN Model hazir")

class_indices = {"healthy": 0, "parkinson": 1}
if os.path.exists("model_results.json"):
    with open("model_results.json") as f:
        d = json.load(f)
    if "class_indices" in d:
        class_indices = d["class_indices"]

# ── MEDIAPIPE HAZIRLIK (Eski solutions artik kullanilmiyor) ──
# detector objesi yukarida global olarak initialize edildi

# ── KAMERA THREAD ──
class CameraThread:
    def __init__(self):
        self.frame     = None
        self.lock      = threading.Lock()
        self.running   = False
        self.cap       = None
        self.thread    = None

    def start(self):
        with self.lock:
            if self.running: return True
            
            # Robust Camera Detection (Indices 0, 1, 2)
            valid_cap = None
            for i in range(4): # Try one more index
                try:
                    print(f"  [DEBUG] Kamera index {i} deneniyor...", flush=True)
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        valid_cap = cap
                        print(f"  [OK] Kamera index {i} bulundu.", flush=True)
                        break
                    else:
                        cap.release()
                except Exception as e:
                    print(f"  [DEBUG] Index {i} hatasi: {str(e)}", flush=True)
            
            if not valid_cap or not valid_cap.isOpened():
                print("  [DEBUG] DSHOW başarısız, normal mod deneniyor...", flush=True)
                try:
                    valid_cap = cv2.VideoCapture(0)
                except Exception as e:
                    print(f"  [DEBUG] Fallback hatasi: {str(e)}", flush=True)
                
            if not valid_cap or not valid_cap.isOpened():
                print("  [ERROR] Kamera acilamadi!", flush=True)
                return False

                
            self.cap = valid_cap
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 20)
            
            self.running = True
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            print("  [OK] Kamera donanimi aktif")
            return True


    def stop(self):
        with self.lock:
            self.running = False
            self.frame = None
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        print("  [OK] Kamera donanimi kapatildi")

    def _read_loop(self):
        # Camera is now initialized in start()
        time.sleep(1.0) # Give camera time to warm up
        print(f"  Kamera okuma döngüsü başladı. Durum: {self.cap.isOpened()}")


        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                frame = cv2.flip(frame, 1)
                with self.lock:
                    self.frame = frame.copy()
            else:
                time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

camera = CameraThread()
# Acilista kamera kapatildi, artik butonlarla acilmasi bekleniyor
# camera.start() silindi

# Titreme state
tremor_state = {
    "active": False,
    "positions": [],
    "timestamps": [],
    "hand_sizes": [], # Mesafe normalizasyonu için
    "result": None,
    "medication": False # İlaç durumu
}
tremor_lock  = threading.Lock()

# ── SEANS BELLEĞİ (ON/OFF KIYASLAMASI İÇİN) ──
session_data = {
    "tremor": {"none": None, "med": None},
    "tapping": {"none": None, "med": None}
}

# Bradikinezi (Finger Tapping) state
tapping_state = {
    "active": False,
    "distances": [],
    "timestamps": [],
    "result": None,
    "medication": False,
    "current_dist": 0.0 
}
tapping_lock = threading.Lock()

class BradikineziaAnalyzer:


    def __init__(self, fps=20):
        self.fps = fps
    
    def analyze(self, distances, timestamps):
        if len(distances) < 20: 
            return None
        
        d = np.array(distances)
        t = np.array(timestamps)
        duration = max(t[-1] - t[0], 0.1)
        fs = len(d) / duration
        
        # ── SİNYAL TEMİZLEME ──
        def lowpass_filter(data, cutoff, fs, order=3):
            nyq = 0.5 * fs
            if cutoff >= nyq: cutoff = nyq * 0.9
            b, a = butter(order, cutoff/nyq, btype='low')
            return lfilter(b, a, data)
        
        try:
            d_filt = lowpass_filter(d, 4.0, fs)
        except:
            d_filt = d

        # ── YÜKSEK HASSASİYETLİ TEPE TESPİTİ (PROMINENCE) ──
        # d_filt zaten (dist / hand_size) birimindedir.
        # Belirginlik (Prominence) 0.08: Çevresinden el boyunun %8'i kadar yükselen her dalgayı vuruş sayar.
        # distance parametresi çökmemesi için en az 1 olmalıdır.
        peaks, properties = find_peaks(d_filt, prominence=0.08, distance=max(1, int(fs * 0.1)))

        
        if len(peaks) < 3:
            # Sinyal kalitesi kontrolü
            if np.ptp(d_filt) < 0.15:
                # Hareket çok küçük veya el çok uzak
                return {"error": "Hareket menzili çok dar. Lütfen parmaklarınızı daha geniş açın veya elinizi yaklaştırın."}
            return None

        # Metrik 1: Vuruş hızı (taps/sn)

        tap_rate = len(peaks) / duration
        
        # Metrik 2: Amplitüd Azalması (Windowed Decay)
        # Sadece lineer regresyon yerine, ilk 3 vuruş ile son 3 vuruşun ortalamasını kıyaslıyoruz.
        # Bu yöntem MDS-UPDRS'teki "progressive fatigue" mantığına daha uygundur.
        prominences = properties['prominences']
        if len(prominences) >= 4:
            first_avg = np.mean(prominences[:3])
            last_avg  = np.mean(prominences[-3:])
            amp_decay = 1.0 - (last_avg / (first_avg + 1e-6))
            amp_decay = max(0.0, float(amp_decay))
        else:
            amp_decay = 0.0
            
        # Metrik 3: Ritim Değişkenliği (CoV)
        intervals = np.diff(t[peaks])
        cov = np.std(intervals) / (np.mean(intervals) + 1e-6)
            
        # MDS-UPDRS Puanlama (Klinik Standartlar)
        score = self._compute_updrs(tap_rate, amp_decay, cov)
        
        # Sonucu Seans Belleğine Kaydet
        m_key = "med" if tapping_state["medication"] else "none"
        session_data["tapping"][m_key] = {
            "rate": tap_rate,
            "decay": amp_decay,
            "score": score
        }

        # Renk ve Şiddet
        severities = {
            4: ("şiddetli", "#f87171"),
            3: ("belirgin", "#f87171"),
            2: ("orta", "#fb923c"),
            1: ("hafif", "#fbbf24"),
            0: ("normal", "#4ade80")
        }
        sev, col = severities.get(score, ("normal", "#4ade80"))
            
        recommendations = {
            0: "Vuruş hızı ve genliği normal sınırlarda. Klinik bulgu saptanmadı.",
            1: "Hafif hız kaybı veya yorulma saptandı. Erken evre bulgusu olabilir.",
            2: "Belirgin yorulma ve vuruş genliğinde daralma (MDS-UPDRS 2).",
            3: "Ciddi bradikinezi bulguları. Vuruşlar arasında duraksamalar gözleniyor.",
            4: "Vuruş düzeni sürdürülemiyor, ileri derece hareket kısıtlılığı."
        }
        
        return {
            "tap_rate": round(tap_rate, 2),
            "amp_decay_pct": round(amp_decay * 100, 1),
            "rhythm_cov": round(cov, 3),
            "updrs_score": score,
            "severity": sev,
            "recommendation": recommendations.get(score, "Analiz tamamlandı."),
            "color": col,
            "duration": round(duration, 1),
            "tap_count": len(peaks),
            "risk_score": round(score * 25, 1) 
        }
    
    def _compute_updrs(self, rate, decay, cov):
        # 0: Normal, 1: Hafif, 2: Orta, 3: Belirgin, 4: Şiddetli

        
        # Hız (Speed) - MDS-UPDRS Benchmarkları
        if rate < 1.2: s1 = 4
        elif rate < 1.8: s1 = 3
        elif rate < 2.5: s1 = 2
        elif rate < 3.5: s1 = 1
        else: s1 = 0
        
        # Genlik Kaybı (Amplitude Decrement)
        if decay > 0.60: s2 = 4
        elif decay > 0.40: s2 = 3
        elif decay > 0.25: s2 = 2
        elif decay > 0.10: s2 = 1
        else: s2 = 0
        
        # Ritim (Rhythm)
        if cov > 0.50: s3 = 4
        elif cov > 0.35: s3 = 3
        elif cov > 0.20: s3 = 2
        elif cov > 0.10: s3 = 1
        else: s3 = 0
        
        # Klinik ağırlıklı ortalama
        final_score = round((s1 * 0.40) + (s2 * 0.50) + (s3 * 0.10))
        return min(4, final_score)


tapping_analyzer = BradikineziaAnalyzer(fps=20)

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def analyze_tremor(positions, timestamps, hand_sizes):
    if len(positions) < 30: 
        return None
    
    positions  = np.array(positions)
    timestamps = np.array(timestamps)
    h_sizes    = np.array(hand_sizes) if hand_sizes else np.array([1.0])
    
    avg_h_size = np.mean(h_sizes) if len(h_sizes) > 0 else 0.1
    duration   = max(timestamps[-1] - timestamps[0], 0.1)
    fs         = len(positions) / duration 
    
    # ── SİNYAL TEMİZLEME ──
    x = positions[:, 0] - np.mean(positions[:, 0])
    y = positions[:, 1] - np.mean(positions[:, 1])
    
    def bandpass_filter(data, lowcut, highcut, fs, order=3):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        if high >= 1.0: high = 0.95
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

    try:
        x_filt = bandpass_filter(x, 2.0, 15.0, fs)
        y_filt = bandpass_filter(y, 2.0, 15.0, fs)
    except:
        x_filt = x; y_filt = y

    # ── FFT (FREKANS ANALİZİ) ──
    N = len(x_filt)
    yf = fft(x_filt)
    xf = fftfreq(N, 1 / fs)
    pos_mask = (xf > 2) & (xf < 15)
    xf_pos = xf[pos_mask]
    yf_pos = np.abs(yf[pos_mask])
    
    frequency = float(xf_pos[np.argmax(yf_pos)]) if len(xf_pos) > 0 else 0.0

    # ── NORMALİZE AMPLİTÜD (BİLİMSEL NORMALİZASYON) ──
    # Piksellerin standart sapmasını el boyutuna (wrist-mcp) oranlıyoruz
    # Bu değişken, elin kameraya uzaklığından bağımsız hale gelir.
    std_x = np.std(x_filt)
    std_y = np.std(y_filt)
    raw_amplitude = float(std_x + std_y)
    
    # Normalizasyon: (Std Dev / Avg Hand Size) * 100 (Yüzdelik oran)
    norm_amplitude_pct = (raw_amplitude / (avg_h_size + 1e-6)) * 100
    
    # Parkinson Kriterleri: 3.5 - 7.5 Hz arası kritik
    is_parkinson_freq = 3.5 <= frequency <= 7.5
    
    # MDS-UPDRS Uyumlu Puanlama (Normalleştirilmiş Genlik Üzerinden)
    if norm_amplitude_pct < 1.0: score = 0
    elif norm_amplitude_pct < 3.5: score = 1
    elif norm_amplitude_pct < 10.0: score = 2
    elif norm_amplitude_pct < 20.0: score = 3
    else: score = 4
    
    # Seans Belleğine Kaydet
    m_key = "med" if tremor_state["medication"] else "none"
    session_data["tremor"][m_key] = {
        "freq": frequency,
        "amp": norm_amplitude_pct,
        "score": score
    }

    # Risk Puanı (Frekans ağırlıklı)
    risk_factor = 1.0
    if is_parkinson_freq:
        risk_factor = 1.8 if frequency < 6.0 else 1.4
        
    risk_score = min(100.0, score * 25.0 * risk_factor)
    
    severities = {
        4: ("şiddetli", "#f87171"),
        3: ("belirgin", "#f87171"),
        2: ("orta", "#fb923c"),
        1: ("hafif", "#fbbf24"),
        0: ("normal", "#4ade80")
    }
    sev, col = severities.get(score, ("normal", "#4ade80"))
    
    # Klinik Öneri
    if score >= 3 and is_parkinson_freq:
        rec = "Kritik frekans bandında (Parkinsonian) şiddetli titreme. Klinik takip şarttır."
    elif is_parkinson_freq and score >= 1:
        rec = "Parkinson ile uyumlu düşük frekanslı ritmik titreme aktivitesi saptandı."
    elif score >= 1:
        rec = f"Düşük amplitüdlü {frequency} Hz titreme saptandı (Fizyolojik olasılığı yüksek)."
    else:
        rec = "Titreme aktivitesi klinik sınırların altında."
        
    return {
        "amplitude": round(raw_amplitude, 1),
        "norm_amp_pct": round(norm_amplitude_pct, 1),
        "frequency": round(frequency, 2),
        "severity": sev,
        "risk": "Kritik" if risk_score > 75 else "Yüksek" if risk_score > 50 else "Düşük",
        "risk_score": round(risk_score, 1),
        "color": col,
        "recommendation": rec,
        "duration": round(duration, 1),
        "is_parkinson_freq": is_parkinson_freq,
        "updrs_score": score
    }


def get_landmarks(frame):
    if not MP_AVAILABLE: return None
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        res = detector.detect(mp_image)
        if res.hand_landmarks:
            return res.hand_landmarks[0]
    except:
        pass
    return None

def detect_hand(frame, landmarks=None):
    if landmarks:
        try:
            h, w = frame.shape[:2]
            itip = landmarks[8]
            cx, cy = int(itip.x * w), int(itip.y * h)
            cv2.circle(frame, (cx,cy), 10, (99,179,237), 2)
            return cx, cy
        except: pass
    
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

def get_finger_distance_from_lms(frame, landmarks):
    if not landmarks: return None
    try:
        # Landmarks
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        wrist     = landmarks[0]
        middle_mcp = landmarks[9]
        
        # Screen coords for visualization
        h, w = frame.shape[:2]
        t_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
        i_pos = (int(index_tip.x * w), int(index_tip.y * h))
        
        # Euclidean distance (normalized by hand size)
        hand_size = np.sqrt((wrist.x - middle_mcp.x)**2 + (wrist.y - middle_mcp.y)**2)
        raw_dist  = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
        norm_dist = raw_dist / (hand_size + 1e-6)
        
        # Draw
        cv2.line(frame, t_pos, i_pos, (74, 222, 128), 2)
        cv2.circle(frame, t_pos, 5, (99, 179, 237), -1)
        cv2.circle(frame, i_pos, 5, (99, 179, 237), -1)
        
        return norm_dist
    except:
        return None

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

        # Landmark Tespiti (Tasks API - Tek Geçiş)
        landmarks = get_landmarks(frame)
        
        cx, cy = detect_hand(frame, landmarks)
        hand_detected = cx is not None
        
        norm_dist = get_finger_distance_from_lms(frame, landmarks)
        pair_detected = norm_dist is not None

        with tremor_lock:
            is_tremor_active = tremor_state["active"]
        with tapping_lock:
            is_tapping_active = tapping_state["active"]

        # Tremor Kaydı
        if is_tremor_active and hand_detected:
            # Mesafe normalizasyonu için el boyutu hesapla (Wrist -> Middle MCP)
            h_size = 0.1
            if landmarks:
                try:
                    wrist = landmarks[0]
                    mcp = landmarks[9] # Middle MCP
                    h_size = np.sqrt((wrist.x - mcp.x)**2 + (wrist.y - mcp.y)**2)
                except: pass

            with tremor_lock:
                tremor_state["positions"].append([float(cx), float(cy)])
                tremor_state["timestamps"].append(time.time())
                tremor_state["hand_sizes"].append(float(h_size))
                
                if len(tremor_state["timestamps"]) > 1:
                    elapsed = tremor_state["timestamps"][-1] - tremor_state["timestamps"][0]
                    if elapsed >= 5.0:
                        res = analyze_tremor(
                            tremor_state["positions"], 
                            tremor_state["timestamps"],
                            tremor_state["hand_sizes"]
                        )
                        if res:
                            res["medication"] = tremor_state["medication"]
                        tremor_state["result"] = res
                        tremor_state["active"] = False
                        tremor_state["positions"] = []
                        tremor_state["timestamps"] = []
                        tremor_state["hand_sizes"] = []

        # Tapping (Bradikinezi) Kaydı
        if is_tapping_active and pair_detected:
            with tapping_lock:
                # EMA Filtresi (Jitter temizleme)
                prev = tapping_state["distances"][-1] if tapping_state["distances"] else norm_dist
                alpha = 0.4 # Yumuşatma katsayısı
                smooth_dist = alpha * norm_dist + (1 - alpha) * prev
                
                tapping_state["distances"].append(float(smooth_dist))
                tapping_state["timestamps"].append(time.time())
                tapping_state["current_dist"] = float(smooth_dist)
                
                if len(tapping_state["timestamps"]) > 1:
                    elapsed = tapping_state["timestamps"][-1] - tapping_state["timestamps"][0]
                    if elapsed >= 10.0: # Bradikinezi testi genellikle 10sn sürer
                        res = tapping_analyzer.analyze(tapping_state["distances"], tapping_state["timestamps"])
                        if res:
                            res["medication"] = tapping_state["medication"]
                        tapping_state["result"] = res
                        tapping_state["active"] = False
                        tapping_state["distances"] = []
                        tapping_state["timestamps"] = []

        # Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (w,36), (13,17,23), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        with tremor_lock:
            active  = tremor_state["active"]
            count   = len(tremor_state["timestamps"])
            elapsed = tremor_state["timestamps"][-1]-tremor_state["timestamps"][0] if count>1 else 0

        with tapping_lock:
            t_active = tapping_state["active"]
            t_count  = len(tapping_state["distances"])
            t_elapsed = tapping_state["timestamps"][-1]-tapping_state["timestamps"][0] if t_count>1 else 0

        if active and count > 1:
            remaining = max(0, 5.0 - elapsed)
            progress  = min(int((elapsed/5.0)*(w-16)), w-16)
            cv2.putText(frame, f"TITREME KAYIT: {remaining:.1f}s", (8,24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (74,222,128), 2)
            cv2.rectangle(frame, (8,28), (8+progress,33), (74,222,128), -1)
        elif t_active and t_count > 1:
            remaining = max(0, 10.0 - t_elapsed)
            progress  = min(int((t_elapsed/10.0)*(w-16)), w-16)
            cv2.putText(frame, f"VURUS KAYIT: {remaining:.1f}s", (8,24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (251,146,60), 2)
            cv2.rectangle(frame, (8,28), (8+progress,33), (251,146,60), -1)
        elif hand_detected or pair_detected:
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

@app.route("/camera/open", methods=["POST"])
def camera_open():
    print("[API] /camera/open isteği geldi", flush=True)
    success = camera.start()
    print(f"[API] /camera/open sonucu: {success}", flush=True)
    return jsonify({"status": "opened" if success else "failed"})


@app.route("/camera/close", methods=["POST"])
def camera_close():
    camera.stop()
    return jsonify({"status": "closed"})

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
        hs  = tremor_state["hand_sizes"].copy()
        med  = tremor_state["medication"]
    result = analyze_tremor(pos, ts, hs)
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

# ── TAPPING (BRADIKINEZI) ENDPOINTS ──
@app.route("/tapping/start", methods=["POST"])
def tapping_start():
    data = request.json or {}
    med_status = data.get("medication", False)
    with tapping_lock:
        tapping_state.update({
            "active": True,
            "distances": [],
            "timestamps": [],
            "result": None,
            "medication": med_status
        })
    return jsonify({"status":"started", "medication": med_status})

@app.route("/tapping/stop", methods=["POST"])
def tapping_stop():
    with tapping_lock:
        tapping_state["active"] = False
        dist = tapping_state["distances"].copy()
        ts  = tapping_state["timestamps"].copy()
    result = tapping_analyzer.analyze(dist, ts)
    with tapping_lock:
        tapping_state["result"] = result
    return jsonify({"status":"stopped","result":result})

@app.route("/tapping/result")
def tapping_result():
    with tapping_lock:
        active  = tapping_state["active"]
        result  = tapping_state["result"]
        med     = tapping_state["medication"]
        count   = len(tapping_state["distances"])
        elapsed = tapping_state["timestamps"][-1]-tapping_state["timestamps"][0] if count>1 else 0
        current_v = tapping_state["current_dist"]
    return jsonify({
        "active": active,
        "result": result,
        "medication": med,
        "count": count,
        "elapsed": round(elapsed, 1),
        "current_dist": current_v
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

@app.route("/report/clinical")
def get_clinical_report():
    t_data = session_data["tremor"]
    p_data = session_data["tapping"]
    
    report = {
        "comparisons": [],
        "overall_status": "Veri Bekleniyor",
        "timestamp": time.strftime("%H:%M:%S")
    }
    
    # Titreme Kıyaslaması
    if t_data["none"] and t_data["med"]:
        diff = t_data["none"]["score"] - t_data["med"]["score"]
        improvement = (diff / max(t_data["none"]["score"], 1)) * 100
        report["comparisons"].append({
            "type": "Titreme",
            "off_score": t_data["none"]["score"],
            "on_score": t_data["med"]["score"],
            "improvement": round(improvement, 1),
            "status": "Olumlu Yanıt" if improvement >= 25 else "Kısıtlı Yanıt"
        })

    # Vuruş Kıyaslaması
    if p_data["none"] and p_data["med"]:
        diff = p_data["none"]["score"] - p_data["med"]["score"]
        improvement = (diff / max(p_data["none"]["score"], 1)) * 100
        report["comparisons"].append({
            "type": "Bradikinezi",
            "off_score": p_data["none"]["score"],
            "on_score": p_data["med"]["score"],
            "improvement": round(improvement, 1),
            "status": "Olumlu Yanıt" if improvement >= 25 else "Kısıtlı Yanıt"
        })
    
    if report["comparisons"]:
        avg_imp = sum(c["improvement"] for c in report["comparisons"]) / len(report["comparisons"])
        if avg_imp >= 30: report["overall_status"] = "Optimal Tedavi Yanıtı"
        elif avg_imp >= 15: report["overall_status"] = "Kısmi Tedavi Yanıtı"
        else: report["overall_status"] = "Düşük Tedavi Yanıtı"
        
    return jsonify(report)

if __name__ == "__main__":
    print(f"\n  http://127.0.0.1:5000")
    print("="*50+"\n")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)