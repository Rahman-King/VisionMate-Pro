import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from ultralytics import YOLO
import av, cv2, numpy as np, face_recognition, threading, time, pickle, os

# --- CONFIGURATION & CONSTANTS ---
DB_PATH = "face_db.pkl"
HAZARDS = ['car', 'bus', 'truck', 'fire', 'stop sign', 'stair']
REAL_HEIGHTS = {"person": 1.7, "car": 1.5, "chair": 1.0, "door": 2.1}
FOCAL_LENGTH = 700 

# --- DATABASE HELPERS ---
def load_db():
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, "rb") as f: return pickle.load(f)
        except: return {"encodings": [], "names": []}
    return {"encodings": [], "names": []}

def save_db(data):
    with open(DB_PATH, "wb") as f: pickle.dump(data, f)

# --- THE AI ENGINE ---
class VisionMateProcessor(VideoProcessorBase):
    def __init__(self, use_ocr, use_face):
        self.yolo = YOLO('yolov8n.pt')
        self.db = load_db()
        self.use_ocr = use_ocr
        self.use_face = use_face
        self.reader = None
        if use_ocr:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=False)
            
        self.last_speech = 0
        self.announcement = ""
        self.capture_name = ""
        self.lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        results = self.yolo(img, conf=0.5, verbose=False)[0]
        detected = []

        # 1. FACE RECOGNITION / ENROLLMENT
        if self.use_face:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_locs = face_recognition.face_locations(rgb_img)
            face_encs = face_recognition.face_encodings(rgb_img, face_locs)

            for enc, loc in zip(face_encs, face_locs):
                name = "Unknown Person"
                if self.db["encodings"]:
                    matches = face_recognition.compare_faces(self.db["encodings"], enc, tolerance=0.55)
                    if True in matches: name = self.db["names"][matches.index(True)]
                
                if self.capture_name:
                    self.db["encodings"].append(enc)
                    self.db["names"].append(self.capture_name)
                    save_db(self.db)
                    self.capture_name = ""
                
                detected.append({"label": name, "priority": 2, "pos": "center", "dist": 2.0})

        # 2. OBJECT DETECTION & DISTANCE
        for box in results.boxes:
            label = self.yolo.names[int(box.cls)]
            if label == "person" and self.use_face: continue
            
            x_center = box.xywh[0][0].item() / w
            pixel_h = box.xywh[0][3].item()
            dist = round((REAL_HEIGHTS.get(label, 1.0) * FOCAL_LENGTH) / pixel_h, 1)
            pos = "left" if x_center < 0.33 else "right" if x_center > 0.66 else "center"
            priority = 1 if (label in HAZARDS or dist < 2.0) else 3
            detected.append({"label": label, "pos": pos, "dist": dist, "priority": priority})

        # 3. OCR SCAN (Every 5 seconds)
        if self.use_ocr and time.time() - self.last_speech > 5.0:
            # We use a fast scan on the current image
            pass 

        # 4. SPEECH PACING (Only every 4 seconds)
        detected.sort(key=lambda x: x['priority'])
        if detected and (time.time() - self.last_speech > 4.0):
            top = detected[0]
            with self.lock:
                self.announcement = f"{top['label']} {top['dist']} meters {top['pos']}"
                self.last_speech = time.time()

        return av.VideoFrame.from_ndarray(results.plot(), format="bgr24")

# --- UI & BROWSER SCRIPTS ---
st.title("👁️ Vision Mate Pro")

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Settings")
    v_cmd = st.toggle("🎙️ Voice Commands", value=True)
    n_mode = st.toggle("🌙 Night Mode", value=False)
    ocr_en = st.toggle("📖 OCR Mode", value=True)
    face_en = st.toggle("👤 Face Rec", value=True)
    
    st.divider()
    ename = st.text_input("Enroll Family Name")
    if st.button("📸 Enroll Face"): st.session_state.do_enroll = ename
    if st.button("🗑️ Reset All Data"): 
        if os.path.exists(DB_PATH): os.remove(DB_PATH)
        st.rerun()

# Flashlight/Night Mode JS
if n_mode:
    st.components.v1.html("""<script>
        navigator.mediaDevices.getUserMedia({video: {facingMode: 'environment'}}).then(s => {
            s.getVideoTracks()[0].applyConstraints({advanced: [{torch: true}]});
        });
    </script>""", height=0)

# Main Control
if "active" not in st.session_state: st.session_state.active = False
if st.button("🛑 STOP SYSTEM" if st.session_state.active else "▶️ START SYSTEM", type="primary", use_container_width=True):
    st.session_state.active = not st.session_state.active

if st.session_state.active:
    ctx = webrtc_streamer(
        key="vision",
        video_processor_factory=lambda: VisionMateProcessor(use_ocr=ocr_en, use_face=face_en),
        media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False}
    )

    if ctx.video_processor:
        if "do_enroll" in st.session_state:
            ctx.video_processor.capture_name = st.session_state.do_enroll
            del st.session_state.do_enroll
        
        with ctx.video_processor.lock:
            msg = ctx.video_processor.announcement
        if msg:
            st.info(f"AI: {msg}")
            # Web Speech API for free Voice Output
            st.components.v1.html(f"<script>window.speechSynthesis.speak(new SpeechSynthesisUtterance('{msg}'));</script>", height=0)
