"""
BrewWatch - Konfigurasi Utama
Adaptasi dari: github.com/Binivert/security-system
"""

from pathlib import Path
import os

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# ─── TELEGRAM ────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("BOT_TOKEN")   # dari @BotFather
TELEGRAM_CHAT_ID   = os.getenv("CHAT_ID")     # chat ID kamu

# ─── KAMERA ──────────────────────────────────────────────
CAMERA_SOURCE = 0          # 0 = webcam laptop
                           # "http://192.168.x.x:81/stream" = ESP32-CAM

# ─── YOLO ────────────────────────────────────────────────
YOLO_MODEL      = "yolov8n-pose.pt"   # auto-download pertama kali
YOLO_CONFIDENCE = 0.4

# ─── SITTING DETECTION ───────────────────────────────────
SITTING_THRESHOLD_SEC  = 10    # detik duduk sebelum alert dikirim
ALERT_COOLDOWN_SEC     = 30    # jeda minimal antar alert (per orang)

# ─── WEB SERVER ──────────────────────────────────────────
WEB_HOST = "0.0.0.0"
WEB_PORT = 8080

# ─── FOLDER ──────────────────────────────────────────────
SNAPSHOTS_DIR = BASE_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(exist_ok=True)

# ─── MEDIAPIPE KEYPOINT INDEX ────────────────────────────
# (33 landmark MediaPipe / 17 COCO YOLOv8-pose)
# Untuk YOLOv8-pose (17 keypoints COCO):
KP_NOSE          = 0
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER= 6
KP_LEFT_HIP      = 11
KP_RIGHT_HIP     = 12
KP_LEFT_KNEE     = 13
KP_RIGHT_KNEE    = 14
KP_LEFT_ANKLE    = 15
KP_RIGHT_ANKLE   = 16

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
