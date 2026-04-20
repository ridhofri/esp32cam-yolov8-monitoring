# BrewWatch ☕

**Smart Staff Monitoring System — AI-Powered Sitting Detection with Instant Alert**

> Adaptasi dari [Binivert/security-system](https://github.com/Binivert/security-system)  
> Modifikasi: Web GUI (Flask), YOLOv8-Pose, Sitting Detection Timer, tanpa PyQt6

---

## 📁 Struktur Folder

```
brewwatch/
├── app.py              ← Entry point Flask web server
├── camera.py           ← Thread kamera + deteksi + MJPEG stream
├── detector.py         ← YOLOv8-Pose + Sitting Detection (adaptasi dari detectors.py)
├── telegram_bot.py     ← Telegram Bot async (adaptasi dari telegram_bot.py)
├── config.py           ← Semua konfigurasi di sini
├── requirements.txt
├── snapshots/          ← Foto alert tersimpan di sini
└── templates/
    └── index.html      ← Web GUI dashboard
```

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Model `yolov8n-pose.pt` (~6MB) otomatis diunduh saat pertama kali dijalankan.

### 2. Konfigurasi Telegram Bot

Edit `config.py`:

```python
TELEGRAM_BOT_TOKEN = "token_dari_BotFather"
TELEGRAM_CHAT_ID   = "chat_id_kamu"
```

**Cara dapat token & chat ID:**

1. Buka Telegram → cari `@BotFather` → `/newbot`
2. Salin token
3. Buka `@userinfobot` → salin ID kamu

### 3. Jalankan

```bash
python app.py
```

Buka browser: **http://localhost:5000**

---

## 🔧 Konfigurasi (config.py)

| Parameter               | Default           | Keterangan                         |
| ----------------------- | ----------------- | ---------------------------------- |
| `CAMERA_SOURCE`         | `0`               | 0 = webcam laptop, URL = ESP32-CAM |
| `SITTING_THRESHOLD_SEC` | `10`              | Detik duduk sebelum alert          |
| `ALERT_COOLDOWN_SEC`    | `30`              | Jeda minimal antar alert per orang |
| `YOLO_MODEL`            | `yolov8n-pose.pt` | Model YOLOv8 pose                  |
| `YOLO_CONFIDENCE`       | `0.4`             | Confidence threshold               |

### Ganti ke ESP32-CAM (nanti)

```python
CAMERA_SOURCE = "http://192.168.1.100:81/stream"
```

---

## 🦴 Logika Sitting Detection

Seseorang dianggap **duduk** jika:

- Lutut dan pinggul hampir sejajar secara vertikal (selisih Y kecil)
- Bahu masih berada di atas pinggul (tidak rebahan)

Jika duduk > `SITTING_THRESHOLD_SEC` detik → snapshot diambil → Telegram alert dikirim.

---

## 📱 Perintah Telegram Bot

| Perintah  | Fungsi             |
| --------- | ------------------ |
| `/start`  | Tampilkan menu     |
| `/status` | Status sistem live |
| `/snap`   | Minta screenshot   |

---

## 🌐 Web GUI Features

- Live MJPEG stream dengan skeleton overlay
- Dashboard stats: FPS, jumlah orang, duduk, total alert
- Timer bar per orang (progress menuju threshold)
- Log alert real-time
- Tombol snapshot manual
