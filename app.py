"""
BrewWatch - Flask Web App
Web GUI untuk monitoring skeleton tracking karyawan coffeeshop.
"""

from flask import Flask, Response, render_template, jsonify, request
from camera import CameraStream, generate_mjpeg
from telegram_bot import TelegramBot
from config import WEB_HOST, WEB_PORT, SITTING_THRESHOLD_SEC, ALERT_COOLDOWN_SEC
import threading

app = Flask(__name__)

# Inisialisasi komponen
telegram = TelegramBot()
camera   = CameraStream(telegram_bot=telegram)


def handle_telegram_command(cmd: str):
    """Handle perintah dari Telegram."""
    if cmd == "snap":
        path = camera.take_snapshot("telegram_snap")
        if path:
            telegram.send_photo(path, "📸 *Snapshot diminta via Telegram*")
        else:
            telegram.send_message("❌ Kamera tidak aktif.")
    elif cmd == "/status" or cmd == "status":
        status = camera.get_status()
        telegram.send_message(
            f"📊 *Status BrewWatch*\n"
            f"▶ Running: {'Ya' if status['running'] else 'Tidak'}\n"
            f"📷 FPS: {status['fps']}\n"
            f"👤 Orang terdeteksi: {status['person_count']}\n"
            f"⚠ Total alert: {status['total_alerts']}\n"
            f"⏱ Uptime: {status['uptime']}s"
        )


telegram._on_command = handle_telegram_command


# ── ROUTES ────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           sitting_threshold=SITTING_THRESHOLD_SEC,
                           alert_cooldown=ALERT_COOLDOWN_SEC)


@app.route("/video_feed")
def video_feed():
    """MJPEG stream endpoint."""
    return Response(
        generate_mjpeg(camera),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/status")
def api_status():
    """JSON status untuk update dashboard real-time."""
    return jsonify(camera.get_status())


@app.route("/api/snapshot", methods=["POST"])
def api_snapshot():
    """Ambil snapshot manual."""
    path = camera.take_snapshot("manual")
    if path:
        return jsonify({"success": True, "path": path})
    return jsonify({"success": False, "message": "Kamera tidak aktif"}), 500


@app.route("/api/config", methods=["GET"])
def api_config():
    return jsonify({
        "sitting_threshold": SITTING_THRESHOLD_SEC,
        "alert_cooldown": ALERT_COOLDOWN_SEC,
        "camera_source": str(camera.detector._loaded),
        "model": "yolov8n-pose.pt"
    })


# ── MAIN ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  BrewWatch — Staff Monitoring System")
    print("=" * 50)

    # Start Telegram bot
    telegram.start()
    print("[Telegram] Bot dimulai.")

    # Start kamera & deteksi
    camera.start()
    print(f"[Web] Buka browser: http://localhost:{WEB_PORT}")
    print("[Info] Tekan Ctrl+C untuk berhenti.\n")

    telegram.send_message(
        "🟢 *BrewWatch aktif!*\n"
        f"⏱ Threshold duduk: {SITTING_THRESHOLD_SEC} detik\n"
        "Ketik /status untuk cek kondisi sistem."
    )

    # Jalankan Flask
    app.run(host=WEB_HOST, port=WEB_PORT, debug=False, threaded=True)
