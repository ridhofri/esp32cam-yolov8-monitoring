"""
BrewWatch - Camera & Detection Thread
Menghasilkan MJPEG stream untuk web GUI Flask.
"""

import cv2
import time
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from detector import PersonDetector, PersonDetection
from config import (
    CAMERA_SOURCE, SITTING_THRESHOLD_SEC,
    ALERT_COOLDOWN_SEC, SNAPSHOTS_DIR
)


class CameraStream:
    """
    Thread kamera yang:
    - Baca frame dari webcam/ESP32-CAM
    - Jalankan YOLOv8-pose detection
    - Cek sitting > threshold → trigger alert
    - Generate MJPEG untuk streaming ke browser
    """

    def __init__(self, telegram_bot=None):
        self.alert_log   = []
        self.telegram    = telegram_bot
        self.detector    = PersonDetector()
        self._cap        = None
        self._running    = False
        self._lock       = threading.Lock()
        self._frame_raw  = None      # frame mentah (untuk snapshot)
        self._frame_out  = None      # frame dengan anotasi (untuk stream)
        self._thread     = None

        # Status untuk web GUI
        self.fps          = 0
        self.person_count = 0
        self.sitting_info: List[dict] = []
        self.total_alerts = 0
        self._start_time  = time.time()

        # Track alert yang sudah dikirim
        self._alerted: dict = {}   # {track_id: last_alert_time}

    # ── PUBLIC ────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[Camera] Stream dimulai dari source: {CAMERA_SOURCE}")

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()

    def get_jpeg_frame(self) -> Optional[bytes]:
        """Ambil frame terbaru sebagai JPEG bytes untuk MJPEG stream."""
        with self._lock:
            frame = self._frame_out
        if frame is None:
            return None
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes()

    def take_snapshot(self, label="snap") -> str:
        """Simpan snapshot dan return path-nya."""
        with self._lock:
            frame = self._frame_out.copy() if self._frame_out is not None else None
        if frame is None:
            return ""
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = str(SNAPSHOTS_DIR / f"{label}_{ts}.jpg")
        cv2.imwrite(path, frame)
        print(f"[Camera] Snapshot: {path}")
        return path

    def get_status(self) -> dict:
        uptime = int(time.time() - self._start_time)
        return {
            "running":      self._running,
            "fps":          round(self.fps, 1),
            "person_count": self.person_count,
            "sitting_info": self.sitting_info,
            "total_alerts": self.total_alerts,
            "uptime":       uptime,
            "alert_log":    self.alert_log,   # ← pastikan ada ini
        }

    # ── INTERNAL LOOP ─────────────────────────────────────────────

    def _loop(self):
        self._cap = cv2.VideoCapture(CAMERA_SOURCE)
        if not self._cap.isOpened():
            print(f"[Camera] ❌ Tidak bisa buka kamera: {CAMERA_SOURCE}")
            self._running = False
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        fps_time  = time.time()
        fps_count = 0

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                print("[Camera] Frame gagal, retry...")
                time.sleep(0.5)
                self._cap = cv2.VideoCapture(CAMERA_SOURCE)
                continue

            # Deteksi
            persons, annotated = self.detector.detect(frame)

            # HUD overlay
            annotated = self._draw_hud(annotated, persons)

            # Update state
            self.person_count = len(persons)
            self.sitting_info = [
                {
                    "id":       p.track_id,
                    "sitting":  p.is_sitting,
                    "duration": round(p.sitting_duration, 1),
                    "alert":    p.is_sitting and p.sitting_duration >= SITTING_THRESHOLD_SEC,
                }
                for p in persons
            ]

            # Cek sitting alert
            for p in persons:
                self._check_alert(p, annotated)

            # FPS
            fps_count += 1
            now = time.time()
            if now - fps_time >= 1.0:
                self.fps   = fps_count / (now - fps_time)
                fps_count  = 0
                fps_time   = now

            # Simpan frame untuk stream
            with self._lock:
                self._frame_out = annotated
                self._frame_raw = frame

    def _check_alert(self, person: PersonDetection, frame: np.ndarray):
        """Kirim alert Telegram jika orang duduk melebihi threshold."""
        if not person.is_sitting:
            self._alerted.pop(person.track_id, None)
            return

        if person.sitting_duration < SITTING_THRESHOLD_SEC:
            return

        now  = time.time()
        last = self._alerted.get(person.track_id, 0)
        if now - last < ALERT_COOLDOWN_SEC:
            return

        self._alerted[person.track_id] = now
        self.total_alerts += 1

        ts = datetime.now().strftime("%H:%M:%S")   # ← pastikan persis seperti ini
        self.alert_log.insert(0, {
            "id":       person.track_id + 1,
            "duration": int(person.sitting_duration),
            "time":     ts                          # ← "time" bukan "timestamp" atau lainnya
        })
        if len(self.alert_log) > 20:
            self.alert_log.pop()

        # Simpan snapshot
        snapshot = self.take_snapshot(f"sitting_p{person.track_id}")

        # Kirim Telegram
        if self.telegram:
            self.telegram.send_sitting_alert(
                track_id=person.track_id,
                duration=int(person.sitting_duration),
                snapshot_path=snapshot
            )
        print(f"[Alert] ⚠ Person {person.track_id} duduk {int(person.sitting_duration)}s → alert dikirim")

    def _draw_hud(self, frame: np.ndarray, persons: List[PersonDetection]) -> np.ndarray:
        """Overlay info di layar — diadaptasi dari gui.py Binivert."""
        h, w = frame.shape[:2]

        # Panel atas semi-transparan
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 52), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (100, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

        # Jumlah orang
        cv2.putText(frame, f"Orang: {len(persons)}", (220, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv2.LINE_AA)

        # Sitting count
        sitting_n = sum(1 for p in persons if p.is_sitting)
        cv2.putText(frame, f"Duduk: {sitting_n}", (340, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2, cv2.LINE_AA)

        # Model label
        cv2.putText(frame, "BrewWatch | YOLOv8-Pose", (w - 150, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 180, 180), 1, cv2.LINE_AA)

        # Alert bar bawah jika ada yang duduk terlalu lama
        alert_persons = [p for p in persons if p.is_sitting and p.sitting_duration >= SITTING_THRESHOLD_SEC]
        if alert_persons:
            cv2.rectangle(frame, (0, h - 44), (w, h), (0, 0, 180), -1)
            names = ", ".join([f"Orang #{p.track_id+1} ({int(p.sitting_duration)}s)" for p in alert_persons])
            cv2.putText(frame, f"⚠ ALERT: {names}", (12, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return frame


def generate_mjpeg(camera: CameraStream):
    """Generator MJPEG untuk Flask response streaming."""
    while True:
        frame_bytes = camera.get_jpeg_frame()
        if frame_bytes is None:
            time.sleep(0.05)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame_bytes +
            b"\r\n"
        )
        time.sleep(0.033)   # ~30 fps max
