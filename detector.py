"""
BrewWatch - Detector Module
Diadaptasi dari: github.com/Binivert/security-system/detectors.py
"""

import cv2
import numpy as np
import time
import threading
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from config import (
    YOLO_MODEL, YOLO_CONFIDENCE, SKELETON_CONNECTIONS,
    KP_LEFT_HIP, KP_RIGHT_HIP, KP_LEFT_KNEE, KP_RIGHT_KNEE,
    KP_LEFT_ANKLE, KP_RIGHT_ANKLE, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER
)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[Detector] WARNING: ultralytics tidak terinstall. pip install ultralytics")


@dataclass
class PersonDetection:
    track_id: int
    bbox: Tuple[int, int, int, int]      # x1, y1, x2, y2
    confidence: float
    keypoints: np.ndarray                # shape (17, 2)
    kp_conf: np.ndarray                  # shape (17,)
    is_sitting: bool = False
    sitting_since: Optional[float] = None
    sitting_duration: float = 0.0


class SittingDetector:
    """
    Deteksi posisi duduk dari keypoints YOLOv8-pose (17 titik COCO).
    Logika diadaptasi dari Binivert/security-system zone detection.

    Duduk = pinggul dan lutut sejajar secara vertikal (selisih Y kecil),
            sementara bahu jauh di atas pinggul (orang tegak).
    """

    def is_sitting(self, kp: np.ndarray, kp_conf: np.ndarray, threshold: float = 0.4) -> bool:
        """
        kp      : array (17, 2) koordinat [x, y] normalized atau pixel
        kp_conf : array (17,) confidence tiap keypoint
        """
        # Pastikan keypoint yang diperlukan cukup confident
        required = [KP_LEFT_HIP, KP_RIGHT_HIP, KP_LEFT_KNEE, KP_RIGHT_KNEE]
        for idx in required:
            if kp_conf[idx] < threshold:
                return False

        l_hip   = kp[KP_LEFT_HIP]
        r_hip   = kp[KP_RIGHT_HIP]
        l_knee  = kp[KP_LEFT_KNEE]
        r_knee  = kp[KP_RIGHT_KNEE]

        avg_hip_y  = (l_hip[1]  + r_hip[1])  / 2
        avg_knee_y = (l_knee[1] + r_knee[1]) / 2

        # Duduk: lutut dan pinggul hampir sejajar (Y dekat)
        # Berdiri: lutut jauh di bawah pinggul
        hip_knee_diff = abs(avg_knee_y - avg_hip_y)

        # Cek juga bahu — kalau bahu confidence ok, pastikan orang tegak
        shoulder_ok = (kp_conf[KP_LEFT_SHOULDER] > threshold and
                       kp_conf[KP_RIGHT_SHOULDER] > threshold)
        if shoulder_ok:
            avg_shoulder_y = (kp[KP_LEFT_SHOULDER][1] + kp[KP_RIGHT_SHOULDER][1]) / 2
            shoulder_hip_diff = avg_hip_y - avg_shoulder_y  # positif = bahu di atas pinggul

            # Duduk: hip_knee_diff kecil (lutut sejajar pinggul)
            #        shoulder_hip_diff positif (bahu masih di atas = tidak rebahan)
            frame_height_estimate = max(kp[:, 1]) - min(kp[kp[:, 1] > 0][:, 1]) if np.any(kp[:, 1] > 0) else 480
            norm_hip_knee = hip_knee_diff / (frame_height_estimate + 1e-5)

            return norm_hip_knee < 0.15 and shoulder_hip_diff > 10
        else:
            # Fallback: cek ratio hip-knee saja
            frame_h = 480
            return (hip_knee_diff / frame_h) < 0.12


class PersonDetector:
    """
    Detector utama — YOLOv8-pose untuk deteksi orang + skeleton.
    Diadaptasi dari PersonDetector di Binivert/security-system.
    """

    SKELETON_COLORS = {
        "head":  (0, 255, 255),   # Cyan
        "torso": (0, 255, 0),     # Hijau
        "arm":   (0, 165, 255),   # Oranye
        "leg":   (255, 0, 255),   # Magenta
    }

    CONN_PART = [
        "head", "head", "head", "head",
        "torso",
        "arm", "arm", "arm", "arm",
        "torso", "torso", "torso",
        "leg", "leg", "leg", "leg",
    ]

    def __init__(self):
        self.model = None
        self.sitting_detector = SittingDetector()
        self._lock = threading.Lock()
        self._sitting_timers = {}   # {track_id: timestamp_mulai_duduk}
        self._loaded = False

        if YOLO_AVAILABLE:
            self._load_model()

    def _load_model(self):
        try:
            print(f"[Detector] Memuat model {YOLO_MODEL}...")
            self.model = YOLO(YOLO_MODEL)
            self._loaded = True
            print(f"[Detector] ✅ Model siap.")
        except Exception as e:
            print(f"[Detector] ❌ Gagal load model: {e}")

    def detect(self, frame: np.ndarray) -> Tuple[List[PersonDetection], np.ndarray]:
        """
        Deteksi orang + skeleton + status duduk dari satu frame.
        Return: (list PersonDetection, frame dengan anotasi)
        """
        if frame is None or not self._loaded:
            return [], frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

        output = frame.copy()
        persons = []

        try:
            with self._lock:
                results = self.model(frame, conf=YOLO_CONFIDENCE, verbose=False)

            for result in results:
                if result.keypoints is None or result.boxes is None:
                    continue

                kp_all   = result.keypoints.xy.cpu().numpy()      # (N, 17, 2)
                kp_conf  = result.keypoints.conf.cpu().numpy()    # (N, 17)
                boxes    = result.boxes.xyxy.cpu().numpy()         # (N, 4)
                confs    = result.boxes.conf.cpu().numpy()         # (N,)

                for i in range(len(kp_all)):
                    kp   = kp_all[i]
                    kpc  = kp_conf[i]
                    bbox = tuple(map(int, boxes[i]))
                    conf = float(confs[i])

                    # Sitting detection
                    sitting = self.sitting_detector.is_sitting(kp, kpc)

                    # Timer duduk
                    tid = i  # track_id sederhana (index)
                    now = time.time()
                    sitting_since = None
                    sitting_duration = 0.0

                    if sitting:
                        if tid not in self._sitting_timers:
                            self._sitting_timers[tid] = now
                        sitting_since    = self._sitting_timers[tid]
                        sitting_duration = now - sitting_since
                    else:
                        self._sitting_timers.pop(tid, None)

                    person = PersonDetection(
                        track_id=tid,
                        bbox=bbox,
                        confidence=conf,
                        keypoints=kp,
                        kp_conf=kpc,
                        is_sitting=sitting,
                        sitting_since=sitting_since,
                        sitting_duration=sitting_duration,
                    )
                    persons.append(person)

                    # Gambar anotasi
                    self._draw_person(output, person)

        except Exception as e:
            print(f"[Detector] Error deteksi: {e}")

        return persons, output

    def _draw_person(self, frame: np.ndarray, person: PersonDetection):
        x1, y1, x2, y2 = person.bbox
        kp  = person.keypoints
        kpc = person.kp_conf

        # Warna bbox: merah = duduk, hijau = berdiri
        box_color = (0, 0, 255) if person.is_sitting else (0, 255, 0)

        # Bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # Label status
        if person.is_sitting:
            dur  = int(person.sitting_duration)
            label = f"DUDUK {dur}s"
        else:
            label = f"Berdiri"

        conf_label = f"{label} ({person.confidence:.0%})"
        (tw, th), _ = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), box_color, -1)
        cv2.putText(frame, conf_label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Skeleton connections
        for idx, (i, j) in enumerate(SKELETON_CONNECTIONS):
            if i >= len(kp) or j >= len(kp):
                continue
            xi, yi = int(kp[i][0]), int(kp[i][1])
            xj, yj = int(kp[j][0]), int(kp[j][1])
            ci, cj = kpc[i], kpc[j]
            if ci > 0.4 and cj > 0.4 and xi > 0 and yi > 0 and xj > 0 and yj > 0:
                part  = self.CONN_PART[idx] if idx < len(self.CONN_PART) else "torso"
                color = self.SKELETON_COLORS[part]
                # Glow effect (diadaptasi dari Binivert)
                cv2.line(frame, (xi, yi), (xj, yj),
                         tuple(c // 3 for c in color), 5)
                cv2.line(frame, (xi, yi), (xj, yj), color, 2)

        # Keypoints
        for k in range(len(kp)):
            x, y = int(kp[k][0]), int(kp[k][1])
            if kpc[k] > 0.4 and x > 0 and y > 0:
                cv2.circle(frame, (x, y), 5, (0, 0, 0), -1)
                cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)

    def reset_timers(self):
        self._sitting_timers.clear()
