"""
BrewWatch - Telegram Bot
Diadaptasi dari: github.com/Binivert/security-system/telegram_bot.py
Modifikasi: hapus PyQt6 dependency, tambah sitting alert
"""

import requests
import time
import threading
from queue import Queue, Empty
from pathlib import Path
from typing import Optional
from datetime import datetime
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


class TelegramBot(threading.Thread):
    """
    Telegram bot async — kirim pesan & foto tanpa blocking main thread.
    Diadaptasi dari TelegramBot Binivert, refactor ke threading.Thread biasa.
    """

    def __init__(self, on_command=None):
        super().__init__(daemon=True)
        self.token    = TELEGRAM_BOT_TOKEN
        self.chat_id  = TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.token}"

        self._queue          = Queue()
        self._running        = False
        self._last_update_id = 0
        self._on_command     = on_command  # callback(cmd: str)

        # Cooldown per track_id agar tidak spam
        self._last_alert: dict = {}

    # ── PUBLIC API ────────────────────────────────────────────────

    def send_message(self, text: str):
        """Kirim pesan teks (async, tidak blocking)."""
        self._queue.put(("text", {"text": text}))

    def send_photo(self, path: str, caption: str = ""):
        """Kirim foto dengan caption (async)."""
        self._queue.put(("photo", {"path": path, "caption": caption}))

    def send_sitting_alert(self, track_id: int, duration: int, snapshot_path: str):
        """
        Kirim alert duduk dengan cooldown per orang.
        Diadaptasi dari send_alert_with_photo Binivert.
        """
        now  = time.time()
        last = self._last_alert.get(track_id, 0)
        from config import ALERT_COOLDOWN_SEC
        if now - last < ALERT_COOLDOWN_SEC:
            return  # masih dalam cooldown

        self._last_alert[track_id] = now
        ts      = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        caption = (
            f"⚠️ *DUDUK TERLALU LAMA!*\n"
            f"👤 Orang #{track_id + 1}\n"
            f"⏱ Duduk selama: *{duration} detik*\n"
            f"📅 {ts}\n"
            f"📍 BrewWatch — Coffeeshop Monitor"
        )
        self._queue.put(("photo", {"path": snapshot_path, "caption": caption}))

    def can_alert(self, track_id: int) -> bool:
        from config import ALERT_COOLDOWN_SEC
        return time.time() - self._last_alert.get(track_id, 0) >= ALERT_COOLDOWN_SEC

    # ── THREAD LOOP ───────────────────────────────────────────────

    def run(self):
        self._running = True
        # Thread polling update dari Telegram
        poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        poll_thread.start()

        # Loop kirim pesan dari queue
        while self._running:
            try:
                item = self._queue.get(timeout=1)
                if item is None:
                    break
                msg_type, data = item
                if msg_type == "text":
                    self._send_text(data["text"])
                elif msg_type == "photo":
                    self._send_photo(data["path"], data.get("caption", ""))
            except Empty:
                continue
            except Exception as e:
                print(f"[Telegram] Queue error: {e}")

    def stop(self):
        self._running = False
        self._queue.put(None)

    # ── INTERNAL SEND ─────────────────────────────────────────────

    def _send_text(self, text: str):
        try:
            resp = requests.post(
                f"{self.base_url}/sendMessage",
                json={"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"},
                timeout=15
            )
            if resp.status_code == 200:
                print(f"[Telegram] ✅ Pesan terkirim")
            else:
                print(f"[Telegram] ❌ Gagal: {resp.status_code} - {resp.text[:100]}")
        except Exception as e:
            print(f"[Telegram] Error send_text: {e}")

    def _send_photo(self, path: str, caption: str):
        if not Path(path).exists():
            print(f"[Telegram] File tidak ada: {path}")
            self._send_text(caption + "\n\n_(Foto tidak tersedia)_")
            return

        for attempt in range(3):
            try:
                with open(path, "rb") as f:
                    resp = requests.post(
                        f"{self.base_url}/sendPhoto",
                        data={"chat_id": self.chat_id, "caption": caption, "parse_mode": "Markdown"},
                        files={"photo": f},
                        timeout=30
                    )
                if resp.status_code == 200:
                    print(f"[Telegram] ✅ Foto terkirim")
                    return
                else:
                    print(f"[Telegram] Attempt {attempt+1} gagal: {resp.status_code}")
            except Exception as e:
                print(f"[Telegram] Attempt {attempt+1} error: {e}")
            time.sleep(1)

    # ── POLLING UPDATE ────────────────────────────────────────────

    def _poll_loop(self):
        """Poll update perintah dari Telegram (diadaptasi dari Binivert)."""
        while self._running:
            try:
                resp = requests.get(
                    f"{self.base_url}/getUpdates",
                    params={"offset": self._last_update_id + 1, "timeout": 20},
                    timeout=25
                )
                if resp.status_code != 200:
                    time.sleep(3)
                    continue

                data = resp.json()
                for update in data.get("result", []):
                    self._last_update_id = update["update_id"]
                    self._handle_update(update)

            except requests.exceptions.Timeout:
                pass
            except Exception as e:
                print(f"[Telegram] Poll error: {e}")
                time.sleep(5)

    def _handle_update(self, update: dict):
        """Handle perintah dari Telegram."""
        msg  = update.get("message", {})
        text = msg.get("text", "").strip().lower()
        if not text:
            return

        print(f"[Telegram] Perintah masuk: {text}")
        if self._on_command:
            self._on_command(text)

        # Balas status dasar
        if text in ["/start", "/status", "/help"]:
            self._send_text(
                "🟢 *BrewWatch aktif!*\n\n"
                "Perintah tersedia:\n"
                "/status — Status sistem\n"
                "/snap — Minta snapshot\n"
                "/stop — Hentikan monitoring"
            )
        elif text == "/snap":
            if self._on_command:
                self._on_command("snap")
