"""
Real-Time Facial Expression Detection & Emotion Recognition
Pipeline 2-Tahap: OpenCV Face Detection + YOLO11 Classification
Penggunaan:
    python webcam_demo.py [path_ke_best.pt]
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Path model klasifikasi emosi terbaik
DEFAULT_WEIGHTS = Path("facial_emotion_cls") / "yolo11s_emotion" / "weights" / "best.pt"

# Warna representatif untuk masing-masing emosi (BGR)
EMOTION_COLORS = {
    "happy": (0, 220, 0),      # Hijau cerah
    "neutral": (220, 200, 0),   # Kuning / Cyan
    "anger": (0, 0, 230),      # Merah
    "sad": (200, 100, 50),     # Biru redup
    "fear": (180, 50, 180)     # Ungu
}

def get_model_path():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    if DEFAULT_WEIGHTS.exists():
        return DEFAULT_WEIGHTS
    
    # Cari fallback model klasifikasi di folder facial_emotion_cls
    candidates = list(Path("facial_emotion_cls").glob("**/weights/best.pt"))
    if candidates:
        return candidates[0]
    return DEFAULT_WEIGHTS

def main():
    weights_path = get_model_path()
    if not weights_path.exists():
        print(f"[ERROR] File bobot model belum ditemukan di: {weights_path}")
        print("[INFO] Jalankan pelatihan klasifikasi pada notebook 'facial_expression_yolo.ipynb' terlebih dahulu.")
        return

    print(f"[INFO] Memuat model klasifikasi: {weights_path}")
    model = YOLO(str(weights_path))

    # Inisialisasi Detektor Wajah OpenCV (Ultra-cepat, 100+ FPS)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak dapat diakses. Pastikan kamera tidak digunakan aplikasi lain.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("-" * 60)
    print("DETEKSI EKSPRESI WAJAH REAL-TIME (Face Crop + YOLO SOTA)")
    print("Tekan tombol 'q' pada jendela video untuk keluar.")
    print("-" * 60)

    all_emotions = ["happy", "neutral", "anger", "fear", "sad"]

    # Tracking & Temporal Smoothing State (Anti-Flicker)
    prev_box = None
    missed_frames = 0
    MAX_MISSED_FRAMES = 8   # Tahan kotak selama 8 frame jika wajah terlewat sesaat
    SMOOTH_ALPHA = 0.65     # Faktor kehalusan pergerakan bounding box
    smoothed_probs = None   # Smoothing probabilitas emosi

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Gagal membaca frame kamera.")
            break

        frame_h, frame_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Tahap Deteksi Wajah (Sensitivitas Dioptimalkan)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,      # Pencarian lebih teliti
            minNeighbors=4,       # Lebih sensitif terhadap kepala miring/ekspresi
            minSize=(80, 80)
        )

        active_box = None

        if len(faces) > 0:
            # Pilih wajah terbesar (wajah utama yang menghadap kamera)
            faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
            cur_x, cur_y, cur_w, cur_h = faces[0]

            if prev_box is not None:
                # Exponential Moving Average Smoothing (Kotak bergerak mulus tanpa jitter)
                smooth_x = int(SMOOTH_ALPHA * prev_box[0] + (1 - SMOOTH_ALPHA) * cur_x)
                smooth_y = int(SMOOTH_ALPHA * prev_box[1] + (1 - SMOOTH_ALPHA) * cur_y)
                smooth_w = int(SMOOTH_ALPHA * prev_box[2] + (1 - SMOOTH_ALPHA) * cur_w)
                smooth_h = int(SMOOTH_ALPHA * prev_box[3] + (1 - SMOOTH_ALPHA) * cur_h)
                active_box = (smooth_x, smooth_y, smooth_w, smooth_h)
            else:
                active_box = (cur_x, cur_y, cur_w, cur_h)

            prev_box = active_box
            missed_frames = 0
        else:
            # Jika wajah sempat terlewat 1-8 frame (kedipan mata/gerak cepat), pertahankan posisi sebelumnya (Anti-Hilang-Muncul)
            if prev_box is not None and missed_frames < MAX_MISSED_FRAMES:
                active_box = prev_box
                missed_frames += 1
            else:
                prev_box = None
                smoothed_probs = None

        if active_box is not None:
            x, y, w, h = active_box

            # Berikan padding 15% di sekitar wajah agar kontur dagu dan dahi tidak terpotong
            pad_x = int(0.15 * w)
            pad_y = int(0.15 * h)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame_w, x + w + pad_x)
            y2 = min(frame_h, y + h + pad_y)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size > 0:
                # 2. Tahap Klasifikasi Emosi via YOLO
                res = model(face_crop, verbose=False)[0]
                cur_probs = res.probs.data.cpu().numpy()

                # Smoothing probabilitas emosi antar-frame agar transisi bar halus
                if smoothed_probs is None:
                    smoothed_probs = cur_probs
                else:
                    smoothed_probs = 0.65 * smoothed_probs + 0.35 * cur_probs

                top_id = int(np.argmax(smoothed_probs))
                top_name = model.names[top_id]
                top_conf = float(smoothed_probs[top_id])

                color = EMOTION_COLORS.get(top_name.lower(), (0, 255, 0))

                # Gambar Bounding Box Wajah (Mulus & Tebal)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

                # Label Utama di Atas Bounding Box
                label_text = f"{top_name.upper()} ({top_conf * 100:.1f}%)"
                (txt_w, txt_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                cv2.rectangle(frame, (x, y - txt_h - 12), (x + txt_w + 10, y), color, -1)
                cv2.putText(frame, label_text, (x + 5, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

                # 3. HUD Bar Probabilitas Emosi (Pojok Kiri Atas)
                overlay = frame.copy()
                cv2.rectangle(overlay, (20, 20), (280, 190), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                cv2.putText(frame, "EMOTION CONFIDENCE", (30, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

                for i, emo_key in enumerate(all_emotions):
                    emo_id = None
                    for idx, name in model.names.items():
                        if name.lower() == emo_key:
                            emo_id = idx
                            break

                    p_val = float(smoothed_probs[emo_id]) if emo_id is not None else 0.0
                    bar_y = 70 + (i * 22)
                    bar_w = int(p_val * 130)

                    b_color = EMOTION_COLORS.get(emo_key, (180, 180, 180))
                    # Label nama
                    cv2.putText(frame, f"{emo_key[:7]:<7}", (30, bar_y + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
                    # Background bar
                    cv2.rectangle(frame, (100, bar_y + 2), (230, bar_y + 14), (55, 55, 55), -1)
                    # Nilai bar aktif
                    cv2.rectangle(frame, (100, bar_y + 2), (100 + bar_w, bar_y + 14), b_color, -1)
                    # Persentase
                    cv2.putText(frame, f"{p_val * 100:.0f}%", (238, bar_y + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        cv2.imshow("Facial Emotion Recognition - Real-Time", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Sesi kamera dihentikan.")

if __name__ == "__main__":
    main()
