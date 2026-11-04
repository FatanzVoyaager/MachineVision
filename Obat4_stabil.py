import cv2
import numpy as np
import serial
import time
from collections import deque, Counter

# ============================================================
# KONFIGURASI YANG DAPAT DISESUAIKAN
# ============================================================
COM_PORT = "COM3"
BAUDRATE = 9600
CAMERA_INDEX = 1
CAMERA_BACKEND = cv2.CAP_DSHOW  # Windows; ganti ke cv2.CAP_DSHOW atau default jika perlu
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Aktifkan autofocus bila webcam/driver mendukungnya.
TRY_AUTOFOCUS = True

# Zona keputusan. Bounding box kemasan harus berada sepenuhnya di dalam zona ini.
# Format: (x-kiri, y-atas, x-kanan, y-bawah), dalam proporsi frame 0.0-1.0.
# Ubah nilai ini agar sesuai posisi kemasan saat berada di titik inspeksi konveyor.
ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 0.05, 0.05, 0.95, 0.95

# Waktu dan syarat kestabilan keputusan.
STABLE_SECONDS = 0.5       # Kemasan harus konsisten sekitar 2 detik
MIN_VALID_RATIO = 0.80      # Minimal 80% frame dalam jendela berisi kandidat objek
MIN_VOTE_RATIO = 0.85       # Minimal 85% hasil kandidat harus setuju A atau R
MIN_STABLE_SAMPLES = 10     # Minimum sampel dalam jendela kestabilan
REARM_ABSENT_FRAMES = 15    # Kemasan baru bisa diproses setelah objek hilang 15 frame

# Parameter inspeksi bentuk dari program awal.
MIN_AREA = 3000
MIN_SOLIDITY = 0.88
MIN_RECTANGULARITY = 0.60
TOLERANCE_DEFECT = 2
TOLERANCE_APPROX = 10

# Mask warna kuning HSV OpenCV: H 0-179, S/V 0-255.
LOWER_YELLOW = np.array([15, 35, 50], dtype=np.uint8)
UPPER_YELLOW = np.array([45, 255, 255], dtype=np.uint8)

# Gerbang fokus: keputusan ditahan bila skor ketajaman di bawah ambang.
# Ambang perlu diuji pada kamera dan jarak kerja yang sebenarnya.
REQUIRE_GOOD_FOCUS = True
MIN_FOCUS_SCORE = 35.0

SHOW_DEBUG_WINDOWS = True
DRY_RUN = False  # True = hanya tampilkan keputusan, tidak mengirim A/R ke Arduino


def focus_score(gray_crop):
    """Perkiraan ketajaman memakai varians Laplacian; nilainya bergantung kamera/adegan."""
    if gray_crop is None or gray_crop.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())


def evaluate_contour(contour, frame_shape, roi_bounds):
    """Nilai kontur kuning terbesar yang memenuhi zona dan luas minimum."""
    frame_h, frame_w = frame_shape[:2]
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds
    roi_contours = []

    for cnt in contour:
        area = cv2.contourArea(cnt)
        if area <= MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        # Hindari menilai kemasan saat baru masuk/keluar frame/zona.
        if x < roi_x1 or y < roi_y1 or x + w > roi_x2 or y + h > roi_y2:
            continue
        if w <= 0 or h <= 0:
            continue
        roi_contours.append((area, cnt, x, y, w, h))

    if not roi_contours:
        return None

    # Untuk satu kemasan pada titik inspeksi, pilih kontur kuning dengan luas terbesar.
    area, cnt, x, y, w, h = max(roi_contours, key=lambda item: item[0])
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    rectangularity = area / float(w * h) if w * h > 0 else 0.0

    defect_count = 0
    hull_indices = cv2.convexHull(cnt, returnPoints=False)
    if hull_indices is not None and len(hull_indices) >= 4:
        try:
            defects = cv2.convexityDefects(cnt, hull_indices)
        except cv2.error:
            defects = None

        if defects is not None:
            for i in range(defects.shape[0]):
                _start, _end, far, depth = defects[i, 0]
                depth_px = depth / 256.0
                if depth_px > min(w, h) * 0.025 and depth_px > 5:
                    defect_count += 1

    is_accept = (
        4 <= len(approx) <= TOLERANCE_APPROX
        and solidity >= MIN_SOLIDITY
        and rectangularity >= MIN_RECTANGULARITY
        and defect_count <= TOLERANCE_DEFECT
    )
    status = "A" if is_accept else "R"
    return {
        "contour": cnt,
        "approx": approx,
        "area": area,
        "bbox": (x, y, w, h),
        "solidity": solidity,
        "rectangularity": rectangularity,
        "defects": defect_count,
        "status": status,
    }


def draw_text(frame, text, xy, color=(255, 255, 255), scale=0.6, thickness=2):
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def main():
    camera = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError(
            f"Kamera index {CAMERA_INDEX} tidak dapat dibuka. Periksa indeks kamera/kabel."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if TRY_AUTOFOCUS and hasattr(cv2, "CAP_PROP_AUTOFOCUS"):
        try:
            autofocus_ok = camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            if autofocus_ok:
                print("[INFO] Perintah autofocus diterima oleh driver kamera.")
            else:
                print("[WARN] Kamera/driver tidak mengonfirmasi autofocus; atur fokus secara manual bila perlu.")
        except cv2.error:
            print("[WARN] Autofocus tidak didukung oleh kamera/driver ini.")

    actual_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = camera.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Kamera aktif: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

    # Buka serial setelah kamera siap. Program mengirim 'A'/'R' sekali per kemasan.
    ser = serial.Serial(COM_PORT, BAUDRATE, timeout=1, write_timeout=1)
    time.sleep(2.0)  # Arduino tertentu reset saat port serial dibuka.

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    history = deque()
    empty_frames = 0
    sent_for_current_object = False
    final_decision = "MENUNGGU OBJEK"
    last_candidate_status = None

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None or frame.size == 0:
                print("[WARN] Frame kamera gagal dibaca; melewati frame ini.")
                time.sleep(0.02)
                continue

            if frame.shape[1] != CAMERA_WIDTH or frame.shape[0] != CAMERA_HEIGHT:
                frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_LINEAR)

            now = time.monotonic()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.medianBlur(frame, 3)
            blur = cv2.GaussianBlur(blur, (5, 5), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            out = frame.copy()
            h, w = frame.shape[:2]
            rx1 = int(ROI_X1 * w)
            ry1 = int(ROI_Y1 * h)
            rx2 = int(ROI_X2 * w)
            ry2 = int(ROI_Y2 * h)
            # ROI tetap dipakai untuk membatasi keputusan, tetapi tidak digambar pada frame.

            result = evaluate_contour(contours, frame.shape, (rx1, ry1, rx2, ry2))
            candidate_status = None
            current_focus = 0.0
            if result is not None:
                x, y, bw, bh = result["bbox"]
                crop_gray = gray[y:y + bh, x:x + bw]
                current_focus = focus_score(crop_gray)
                focus_is_good = current_focus >= MIN_FOCUS_SCORE

                if not REQUIRE_GOOD_FOCUS or focus_is_good:
                    candidate_status = result["status"]
                else:
                    candidate_status = None

                box_color = (0, 255, 0) if result["status"] == "A" else (0, 0, 255)
                cv2.drawContours(out, [result["approx"]], -1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.rectangle(out, (x, y), (x + bw, y + bh), box_color, 2, cv2.LINE_AA)
                candidate_label = f"Kandidat: {result['status']}" if (not REQUIRE_GOOD_FOCUS or focus_is_good) else f"TAHAN: fokus rendah (prediksi {result['status']})"
                draw_text(out, candidate_label, (x, max(20, y - 28)), box_color, 0.55, 2)
                draw_text(out, f"Area:{int(result['area'])} Sol:{result['solidity']:.2f} Rect:{result['rectangularity']:.2f}",
                          (10, h - 42), (255, 255, 255), 0.48, 1)
                focus_label = "OK" if focus_is_good else "RENDAH"
                focus_color = (0, 255, 0) if focus_is_good else (0, 165, 255)
                draw_text(out, f"Fokus:{current_focus:.1f} ({focus_label})", (10, h - 20), focus_color, 0.48, 1)
            else:
                draw_text(out, "Tidak ada kemasan utuh di zona inspeksi", (10, h - 20), (0, 165, 255), 0.48, 1)

            # Histori waktu: hasil 'None' ikut dihitung agar deteksi yang putus-putus
            # tidak dianggap stabil. Hanya gunakan jendela waktu terakhir.
            history.append((now, candidate_status))
            while history and now - history[0][0] > STABLE_SECONDS:
                history.popleft()

            if candidate_status is None:
                empty_frames += 1
            else:
                empty_frames = 0
                last_candidate_status = candidate_status

            # Setelah objek benar-benar meninggalkan zona, izinkan kemasan berikutnya.
            if empty_frames >= REARM_ABSENT_FRAMES:
                if sent_for_current_object:
                    print("[INFO] Objek sebelumnya telah meninggalkan zona; sistem siap untuk objek berikutnya.")
                sent_for_current_object = False
                final_decision = "MENUNGGU OBJEK"
                last_candidate_status = None
                # Buang sampel lama supaya kemasan berikutnya mulai dengan histori baru.
                history.clear()
                history.append((now, None))

            valid_samples = [status for _t, status in history if status in ("A", "R")]
            valid_ratio = len(valid_samples) / len(history) if history else 0.0
            vote_status = None
            vote_ratio = 0.0
            if valid_samples:
                counts = Counter(valid_samples)
                vote_status, count = counts.most_common(1)[0]
                vote_ratio = count / len(valid_samples)

            time_span = (history[-1][0] - history[0][0]) if len(history) >= 2 else 0.0
            stable_ready = (
                not sent_for_current_object
                and len(history) >= MIN_STABLE_SAMPLES
                and time_span >= STABLE_SECONDS * 0.75
                and valid_ratio >= MIN_VALID_RATIO
                and len(valid_samples) >= MIN_STABLE_SAMPLES
                and vote_status is not None
                and vote_ratio >= MIN_VOTE_RATIO
            )

            if stable_ready:
                command = vote_status.encode("ascii")
                try:
                    if DRY_RUN:
                        sent_for_current_object = True
                        final_decision = "ACCEPT (DRY-RUN)" if vote_status == "A" else "REJECT (DRY-RUN)"
                        print(f"[DRY-RUN] Keputusan {vote_status}; tidak ada perintah dikirim ke Arduino.")
                    else:
                        sent = ser.write(command)
                        ser.flush()
                        if sent == 1:
                            sent_for_current_object = True
                            final_decision = "ACCEPT TERKONFIRMASI" if vote_status == "A" else "REJECT TERKONFIRMASI"
                            print(
                                f"[SERIAL] Kirim {vote_status} satu kali | valid={valid_ratio:.0%} "
                                f"vote={vote_ratio:.0%} | window={time_span:.2f}s"
                            )
                        else:
                            print("[ERROR] Perintah serial tidak terkirim lengkap.")
                except (serial.SerialException, serial.SerialTimeoutException) as exc:
                    print(f"[ERROR] Gagal mengirim serial: {exc}")

            if sent_for_current_object:
                decision_color = (0, 255, 0) if final_decision.startswith("ACCEPT") else (0, 0, 255)
            elif valid_samples:
                decision_color = (0, 200, 255)
                final_decision = f"MENUNGGU STABIL {time_span:.1f}/{STABLE_SECONDS:.1f}s"
            else:
                decision_color = (0, 165, 255)
                final_decision = "MENUNGGU OBJEK"

            draw_text(out, f"Keputusan: {final_decision}", (10, 32), decision_color, 0.7, 2)
            draw_text(out, f"Konsistensi: valid {valid_ratio:.0%} | vote {vote_ratio:.0%}", (10, 58), (255, 255, 255), 0.5, 1)

            if SHOW_DEBUG_WINDOWS:
                cv2.imshow("Kamera Asli", frame)
                cv2.imshow("Mask Kuning", mask)
            cv2.imshow("Inspeksi Kemasan", out)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    finally:
        camera.release()
        ser.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
