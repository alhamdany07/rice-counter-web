import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024   # batasi unggahan 25 MB

MAX_SIDE = 1280   # foto besar diperkecil agar parameter (kernel, luas) berlaku sama di semua resolusi

def encode_image_to_base64(img_bgr, png=False):
    if png:   # mask biner: PNG agar tepinya tajam
        _, buffer = cv2.imencode('.png', img_bgr)
        return "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')
    _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

def proses_segmentasi_beras(img_bgr, blur_k=3, min_area=60):
    blur_k = max(1, int(blur_k))
    if blur_k % 2 == 0:
        blur_k += 1
    min_area = max(5, int(min_area))

    # 0. Normalisasi resolusi: foto besar diperkecil (kernel & luas minimum dalam piksel jadi konsisten)
    h0, w0 = img_bgr.shape[:2]
    if max(h0, w0) > MAX_SIDE:
        f = MAX_SIDE / max(h0, w0)
        img_bgr = cv2.resize(img_bgr, (int(w0 * f), int(h0 * f)), interpolation=cv2.INTER_AREA)
    h, w = img_bgr.shape[:2]

    # 1. Grayscale & Gaussian Blur
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0) if blur_k > 1 else gray.copy()

    # 2. Morfologi Top-Hat untuk meratakan pencahayaan & menekan latar
    ks = max(9, int(0.035 * max(h, w)) | 1)   # kernel harus lebih besar dari lebar butir
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel_tophat)

    # 3. Thresholding Otsu + Morphological Opening
    _, biner = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    biner = cv2.morphologyEx(biner, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # 4. Filter noise & pantulan garis bingkai: luas kontur dan rasio kotak berputar.
    #    Satu daftar kontur ini dipakai oleh ketiga metode, sehingga jumlah kontur tidak bisa
    #    melebihi jumlah adaptif.
    cnts_all, _ = cv2.findContours(biner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = []
    for c in cnts_all:
        if cv2.contourArea(c) < min_area:
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        if max(rw, rh) / max(1.0, min(rw, rh)) >= 7.0:      # garis tipis (termasuk yang miring)
            continue
        contours.append(c)
    biner_bersih = np.zeros_like(biner)
    cv2.drawContours(biner_bersih, contours, -1, 255, -1)
    areas = [cv2.contourArea(c) for c in contours]

    # --- METODE 1: KONTUR BIASA ---
    jumlah_kontur = len(contours)

    # --- METODE 2: WATERSHED ---
    dist = cv2.distanceTransform(biner_bersih, cv2.DIST_L2, 5)
    md = max(3, int(round(7 * max(h, w) / MAX_SIDE)))
    coords = peak_local_max(dist, min_distance=md, labels=(biner_bersih > 0).astype(int))
    mask_peaks = np.zeros(dist.shape, dtype=bool)
    if len(coords) > 0:
        mask_peaks[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask_peaks)
    ws_labels = watershed(-dist, markers, mask=biner_bersih)
    jumlah_watershed = max(0, len(np.unique(ws_labels)) - 1)

    # --- METODE 3: ADAPTIF (PEMISAHAN BERBASIS LUAS MEDIAN) ---
    solid = []
    for c, a_ in zip(contours, areas):
        hull = cv2.contourArea(cv2.convexHull(c))
        if hull > 0 and a_ / hull >= 0.92:
            solid.append(a_)
    basis = solid if len(solid) >= 2 else areas
    median_area = float(np.median(basis)) if len(basis) > 0 else 150.0
    jumlah_adaptif = 0
    vis_hasil = img_bgr.copy()
    nomor = 1
    butir_tunggal = 0
    butir_klaster = 0

    fs = 0.4 * max(1.0, max(h, w) / 1000)
    for cnt in contours:
        area = cv2.contourArea(cnt)

        estimasi_butir = max(1, int(round(area / median_area)))
        jumlah_adaptif += estimasi_butir

        if estimasi_butir == 1:
            butir_tunggal += 1
            warna = (0, 230, 118)   # Hijau: butir tunggal terpisah
        else:
            butir_klaster += estimasi_butir
            warna = (0, 191, 255)   # Kuning: klaster butir bersentuhan

        cv2.drawContours(vis_hasil, [cnt], -1, warna, 2)

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            label_txt = str(nomor) if estimasi_butir == 1 else f"{nomor}(+{estimasi_butir-1})"
            cv2.putText(vis_hasil, label_txt, (cx - 10, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
            nomor += estimasi_butir

    biner_bgr = cv2.cvtColor(biner_bersih, cv2.COLOR_GRAY2BGR)

    return {
        "jumlah_adaptif": jumlah_adaptif,
        "jumlah_kontur": jumlah_kontur,
        "jumlah_watershed": jumlah_watershed,
        "butir_tunggal": butir_tunggal,
        "butir_klaster": butir_klaster,
        "median_area": round(median_area, 1),
        "img_asli": encode_image_to_base64(img_bgr),
        "img_biner": encode_image_to_base64(biner_bgr, png=True),
        "img_hasil": encode_image_to_base64(vis_hasil),
    }

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/proses", methods=["POST"])
def proses():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file gambar yang diunggah."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong."}), 400

    blur_k = request.form.get("blur", 3)
    min_area = request.form.get("min_area", 60)

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "File bukan gambar valid."}), 400

    hasil = proses_segmentasi_beras(img_bgr, blur_k=blur_k, min_area=min_area)
    hasil["filename"] = file.filename
    return jsonify(hasil)

if __name__ == "__main__":
    # debug=True membuka konsol Werkzeug ke jaringan jika host 0.0.0.0. Pakai debug hanya di 127.0.0.1.
    app.run(debug=False, host="0.0.0.0", port=5000)