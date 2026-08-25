# Medical-AI — Deteksi Mata (Frontend + Backend)

Integrasi front-end statis (HTML/CSS/JS) dengan backend CNN klasifikasi
4 penyakit mata (`cataract`, `diabetic_retinopathy`, `glaucoma`, `normal`)
dari `Calssification_Eye.zip`.

## Struktur

```
medical-ai-eye/
├── backend/
│   ├── api.py                 # FastAPI: GET /health, POST /predict
│   ├── predict.py             # CLI prediksi satu gambar (dari zip asli)
│   ├── train.py               # (dari zip asli, opsional)
│   ├── eye_cnn/                # arsitektur model + transform (dari zip asli)
│   ├── runs/eyecnn/best_model.pt
│   ├── requirements.txt        # dependencies asli (torch, torchvision, dst.)
│   └── requirements-api.txt    # tambahan: fastapi, uvicorn, python-multipart
└── frontend/
    ├── index.html               # homepage (desain sebelumnya)
    ├── deteksi-mata.html        # halaman upload + hasil klasifikasi
    ├── css/style.css            # satu stylesheet dipakai kedua halaman
    └── js/deteksi-mata.js       # logic upload, panggil API, render hasil
```

## Kenapa pendekatan ini (bukan iframe Streamlit)

Karena front-end kamu sudah punya desain khusus (topbar, upload card,
palet warna dark-teal/gold), API custom dengan FastAPI dipilih supaya
halaman "Deteksi Mata" terasa menyatu dengan seluruh situs — bukan
jendela Streamlit yang gayanya beda sendiri. `app.py` (Streamlit) dari
zip aslinya tidak dipakai lagi di alur ini, tapi tetap ada di
`Calssification_Eye.zip` asli kalau suatu saat masih ingin dipakai
untuk eksperimen cepat.

## 1. Menjalankan backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-api.txt

uvicorn api:app --reload --port 8000
```

Cek server hidup: buka `http://127.0.0.1:8000/health` di browser — akan
tampil JSON `{"status":"ok","classes":[...]}`.

> Model (`best_model.pt`, ~11 MB) dimuat sekali saat server start, jadi
> prediksi berikutnya cepat. Endpoint `POST /predict` menerima
> `multipart/form-data` dengan field `file` (JPG/PNG) dan mengembalikan
> JSON berisi prediksi + probabilitas tiap kelas.

## 2. Menjalankan frontend

Buka `frontend/index.html` langsung di browser, atau jalankan server
statis ringan (disarankan, supaya `fetch()` ke API tidak diblokir
CORS oleh `file://`):

```powershell
cd frontend
python -m http.server 5500
```

Lalu buka `http://127.0.0.1:5500/index.html`, klik menu
**CNN → DETEKSI MATA**.

## 3. Menggunakan halaman Deteksi Mata

1. Pastikan API backend (`uvicorn api:app`) sedang berjalan.
2. Buka `deteksi-mata.html` melalui navbar.
3. Kolom **API** di form sudah terisi `http://127.0.0.1:8000/predict`
   secara default — ubah jika backend dijalankan di alamat/port lain.
4. Unggah citra fundus mata → klik **Prediksi Sekarang**.
5. Hasil (kelas dengan probabilitas tertinggi + bar probabilitas semua
   kelas) muncul di panel kanan.

## Catatan

- `CORSMiddleware` di `api.py` diset `allow_origins=["*"]` supaya mudah
  saat pengembangan lokal. Persempit ke domain frontend kamu sebelum
  dipakai di server publik/produksi.
- Disclaimer medis sudah ditampilkan otomatis di bawah hasil prediksi
  pada halaman Deteksi Mata — ini bukan diagnosis resmi.
- Untuk menambahkan **Deteksi Gigi** / **Deteksi Kulit** nanti, pola yang
  sama bisa dipakai: endpoint baru di `api.py` (atau service terpisah)
  + halaman HTML baru yang memakai `css/style.css` yang sama.
