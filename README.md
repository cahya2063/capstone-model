---
title: HistoryLens
emoji: 👍
colorFrom: purple
colorTo: red
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Classification Image using MobileNetV2
metrics:
  - accuracy
pipeline_tag: image-classification
library_name: keras
tags:
  - capstone
  - image-classification
  - keras
  - computer-vision
  - gradio
  - historycal-sites
  - tensorflow
datasets:
  - custom
---

# HistoryLens - Capstone DBS Coding Camp

Proyek ini adalah aplikasi klasifikasi gambar berbasis deep learning untuk mengenali situs cagar budaya di Daerah Istimewa Yogyakarta (DIY). Dibuat menggunakan Gradio dan TensorFlow/Keras, ditujukan untuk membantu pengguna mengenali tempat bersejarah hanya dengan mengunggah foto.

## Persyaratan Sistem

Sistem ini direkomendasikan untuk dijalankan di:

- OS: Windows 10/11 64-bit
- Python 3.10

---

##  Daftar Kelas

Model mengenali 10 lokasi berikut:
- Benteng Vredeburg  
- Candi Borobudur  
- Candi Prambanan  
- Gedung Agung Istana Kepresidenan  
- Masjid Gedhe Kauman  
- Monumen Serangan 1 Maret  
- Museum Gunungapi Merapi  
- Situs Ratu Boko  
- Taman Sari  
- Tugu Yogyakarta

## Arsitektur Model
- MobileNetV2 kustom dengan TensorFlow/Keras
- Input: Gambar RGB berukuran 224x224x3
- Output layer: Softmax (10 kelas)

##  Tools and Library
- Python, TensorFlow/Keras
- Gradio untuk antarmuka pengguna
- Model disimpan dalam format `.json` dan `.h5`
- Huggingface sebagai tools deploy model


##  Fitur
- login dan register
- Upload gambar sesuai yang ada di point kategori
- Model akan memprediksi nama lokasi dari gambar tersebut
- Menampilkan gambar unggahan dan hasil klasifikasinya
- Menampilkan Deskripsi terkait gambar yang diupload
- Menampilkan link Google maps 
- Menyimpan History dari detekesi gambar
- Berjalan langsung di browser
