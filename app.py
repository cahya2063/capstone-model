import os
import uuid
import uvicorn
from io import BytesIO

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json

import gradio as gr
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from description import description
from location import location


UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def load_model_from_file(json_path, h5_path):
    with open(json_path, "r") as f:
        json_config = f.read()
        model = model_from_json(json_config)
    model.load_weights(h5_path)
    return model

model = load_model_from_file("model.json", "my_model.h5")

labels = [
    "Benteng Vredeburg", "Candi Borobudur", "Candi Prambanan", "Gedung Agung Istana Kepresidenan",
    "Masjid Gedhe Kauman", "Monumen Serangan 1 Maret", "Museum Gunungapi Merapi",
    "Situs Ratu Boko", "Taman Sari", "Tugu Yogyakarta"
]

def classify_image(img):
    try:
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        pred = model.predict(img_array)[0]
        confidence = np.max(pred)
        predicted_label = labels[np.argmax(pred)]

        akurasi = float(confidence)
        if confidence < 0.8:
            label_output = "Tidak dapat dikenali (Confidence: {:.2f}%)".format(confidence * 100)
            deskripsi = (
                "Tolong arahkan ke objek yang jelas agar bisa diidentifikasikan. "
                "Pastikan anda berada di salah satu tempat seperti:\n"
                "- Benteng Vredeburg\n- Candi Borobudur\n- Candi Prambanan\n"
                "- Gedung Agung Istana Kepresidenan Yogyakarta\n- Masjid Gedhe Kauman\n"
                "- Monumen Serangan 1 Maret\n- Museum Gunungapi Merapi\n- Situs Ratu Boko\n"
                "- Taman Sari\n- Tugu Yogyakarta"
            )
            lokasi = "-"
        else:
            label_output = f"{predicted_label} (Confidence: {confidence * 100:.2f}%)"
            deskripsi = description.get(predicted_label, "Deskripsi belum tersedia.")
            lokasi = location.get(predicted_label, None)
            if lokasi:
                lokasi = f'<a href="{lokasi}" target="_blank">Lihat Lokasi di Google Maps</a>'
            else:
                lokasi = "Lokasi tidak ditemukan"

        return label_output, deskripsi, lokasi, akurasi

    except Exception as e:
        return "Error", str(e), "-",0.0
    
app = FastAPI()
    
def create_app():
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root():
        return {"message": "Berhasil akses model"}

    @app.post("/api/predict")
    async def predict(request: Request, file: UploadFile = File(...)):
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")

        # Simpan gambar untuk referensi
        ext = file.filename.split('.')[-1]
        filename = f"{uuid.uuid4()}.{ext}"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        img.save(save_path)

        # Buat URL publik untuk gambar
        base_url = str(request.base_url).rstrip("/")
        image_url = f"{base_url}/static/uploads/{filename}"

        label_output, deskripsi, lokasi, confidence = classify_image(img)

        return JSONResponse(content={
            "label_output": label_output,
            "deskripsi": deskripsi,
            "lokasi": lokasi,
            "confidence": confidence,
            "image_url": image_url
        })

gradio_app = gr.Interface(
        fn=classify_image,
        inputs=gr.Image(type="pil", label="Upload Gambar"),
        outputs=[
            gr.Textbox(label="Output Klasifikasi"),
            gr.Textbox(label="Deskripsi Lengkap", lines=20, max_lines=50),
            gr.HTML(label="Link Lokasi"),
        ],
        title="Klasifikasi Tempat Bersejarah di Yogyakarta",
        description="Upload gambar objek bersejarah, dan sistem akan mengklasifikasikan serta memberikan deskripsi dan lokasi."
    )

app = gr.mount_gradio_app(app, gradio_app, path="/gradio")

# ===== Jalankan Server =====
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
