import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import gradio as gr
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
from PIL import Image

from description import description
from location import location

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
        return "Error", str(e), "-"

interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload Gambar"),
    outputs=[
        gr.Textbox(label="Output Klasifikasi"),
        gr.Textbox(label="Deskripsi Lengkap", lines=20, max_lines=50),
        gr.HTML(label="Link Lokasi"),
    ],
    title="Klasifikasi Gambar",
    description="Upload gambar, sistem akan mengklasifikasikan dan memberikan deskripsi mengenai gambar tersebut."
)

interface.launch()
