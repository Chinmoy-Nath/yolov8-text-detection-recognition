# 🪪 Smart Text Recognition (YOLOv8 + Gradio Frontend)

A website-like document and ID text recognition system using YOLOv8 for detection and Deep Learning OCR. Entirely interactive via a user-friendly Gradio web interface.
(This is just a demo model, trained on very few Images, works well for the Maharashtra driving license)
---

## 🚀 Features

- Works only for DL images detection using YOLOv8
- Intelligent orientation and cropping
- Robust text extraction with EasyOCR
- Beautiful Gradio frontend, styled like a modern website
- responsive UI (custom CSS)
- Modular backend for easy integration and extension

---

## 📁 Folder Structure

Main-app/
├── backend/
│ └── DL_script.py 
├── frontend/
│ ├── gradio_app.py 
│ ├── static/
│ │ ├── custom.css 
│ │ └── banner.png/jpeg 
├── models/
│ └── *.pt 
├── requirements.txt
├── README.md

text

---

## ⚡ Quick Start

1. Clone this repo
2. Install dependencies:
pip install -r requirements.txt

text
3. Download and place your model `.pt` files in `models/`.
4. Run the app:
python frontend/gradio_app.py

text
5. Open the local URL printed by Gradio

---

## 🤖 Model Details

- **Detection:** YOLOv8
- **Classifier/Orientation:** Custom YOLOv8 models
- **Text extraction:** EasyOCR
- **Pipeline:** Image → Crop → Orientation → Region Detection → OCR field extraction

---

## 📚 How to Use

1. Upload a clear photo or scan of supported card or document
2. Click **Extract Details**
3. Fields will be displayed (name, date, number, etc.)

---
