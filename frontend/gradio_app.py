import gradio as gr
import cv2
from backend.DL_script import process_image

def gradio_inference(image):
    if image is None:
        return gr.update(value="Please upload an image")
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = process_image(image)
    if "error" in results:
        return results["error"]
    return "\n".join([f"{k}: {v}" for k, v in results.items()])

with gr.Blocks(css="static/custom.css") as demo:
    # HEADER/BANNER
    with gr.Row():
        gr.Image("D:/project/Practice-web-devs/yolov8-text-detection-recognition/frontend/static/banner.png", elem_id="banner", show_label=False, show_download_button=False)  # Your banner
    gr.Markdown("""
    # 🪪 **Smart Text Recognition**
    Welcome to the state-of-the-art text detection and extraction site!
    This is just a demo model.
    """)
    # TABS for Navigation
    with gr.Tab("Extract Text"):
        with gr.Row():
            with gr.Column():
                img_box = gr.Image(type="numpy", label="Upload Driving License")
                submit_btn = gr.Button("Extract Details", elem_classes="primary-btn")
            with gr.Column():
                output = gr.Textbox(label="Extracted Fields", lines=12)
        submit_btn.click(gradio_inference, inputs=img_box, outputs=output)
    with gr.Tab("About"):
        gr.Markdown("""
        This website is powered by a deep learning pipeline (YOLOv8 + OCR) for text extraction.
        - This is just a demo model for testing.
        - Upload your DL Image
        - Model detects, crops, and extracts all key fields automatically
        ---
        **Project by: Chinmoy Nath | Links: [GitHub](https://github.com/Chinmoy-Nath)**
        """)
    # FOOTER
    gr.Markdown("<div style='text-align:center;font-size:13px;padding-top:30px;'>© 2025 Your Project Name • Powered by Gradio</div>")

if __name__ == "__main__":
    demo.launch()
