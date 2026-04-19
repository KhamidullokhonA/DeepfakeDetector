import gradio as gr
import torch
import mimetypes
import time  # Added for timing
from PIL import Image
import cv2
from torchvision.models import efficientnet_b0
from torchvision import transforms

# === Load Model ===
def load_model():
    model = efficientnet_b0()
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    # Using strict=False to ensure it loads even if weights are from a slightly different version
    model.load_state_dict(torch.load("models/best_model.pt", map_location="cpu"), strict=False)
    model.eval()
    return model

model = load_model()

# === Preprocessing ===
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Inference Logic ===
def predict_file(file_obj):
    if file_obj is None:
        return "⚠️ No file selected", "", None, ""

    path = file_obj.name
    mime, _ = mimetypes.guess_type(path)
    
    img = None
    
    # Check if it's an image or video to extract the frame
    if mime and mime.startswith("image"):
        img = Image.open(path).convert("RGB")
    elif mime and mime.startswith("video"):
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return "❌ Error reading video", "", None, ""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
    else:
        return "Unsupported file type", "", None, ""

    # --- Start Performance Timing ---
    tensor = preprocess(img).unsqueeze(0)
    
    start_time = time.time()  # START CLOCK
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
    end_time = time.time()    # STOP CLOCK
    
    inference_time = end_time - start_time
    fps = 1.0 / inference_time if inference_time > 0 else 0
    
    conf, pred = torch.max(probs, dim=0)
    
    # Label logic
    label_base = "🟢 Real" if pred.item() == 0 else "🔴 Deepfake"
    label = f"{label_base} (Video Frame)" if mime.startswith("video") else label_base
    
    # Formatting metrics for research
    metrics = f"⚡ Speed: {fps:.2f} FPS\n⏱️ Latency: {inference_time*1000:.2f} ms"
    
    return label, f"{conf.item()*100:.2f}%", img, metrics

# === Gradio UI ===
with gr.Blocks(title="Deepfake Detector Research Tool") as demo:
    gr.Markdown("## 🧠 Deepfake Detector & Performance Benchmarker")
    gr.Markdown("Analysis of authenticity vs. computational efficiency (FPS).")

    with gr.Row():
        file_input = gr.File(
            label="Upload Image or Video",
            file_types=[".jpg", ".jpeg", ".png", ".mp4", ".mov"],
        )
        preview = gr.Image(label="Processed Frame", interactive=False)

    with gr.Row():
        prediction = gr.Textbox(label="Detection Result", interactive=False)
        confidence = gr.Textbox(label="Confidence (%)", interactive=False)
        # NEW: Performance metrics output
        performance = gr.Textbox(label="Performance Metrics (FPS)", interactive=False)

    file_input.change(
        fn=predict_file,
        inputs=file_input,
        outputs=[prediction, confidence, preview, performance]
    )

demo.launch()
