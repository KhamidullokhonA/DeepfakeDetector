import time
import csv
import os
import glob
import torch
import cv2  # NEW: For video processing
import numpy as np
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import argparse

# Load your trained model
def load_model(model_path="models/best_model.pt"):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(in_features, 2)
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    model.eval()
    return model

# Preprocess and classify a SINGLE IMAGE
def predict_image(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        inference_start = time.time()
        output = model(input_tensor)
        inference_time = time.time() - inference_start

        probs = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()

    label = "FAKE" if pred == 1 else "REAL"
    confidence = probs[pred].item()
    fps = 1.0 / inference_time if inference_time > 0 else float('inf')

    # CSV Logging
    csv_file = "research_results.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Filename", "Prediction", "Confidence", "Inference_Time_Sec", "FPS"])
        writer.writerow([os.path.basename(image_path), label, f"{confidence:.4f}", f"{inference_time:.6f}", f"{fps:.2f}"])

    print(f"[{os.path.basename(image_path)}] -> {label} (Conf: {confidence:.2%} | Speed: {fps:.1f} FPS)")

# Preprocess and classify a VIDEO (New for your demo!)
def predict_video(video_path, model, skip_frames=5):
    print(f"\n🎥 Opening Video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Error: Could not open video file.")
        return

    fake_count, real_count, frame_count, total_fps = 0, 0, 0, 0

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # Video finished
        
        frame_count += 1
        
        # Skip frames so the demo doesn't take 20 minutes to run
        if frame_count % skip_frames != 0:
            continue

        # Convert OpenCV's BGR image format to PIL's RGB format
        color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_converted)
        input_tensor = transform(pil_image).unsqueeze(0)

        with torch.no_grad():
            inference_start = time.time()
            output = model(input_tensor)
            inference_time = time.time() - inference_start

        pred = torch.argmax(torch.softmax(output, dim=1)[0]).item()
        
        if pred == 1:
            fake_count += 1
        else:
            real_count += 1

        fps = 1.0 / inference_time if inference_time > 0 else 0
        total_fps += fps
        
        # Print a live updating progress bar
        print(f"\rAnalyzing frame {frame_count}... Model Speed: {fps:.1f} FPS", end="")

    cap.release()
    
    frames_analyzed = real_count + fake_count
    if frames_analyzed == 0:
        return

    fake_percentage = (fake_count / frames_analyzed) * 100
    avg_fps = total_fps / frames_analyzed

    print("\n\n📊 --- VIDEO RESULTS ---")
    print(f"Frames Sampled: {frames_analyzed}")
    print(f"Average Speed:  {avg_fps:.1f} FPS")
    if fake_percentage > 50:
        print(f"🧠 Final Verdict: FAKE ({fake_percentage:.1f}% of frames flagged)")
    else:
        print(f"🧠 Final Verdict: REAL ({(100 - fake_percentage):.1f}% of frames passed)")

# Run from terminal
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_path", help="Path to image, folder, or video (.mp4)")
    args = parser.parse_args()

    print("Loading model weights... (This cold start is not timed)")
    model = load_model()
    
    # 1. Check if it's a video file
    if args.target_path.lower().endswith(('.mp4', '.mov', '.avi')):
        predict_video(args.target_path, model)
        
    # 2. Check if it's a single image
    elif os.path.isfile(args.target_path):
        predict_image(args.target_path, model)
        
    # 3. Check if it's a folder of images
    elif os.path.isdir(args.target_path):
        image_files = glob.glob(os.path.join(args.target_path, "*.[jJ][pP][gG]")) + \
                      glob.glob(os.path.join(args.target_path, "*.[pP][nN][gG]"))
        if len(image_files) == 0:
            print(f"❌ No images found in '{args.target_path}'")
        else:
            print(f"📁 Found {len(image_files)} images. Starting batch processing...\n")
            for img_path in image_files:
                predict_image(img_path, model)
            print("\n✅ Batch complete! Check your research_results.csv")
    else:
        print("❌ Error: Path not found.")
