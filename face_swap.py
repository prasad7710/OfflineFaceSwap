import os
import glob
import cv2
from PIL import Image
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import torch
import numpy as np

# === Config ===
SOURCE_IMAGE = 'source/source.jpg'
TARGET_FOLDER = 'target'
OUTPUT_FOLDER = 'output'
USE_GFPGAN = True
USE_GPU = False  # Set to True only if you have a working CUDA setup

# === Load GFPGAN if needed ===
gfpgan = None
if USE_GFPGAN:
    try:
        from gfpgan import GFPGANer
        gfpgan = GFPGANer(
            model_path='gfpgan/GFPGANv1.3.pth',
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
            device='cuda' if USE_GPU else 'cpu'
        )
        print("✅ GFPGAN loaded.")
    except Exception as e:
        print(f"⚠️ Failed to load GFPGAN: {e}")
        gfpgan = None

# === Load InsightFace models ===
print("🔍 Loading InsightFace...")
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

swapper = get_model('inswapper_128.onnx', providers=['CPUExecutionProvider'])
print("✅ InsightFace ready.")

# === Load source image ===
print(f"🖼️ Detecting face in source: {SOURCE_IMAGE}")
src_img = cv2.imread(SOURCE_IMAGE)
src_faces = face_app.get(src_img)
if len(src_faces) == 0:
    print("❌ No face found in source image.")
    exit()

src_face = src_faces[0]
print("✅ Source face detected.")

# === Process all target images ===
target_images = glob.glob(os.path.join(TARGET_FOLDER, '*.*'))

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print(f"🎯 Processing {len(target_images)} target image(s)...")

for img_path in target_images:
    try:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        faces = face_app.get(img)
        if not faces:
            print(f"⚠️ No face found in: {img_name}")
            continue

        for face in faces:
            img = swapper.get(img, face, src_face, paste_back=True)

        if gfpgan:
            _, _, restored_img = gfpgan.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            img = restored_img

        cv2.imwrite(os.path.join(OUTPUT_FOLDER, img_name), img)
        print(f"✅ Swapped: {img_name}")
    except Exception as e:
        print(f"❌ Failed on {img_name}: {e}")

print("🎉 Done! Face swap completed.")
