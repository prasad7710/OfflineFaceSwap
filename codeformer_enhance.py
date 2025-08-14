import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from basicsr.archs.codeformer_arch import CodeFormer
import torchvision.transforms as transforms

# Directories
INPUT_FOLDER = "output"
ENHANCED_FOLDER = "enhanced"
os.makedirs(ENHANCED_FOLDER, exist_ok=True)

# Setup CodeFormer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
codeformer = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=["32", "64", "128", "256"]).to(device)
codeformer.eval()

# Preprocess
def pil_to_tensor(pil_img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(pil_img).unsqueeze(0)

# Enhance with CodeFormer
def enhance_image(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).resize((512, 512))
    input_tensor = pil_to_tensor(pil_img).to(device)

    with torch.no_grad():
        output = codeformer(input_tensor)[0]
    output = output.squeeze(0).cpu().clamp_(-1, 1).numpy()
    output = ((output + 1) / 2 * 255).astype(np.uint8)
    output = np.transpose(output, (1, 2, 0))
    return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

# Process images
for filename in tqdm(os.listdir(INPUT_FOLDER), desc="Enhancing"):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    input_path = os.path.join(INPUT_FOLDER, filename)
    img = cv2.imread(input_path)
    enhanced_img = enhance_image(img)

    output_path = os.path.join(ENHANCED_FOLDER, filename)
    cv2.imwrite(output_path, enhanced_img)

print("✅ Enhancement complete.")
