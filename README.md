# OfflineFaceSwap (CMD Version) 🖼️🤖

A **100% offline face swapping tool** built in Python using [InsightFace]
Runs entirely offline — no uploads, no cloud processing and your photos never leave your computer.

---

## ✨ Features
- Batch face swap: replace a face in multiple images at once.
- Optional GFPGAN face enhancement for sharper results.
- Works with CPU or GPU (CUDA).
- Completely offline — privacy-friendly.

---

## 📦 Installation

### 1. Clone the Repository

git clone https://github.com/prasad7710/OfflineFaceSwap.git
cd OfflineFaceSwap

### 2. Create a Virtual Environment
Windows:

python -m venv venv
venv\Scripts\activate

Mac/Linux:
python3 -m venv venv
source venv/bin/activate

### 3. Install Requirements
pip install -r requirements.txt

### 4. Download Required Models

You must manually download the following models (not included in this repo due to size) and place them in the correct folders:

A. GFPGAN Model
File: GFPGANv1.3.pth (~332 MB)
Place in:
OfflineFaceSwap/gfpgan/GFPGANv1.3.pth

B. InsightFace Model
File: inswapper_128.onnx (~528 MB)
Place in:
OfflineFaceSwap/inswapper_128.onnx

C. Shape Predictor
File: shape_predictor_68_face_landmarks.dat (~97 MB)
Place in:
OfflineFaceSwap/shape_predictor_68_face_landmarks.dat

▶️ Usage
Basic CMD Run:
python face_swap.py

*** Make sure you have ***
source/source.jpg → image containing the face to swap
target/ folder → contains all images where the face will be swapped
output/ folder → will be created automatically for results

☕ Support My Work

If you like this tool and want to support further development:
