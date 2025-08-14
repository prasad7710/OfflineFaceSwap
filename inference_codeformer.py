import sys
import os

sys.path.append(os.path.abspath("codeformer"))

from basicsr.archs.codeformer_arch import CodeFormer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_codeformer(model_path='CodeFormer/weights/CodeFormer.pth'):
    model = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                       connect_list=['32', '64', '128', '256']).to(device)
    checkpoint = torch.load(model_path)['params_ema']
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def enhance_face_with_codeformer(model, face):
    face = cv2.resize(face, (512, 512))
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    tensor = ToTensor()(face_rgb).unsqueeze(0).to(device)
    normalize(tensor[0], [0.5]*3, [0.5]*3)
    with torch.no_grad():
        output = model(tensor, w=0.7, adain=True)[0]
    output = output.squeeze().cpu().numpy().transpose(1, 2, 0)
    output = ((output + 1) * 127.5).clip(0, 255).astype('uint8')
    return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
