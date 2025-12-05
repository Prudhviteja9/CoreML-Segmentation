import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
import os

INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cityscapes-like color map (simple)
COLORS = np.array([
    [0, 0, 0],        # background
    [128, 64, 128],   # road
    [70, 70, 70],     # building
    [0, 0, 142],      # car
    [220, 20, 60],    # person
    [0, 80, 100],     # sign
    [0, 0, 230],      # sky
], dtype=np.uint8)

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_model(device):
    # Pretrained DeepLabv3 on COCO
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights="DEFAULT")
    model.eval()
    model.to(device)
    return model

def preprocess(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    tensor = transform(img_rgb).unsqueeze(0)  # (1,3,H,W)
    return tensor

def decode_mask(mask):
    # Mask: (H, W) with class indices
    num_classes = COLORS.shape[0]
    mask = np.clip(mask, 0, num_classes - 1)
    color_mask = COLORS[mask]
    return color_mask

def overlay_mask(image, mask, alpha=0.5):
    overlay = cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)
    return overlay

def main():
    device = get_device()
    print(f"Using device: {device}")

    model = load_model(device)

    image_files = [f for f in os.listdir(INPUT_DIR)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for img_name in image_files:
        img_path = os.path.join(INPUT_DIR, img_name)
        print(f"Processing {img_path}")

        img_bgr = cv2.imread(img_path)
        h, w, _ = img_bgr.shape

        inp = preprocess(img_bgr).to(device)

        with torch.no_grad():
            output = model(inp)["out"]  # (1, C, H, W)
            preds = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

        mask_color = cv2.resize(decode_mask(preds), (w, h), interpolation=cv2.INTER_NEAREST)
        overlay = overlay_mask(img_bgr, mask_color)

        out_path = os.path.join(OUTPUT_DIR, f"seg_{img_name}")
        cv2.imwrite(out_path, overlay)
        print(f"Saved segmentation overlay to {out_path}\n")

if __name__ == "__main__":
    main()
