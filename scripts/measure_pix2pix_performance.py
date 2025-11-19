import os
import torch
from torchvision import transforms
from PIL import Image
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as SSIM
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torchvision.transforms.functional as TF


# ==========================
# CONFIGURATION
# ==========================
input_folder = "/remote/tnas1/algo/koren/dfw-pix2pix-dc/data_v5/trainA_crops"
gt_folder    = "/remote/tnas1/algo/koren/dfw-pix2pix-dc/data_v5/trainB_crops"

output_folder = "/remote/tnas1/algo/koren/dfw-pix2pix-dc/results_v19/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your trained generator
from models.networks import define_G  # adjust import to your pix2pix repo structure


netG = define_G(input_nc=3, output_nc=3, ngf=128, netG='unet_256', norm='instance',
                predict_delta=True, use_dropout=True, use_upsample=False, use_modulated_conv=False)
checkpoint_path = "/home/korenbe/repos/pytorch-CycleGAN-and-pix2pix_research/checkpoints/v19/latest_net_G.pth"
netG.load_state_dict(torch.load(checkpoint_path, map_location=device))
netG.to(device)
netG.eval()

# ==========================
# PREPROCESSING & POSTPROCESSING
# ==========================
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# We’ll normalize the same way as in training
normalize = transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))


def preprocess_v1(img):
    """Convert PIL → Tensor in [-1,1] range"""
    return normalize(to_tensor(img)).unsqueeze(0).to(device)

def preprocess(img):
    w, h = img.size
    target_w = int(np.ceil(w / 256) * 256)
    target_h = int(np.ceil(h / 256) * 256)
    pad_w1 = max(target_w - w, 0) // 2
    pad_h1 = max(target_h - h, 0) // 2
    img = TF.pad(img, (pad_w1, pad_h1, target_w - w - pad_w1, target_h-h-pad_h1), fill=(251, 251, 251))  # left, top, right, bottom
    return normalize(to_tensor(img)).unsqueeze(0).to(device)

def center_crop(img, target_w, target_h):
    w, h = img.size

    left = (w - target_w) // 2
    top = (h - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    return img.crop((left, top, right, bottom))

def postprocess_v1(tensor):
    """Convert model output [-1,1] → uint8 PIL"""
    img = (tensor.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
    return to_pil(img)

def postprocess(tensor, org_w, org_h):
    """Convert model output [-1,1] → uint8 PIL"""
    img = (tensor.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
    pil_img = to_pil(img)
    trimmed = center_crop(pil_img, org_w, org_h)  # or whatever size you want
    return trimmed


# ==========================
# EVALUATION LOOP
# ==========================
os.makedirs(output_folder, exist_ok=True)
ssim_metric = SSIM(data_range=1.0).to(device)
ssim_scores = []
mse_scores = []  # <-- NEW


with torch.no_grad():
    for fname in tqdm(sorted(os.listdir(input_folder))):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            continue

        # Load input (pre-filtered)
        imgA = Image.open(os.path.join(input_folder, fname)).convert("RGB")

        # Load ground truth (post-filtered)
        imgB_path = os.path.join(gt_folder, fname)
        if not os.path.exists(imgB_path):
            print(f"⚠️ Warning: no matching ground-truth for {fname}")
            continue
        imgB = Image.open(imgB_path).convert("RGB")

        w, h = imgA.size
        # Prepare tensors
        input_tensor = preprocess(imgA)
        output_tensor = netG(input_tensor)
        output_img = postprocess(output_tensor, w, h)

        # Prepare tensors_v1
        # input_tensor = preprocess_v1(imgA)
        # output_tensor = netG(input_tensor)
        # output_img = postprocess_v1(output_tensor)
        output_img.save(os.path.join(output_folder, fname))

        # vis_A = cv2.cvtColor(np.array(imgA), cv2.COLOR_BGR2RGB)
        # vis_B = cv2.cvtColor(np.array(imgB), cv2.COLOR_BGR2RGB)
        # vis_R = cv2.cvtColor(np.array(output_img), cv2.COLOR_BGR2RGB)
        # # diff_img = np.abs(np.subtract(vis_B, vis_R))
        # vis_img = np.concatenate((vis_A, vis_B, vis_R), axis=1)
        # cv2.imshow("imgA/imgB/output_img", vis_img)
        # cv2.waitKey(0)

        # --- SSIM against ground truth ---
        gt_tensor = to_tensor(imgB).unsqueeze(0).to(device)
        output_for_metric = to_tensor(output_img).unsqueeze(0).to(device)

        ssim_value = ssim_metric(output_for_metric, gt_tensor)
        ssim_scores.append(ssim_value.item())

        # ==========================
        # MSE (pixelwise, over [0,1])
        # ==========================
        mse_value = torch.mean((output_for_metric - gt_tensor) ** 2).item()
        mse_scores.append(mse_value)

# ==========================
# RESULTS
# ==========================
mean_ssim = sum(ssim_scores) / len(ssim_scores)
mean_mse = sum(mse_scores) / len(mse_scores)

print(f"\n✅ Mean SSIM (Fake vs Ground Truth) over {len(ssim_scores)} images: {mean_ssim:.4f}")
print(f"📉 Mean MSE  over {len(mse_scores)} images: {mean_mse:.6f}")

