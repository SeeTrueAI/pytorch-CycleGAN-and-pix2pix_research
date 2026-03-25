import os

import cv2
import torch
from torchvision import transforms
from PIL import Image
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as SSIM
from tqdm import tqdm
import numpy as np
import torchvision.transforms.functional as TF


# ==========================
# CONFIGURATION
# ==========================
input_folder = "/remote/tnas1/algo/koren/dfw-pix2pix-dc/data_v4/trainA_crops"
gt_folder    = "/remote/tnas1/algo/koren/dfw-pix2pix-dc/data_v4/trainB_crops"

output_folder = "/remote/tnas1/algo/koren/dfw-pix2pix-dc/results_v19_train/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your trained generator
from models.networks import define_G  # adjust import to your pix2pix repo structure


netG = define_G(input_nc=3, output_nc=3, ngf=128, netG='unet_256', norm='instance',
                predict_delta=True, use_dropout=True, use_upsample=False, use_modulated_conv=False)
checkpoint_path = "/home/korenbe/repos/pytorch-CycleGAN-and-pix2pix_research/checkpoints/v19/latest_net_G.pth"
preprocess_method = "resize_and_crop"
postprocess_method = "train"
load_size = 512
crop_size = 512
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

def preprocess(img, size=256, pad_value=251):
    w, h = img.size
    target_w = int(np.ceil(w / size) * size)
    target_h = int(np.ceil(h / size) * size)
    pad_w1 = max(target_w - w, 0) // 2
    pad_h1 = max(target_h - h, 0) // 2
    img = TF.pad(img, (pad_w1, pad_h1, target_w - w - pad_w1, target_h-h-pad_h1), fill=(pad_value, pad_value, pad_value))  # left, top, right, bottom
    return normalize(to_tensor(img)).unsqueeze(0).to(device)

def preprocess_resize_and_crop(img, target_size, crop_size):
    img_resize = TF.resize(img, target_size)
    img_crop = TF.crop(img_resize, 0, 0, crop_size, crop_size)
    return normalize(to_tensor(img_crop)).unsqueeze(0).to(device)

def postprocess_overfit(tensor, load_size, crop_size):
    tensor_resize = TF.resize(tensor, load_size)  # left, top, right, bottom
    tensor = TF.crop(tensor_resize, 0, 0, crop_size, crop_size)
    img = (tensor.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
    pil_img = to_pil(img)
    return pil_img

def preprocess_training_like(img, load_size, crop_size):
    # same steps as training preprocess
    img = TF.resize(img, load_size)
    img = TF.crop(img, 0, 0, crop_size, crop_size)  # use same crop for A and B!
    img = to_tensor(img)
    img = normalize(img)        # [-1,1]
    return img.unsqueeze(0).to(device)

def preprocess_training_like_img(img, load_size, crop_size):
    img = TF.resize(img, load_size)
    img = TF.crop(img, 0, 0, crop_size, crop_size)  # use same crop for A and B!
    return img

def to_uint8_image(t):
    t = (t.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)  # [-1,1] → [0,1]
    return to_pil(t)


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
mse_scores = []
# smooth_mse_scores = []


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
        if preprocess_method == "crop":
            input_tensor = preprocess(imgA)
        else:
            input_tensor = preprocess_training_like(imgA, load_size, crop_size)
        output_tensor = netG(input_tensor)
        if postprocess_method == "test":
            output_img = postprocess(output_tensor, w, h)
        else:
            output_img = to_uint8_image(output_tensor)
        output_for_metric = to_tensor(output_img).unsqueeze(0).to(device)

        # Prepare tensors_v1
        # input_tensor = preprocess_v1(imgA)
        # output_tensor = netG(input_tensor)
        # output_img = postprocess_v1(output_tensor)
        output_img.save(os.path.join(output_folder, fname))

        if fname == "1_top_3.png":
            vis_A = cv2.cvtColor(np.array(preprocess_training_like_img(imgA, load_size, crop_size)), cv2.COLOR_BGR2RGB)
            vis_B = cv2.cvtColor(np.array(preprocess_training_like_img(imgB, load_size, crop_size)), cv2.COLOR_BGR2RGB)
            vis_R = cv2.cvtColor(np.array(output_img), cv2.COLOR_BGR2RGB)
            vis_img = np.concatenate((vis_A, vis_B, vis_R), axis=1)
            cv2.imshow("imgA/imgB/output_img", vis_img)
            cv2.waitKey(0)

        # --- SSIM against ground truth ---
        if postprocess_method == "train":
            gt_tensor = preprocess_training_like(imgB, load_size, crop_size)
            gt_tensor_metric = to_tensor(to_uint8_image(gt_tensor)).unsqueeze(0).to(device)
        else:
            gt_tensor_metric = to_tensor(imgB).unsqueeze(0).to(device)

        ssim_value = ssim_metric(output_for_metric, gt_tensor_metric)
        ssim_scores.append(ssim_value.item())

        # ==========================
        # MSE (pixelwise, over [0,1])
        # ==========================
        mse_value = torch.mean((output_for_metric - gt_tensor_metric) ** 2).item()
        mse_scores.append(mse_value)

        # abs_diff = cv2.absdiff(np.array(output_img), np.array(imgB))
        # abs_diff_smooth = cv2.GaussianBlur(abs_diff, (5, 5), 0)
        # smooth_mse_value = cv2.mean(abs_diff_smooth)
        # smooth_mse_scores.append(smooth_mse_value)


# ==========================
# RESULTS
# ==========================
mean_ssim = sum(ssim_scores) / len(ssim_scores)
mean_mse = sum(mse_scores) / len(mse_scores)
# mean_mad = sum(smooth_mse_scores) / len(smooth_mse_scores)

print(f"\n✅ Mean SSIM (Fake vs Ground Truth) over {len(ssim_scores)} images: {mean_ssim:.4f}")
print(f"📉 Mean MSE  over {len(mse_scores)} images: {mean_mse:.6f}")
# print(f"📉 Mean smooth MAD: {mean_mad:.4f}")

