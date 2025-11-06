import os
import torch
from torchvision import transforms
from PIL import Image
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as SSIM
from tqdm import tqdm

# ==========================
# CONFIGURATION
# ==========================
input_folder = "/remote/tnas1/algo/koren/dfw-pix2pix-dc/raw/trainA_pre_crop_crops/"
gt_folder    = "/remote/tnas1/algo/koren/dfw-pix2pix-dc/raw/trainB_pre_crop_crops/"

output_folder = "/remote/tnas1/algo/koren/dfw-pix2pix-dc/results_resnet9_instance_delta_10_512"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your trained generator
from models.networks import define_G  # adjust import to your pix2pix repo structure


netG = define_G(input_nc=3, output_nc=3, ngf=64, netG='resnet_9blocks', norm='none',
                predict_delta=True, use_dropout=True)
checkpoint_path = "/home/korenbe/repos/pytorch-CycleGAN-and-pix2pix_research/checkpoints/none_512_2-4_10_resnet_64_4_delta/latest_net_G.pth"
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


def preprocess(img):
    """Convert PIL → Tensor in [-1,1] range"""
    return normalize(to_tensor(img)).unsqueeze(0).to(device)


def postprocess(tensor):
    """Convert model output [-1,1] → uint8 PIL"""
    img = (tensor.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
    return to_pil(img)


# ==========================
# EVALUATION LOOP
# ==========================
os.makedirs(output_folder, exist_ok=True)
ssim_metric = SSIM(data_range=1.0).to(device)
ssim_scores = []

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

        # Prepare tensors
        input_tensor = preprocess(imgA)
        output_tensor = netG(input_tensor)
        output_img = postprocess(output_tensor)
        output_img.save(os.path.join(output_folder, fname))

        # --- SSIM against ground truth ---
        gt_tensor = to_tensor(imgB).unsqueeze(0).to(device)
        output_for_metric = to_tensor(output_img).unsqueeze(0).to(device)

        ssim_value = ssim_metric(output_for_metric, gt_tensor)
        ssim_scores.append(ssim_value.item())

# ==========================
# RESULTS
# ==========================
mean_ssim = sum(ssim_scores) / len(ssim_scores)
print(f"\n✅ Mean SSIM (Fake vs Ground Truth) over {len(ssim_scores)} images: {mean_ssim:.4f}")

