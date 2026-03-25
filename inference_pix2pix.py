import os
import torch
from torchvision import transforms
from PIL import Image, TiffImagePlugin
from tqdm import tqdm
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse
from typing import Optional

# ==========================
# CONFIGURATION
# ==========================
input_list = "/remote/tnas1/algo/koren/changi_3_3/list.txt"
input_folder = "/remote/tnas1/algo/koren/changi_3_3/brightened_w_pi"
output_folder = "/remote/tnas1/algo/koren/changi_3_3/pix2pix_w_pi"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your trained generator
from models.networks import define_G  # adjust import to your pix2pix repo structure


checkpoint_path = "/home/korenbe/repos/pytorch-CycleGAN-and-pix2pix_research/checkpoints/v19/latest_net_G.pth"

# ==========================
# PREPROCESSING & POSTPROCESSING
# ==========================
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# We’ll normalize the same way as in training
normalize = transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))

from PIL import Image


def bottom_right_crop(img, target_w, target_h):
    return img.crop((0, 0, target_w, target_h))


def center_crop(img, target_w, target_h):
    w, h = img.size

    left = (w - target_w) // 2
    top = (h - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    return img.crop((left, top, right, bottom))

def preprocess(img):
    w, h = img.size
    target_w = int(np.ceil(w / 256) * 256)
    target_h = int(np.ceil(h / 256) * 256)
    pad_w1 = max(target_w - w, 0) // 2
    pad_h1 = max(target_h - h, 0) // 2
    img = TF.pad(img, (pad_w1, pad_h1, target_w - w - pad_w1, target_h-h-pad_h1), fill=(255, 255, 255))  # left, top, right, bottom
    return normalize(to_tensor(img)).unsqueeze(0).to(device)


def postprocess(tensor, org_w, org_h):
    """Convert model output [-1,1] → uint8 PIL"""
    img = (tensor.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
    pil_img = to_pil(img)
    trimmed = center_crop(pil_img, org_w, org_h)  # or whatever size you want
    return trimmed



def _load_generator(checkpoint_path_in: str):
    netG = define_G(
        input_nc=3,
        output_nc=3,
        ngf=128,
        netG="unet_256",
        norm="instance",
        predict_delta=True,
        use_dropout=True,
        use_upsample=False,
        use_modulated_conv=False,
    )
    netG.load_state_dict(torch.load(checkpoint_path_in, map_location=device))
    netG.to(device)
    netG.eval()
    return netG


def run_pix2pix_inference(
    input_folder_in: str,
    output_folder_in: str,
    input_list_in: Optional[str] = None,
    checkpoint_path_in: str = checkpoint_path,
) -> str:
    global input_folder, output_folder, input_list

    input_folder = input_folder_in
    output_folder = output_folder_in
    input_list = input_list_in

    netG = _load_generator(checkpoint_path_in)

    os.makedirs(output_folder, exist_ok=True)
    images_list = []

    if input_list:
        with open(input_list, "r") as f:
            images_list_full = f.read().splitlines()
            images_list = tqdm(sorted([os.path.basename(img) for img in images_list_full]))
    else:
        images_list = tqdm(sorted(os.listdir(input_folder)))

    with torch.no_grad():
        for fname in images_list:
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                continue

            imgA = Image.open(os.path.join(input_folder, fname)).convert("RGB")
            w, h = imgA.size
            input_tensor = preprocess(imgA)
            output_tensor = netG(input_tensor)
            output_img = postprocess(output_tensor, w, h)

            output_img.save(os.path.join(output_folder, fname))

    return output_folder


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", default=input_folder)
    parser.add_argument("--output-folder", default=output_folder)
    parser.add_argument("--input-list", default=input_list)
    parser.add_argument("--checkpoint-path", default=checkpoint_path)
    args = parser.parse_args(argv)

    run_pix2pix_inference(
        input_folder_in=args.input_folder,
        output_folder_in=args.output_folder,
        input_list_in=args.input_list,
        checkpoint_path_in=args.checkpoint_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

