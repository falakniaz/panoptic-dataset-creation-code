#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Composer (single-object → combined composites)
=====================================================
Input (per split):
  ORIGINAL_ROOT/
    input/<train|val|test>/<sub>/<visible|infrared>/*.jpg
    labels/<train|val|test>/<sub>/<visible|infrared>/*_mask.png  (or *.png)

Output:
  SEMANTIC_OUT/
    input/<split>/<sub>/<mod>/*.jpg          (combined images)
    labels/<split>/<sub>/<mod>/*.png         (combined masks, class-id)

Notes:
- Uses **bilinear** for images, **nearest** for masks (to preserve labels).
- Tries both *_mask.png and just *.png when finding masks.
- Randomly combines 2–5 images from the same <sub>/<mod>, resized to target size.

You can run it without CLI (paths below) or override via arguments.
"""

import os
import cv2
import sys
import json
import random
import argparse
from typing import List
import numpy as np
from tqdm import tqdm

# === PATHS (EDIT THESE) ===
ORIGINAL_ROOT_DEFAULT = "/home/falak/UAV_dataset/combined_uav"
SEMANTIC_OUT_DEFAULT  = "/home/falak/UAV_dataset/semantic_composed"
TARGET_W_DEFAULT = 1920
TARGET_H_DEFAULT = 1080
PER_SUBFOLDER_DEFAULT = 50
SEED_DEFAULT = 0


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts]


def img_to_mask_candidates(img_name: str) -> List[str]:
    base = os.path.splitext(img_name)[0]
    return [f"{base}_mask.png", f"{base}.png"]


def combine_images_fixed_size(image_dir: str, image_files: List[str], target_w: int, target_h: int) -> np.ndarray:
    imgs = []
    for f in image_files:
        p = os.path.join(image_dir, f)
        im = cv2.imread(p)
        if im is None:
            return None
        h, w = im.shape[:2]
        new_w = max(1, int(round(w * (target_h / float(h)))))
        im = cv2.resize(im, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
        imgs.append(im)
    if not imgs:
        return None
    combined = np.hstack(imgs)
    combined = cv2.resize(combined, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return combined


def combine_masks_fixed_size(mask_dir: str, mask_files: List[str], target_w: int, target_h: int) -> np.ndarray:
    ms = []
    for f in mask_files:
        p = os.path.join(mask_dir, f)
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        h, w = m.shape[:2]
        new_w = max(1, int(round(w * (target_h / float(h)))))
        m = cv2.resize(m, (new_w, target_h), interpolation=cv2.INTER_NEAREST)
        ms.append(m)
    if not ms:
        return None
    combined = np.hstack(ms)
    combined = cv2.resize(combined, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return combined


def main(original_root: str, semantic_out: str, per_subfolder: int, target_w: int, target_h: int, seed: int):
    random.seed(seed)
    np.random.seed(seed)

    ensure_dir(semantic_out)
    splits = [s for s in ["train","val","test"] if os.path.isdir(os.path.join(original_root, "input", s))]
    if not splits:
        print("[ERROR] no splits found under original_root/input/")
        return

    for split in splits:
        split_path = os.path.join(original_root, "input", split)
        subfolders = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        for sub in tqdm(subfolders, desc=f"[compose] {split}"):
            for mod in ["visible","infrared"]:
                in_dir  = os.path.join(original_root, "input", split, sub, mod)
                lab_dir = os.path.join(original_root, "labels", split, sub, mod)
                if not (os.path.isdir(in_dir) and os.path.isdir(lab_dir)):
                    continue
                imgs = list_images(in_dir)
                # filter to those with masks present
                pairs = []
                for im in imgs:
                    for cand in img_to_mask_candidates(im):
                        if os.path.isfile(os.path.join(lab_dir, cand)):
                            pairs.append((im, cand))
                            break
                if len(pairs) < 2:
                    continue
                out_img_dir = os.path.join(semantic_out, "input", split, sub, mod)
                out_lab_dir = os.path.join(semantic_out, "labels", split, sub, mod)
                ensure_dir(out_img_dir); ensure_dir(out_lab_dir)

                success = 0; attempts = 0; max_attempts = per_subfolder * 6
                while success < per_subfolder and attempts < max_attempts:
                    attempts += 1
                    k = random.randint(2, min(5, len(pairs)))
                    sel = random.sample(pairs, k)
                    sel_imgs = [a for a, _ in sel]
                    sel_masks= [b for _, b in sel]
                    C = combine_images_fixed_size(in_dir, sel_imgs, target_w, target_h)
                    M = combine_masks_fixed_size(lab_dir, sel_masks, target_w, target_h)
                    if C is None or M is None:
                        continue
                    base = f"combined_{k}_{success+1}"
                    cv2.imwrite(os.path.join(out_img_dir, base+".jpg"), C)
                    cv2.imwrite(os.path.join(out_lab_dir, base+".png"), M)
                    success += 1

    print("[OK] Semantic composition finished ->", semantic_out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--original_root", default=ORIGINAL_ROOT_DEFAULT)
    ap.add_argument("--out_root", default=SEMANTIC_OUT_DEFAULT)
    ap.add_argument("--per_subfolder", type=int, default=PER_SUBFOLDER_DEFAULT)
    ap.add_argument("--target_w", type=int, default=TARGET_W_DEFAULT)
    ap.add_argument("--target_h", type=int, default=TARGET_H_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = ap.parse_args()
    main(args.original_root, args.out_root, args.per_subfolder, args.target_w, args.target_h, args.seed)
