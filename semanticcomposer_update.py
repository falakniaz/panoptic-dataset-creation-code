#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Composer (single-object → combined composites)
=====================================================
Creates composite samples by horizontally stitching 2–5 images (and masks)
from the same subfolder + modality (visible/infrared). Outputs standardized
image/mask pairs suitable for common CV pipelines.

Input (per split):
  ORIGINAL_ROOT/
    input/<train|val|test>/<sub>/<visible|infrared>/*.(jpg|jpeg|png|bmp)
    labels/<train|val|test>/<sub>/<visible|infrared>/*_mask.png  (or *.png)
      - By default, ONLY PNG masks are allowed (lossless, ID-safe).
      - You can enable JPG masks with --allow_jpg_masks (NOT recommended).

Output:
  OUT_ROOT/
    input/<split>/<sub>/<mod>/*.jpg          (composed images)
    labels/<split>/<sub>/<mod>/*.png         (composed masks, class-id)

Notes:
- Images: bilinear interpolation for resizing.
- Masks: nearest neighbor interpolation to preserve class IDs.
- Deterministic listing + --seed for reproducible sampling.
- Optional no-distortion mode via --fit_mode padcrop (default: resize).
- Optional manifest JSONL with provenance of each composite (--manifest).
- Optional color-coded mask conversion via --color_to_id_json mapping.

Example:
python3 semantic_composer.py \
  --original_root /home/falak/UAV_dataset/combined_uav \
  --out_root /home/falak/UAV_dataset/semantic_composed \
  --per_subfolder 30 --target_w 1280 --target_h 720 \
  --fit_mode padcrop --seed 42 \
  --manifest /home/falak/UAV_dataset/semantic_composed/manifest.jsonl
"""

import os
import cv2
import json
import random
import argparse
from typing import List, Optional, Dict, Tuple
import numpy as np
from tqdm import tqdm

# === DEFAULTS (EDIT IF NEEDED) ===
ORIGINAL_ROOT_DEFAULT = "/home/falak/UAV_dataset/combined_uav"
OUT_ROOT_DEFAULT       = "/home/falak/UAV_dataset/semantic_composed"
TARGET_W_DEFAULT = 1920
TARGET_H_DEFAULT = 1080
PER_SUBFOLDER_DEFAULT = 50
SEED_DEFAULT = 0
MODALITIES_DEFAULT = "visible,infrared"  # comma-separated

# ---------------------------
# Utilities
# ---------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_images(folder: str) -> List[str]:
    """Deterministic listing of image filenames (not paths)."""
    if not os.path.isdir(folder):
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(
        f.name
        for f in os.scandir(folder)
        if f.is_file() and os.path.splitext(f.name)[1].lower() in exts
    )

def img_to_mask_candidates(img_name: str, allow_jpg_masks: bool = False) -> List[str]:
    """Return possible mask filenames for a given image base."""
    base = os.path.splitext(img_name)[0]
    cands = [f"{base}_mask.png", f"{base}.png"]
    if allow_jpg_masks:
        cands += [f"{base}_mask.jpg", f"{base}.jpg"]
    return cands

def parse_modalities(mod_string: str) -> List[str]:
    return [m.strip() for m in mod_string.split(",") if m.strip()]

# ---------------------------
# Mask reading / color→ID
# ---------------------------

def load_color_to_id_map(json_path: str) -> Dict[Tuple[int, int, int], int]:
    """
    JSON format: { "r,g,b": id, ... } e.g., { "0,0,0": 0, "255,0,0": 1 }
    """
    with open(json_path, "r") as f:
        raw = json.load(f)
    mapping: Dict[Tuple[int, int, int], int] = {}
    for k, v in raw.items():
        parts = [int(x) for x in k.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid color key '{k}' in {json_path}; expected 'r,g,b'.")
        mapping[(parts[0], parts[1], parts[2])] = int(v)
    return mapping

def convert_color_mask_to_ids(color_mask: np.ndarray, color_to_id: Dict[Tuple[int,int,int], int]) -> np.ndarray:
    """
    Convert a 3-channel color-coded mask to single-channel ID mask using mapping.
    color_mask: HxWx3 (BGR as read by OpenCV)
    mapping keys are RGB; convert accordingly.
    """
    if color_mask.ndim != 3 or color_mask.shape[2] != 3:
        raise ValueError("Expected 3-channel color mask for conversion.")
    # OpenCV loads as BGR; convert to RGB tuples for mapping
    b, g, r = color_mask[...,0], color_mask[...,1], color_mask[...,2]
    h, w = color_mask.shape[:2]
    out = np.zeros((h, w), dtype=np.uint16)  # use uint16 to be safe during mapping

    # Build a reverse lookup per unique color to avoid O(HW * K)
    # Find unique colors present
    stacked = np.stack([r, g, b], axis=-1)  # RGB order
    uniq = np.unique(stacked.reshape(-1, 3), axis=0)
    for rgb in uniq:
        rgb_t = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        if rgb_t not in color_to_id:
            raise ValueError(f"Color {rgb_t} not in provided color_to_id map.")
        mask = (r == rgb[0]) & (g == rgb[1]) & (b == rgb[2])
        out[mask] = color_to_id[rgb_t]

    # If IDs fit in uint8, cast down (common case)
    if out.max() <= 255:
        out = out.astype(np.uint8)
    return out

def read_mask_id(path: str, color_to_id: Optional[Dict[Tuple[int,int,int], int]] = None) -> np.ndarray:
    """
    Read a mask as integer ID map.
    - If single-channel -> ensure uint8.
    - If 3-channel and color_to_id provided -> convert.
    - If 3-channel and no mapping -> raise (avoid silent mistakes).
    """
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 2:
        # Single-channel; ensure uint8 (or keep uint16 if already that)
        if m.dtype == np.uint16:
            return m  # OK (PNG 16-bit)
        if m.dtype != np.uint8:
            m = m.astype(np.uint8)
        return m
    elif m.ndim == 3 and m.shape[2] == 3:
        if color_to_id is None:
            raise ValueError(
                f"Mask appears color-coded (3ch) but no --color_to_id_json provided: {path}"
            )
        return convert_color_mask_to_ids(m, color_to_id)
    else:
        raise ValueError(f"Unsupported mask shape {m.shape} at {path}")

# ---------------------------
# Compose helpers
# ---------------------------

def fit_to_size_no_distortion(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    Height is assumed already == target_h.
    If width < target_w: right-pad with zeros.
    If width > target_w: center-crop.
    Works for HxW (mask) and HxWxC (image).
    """
    h, w = img.shape[:2]
    if h != target_h:
        raise ValueError(f"fit_to_size_no_distortion expects h==target_h, got {h} vs {target_h}")
    if w == target_w:
        return img

    if w < target_w:
        pad = target_w - w
        if img.ndim == 3:
            right = np.zeros((h, pad, img.shape[2]), dtype=img.dtype)
        else:
            right = np.zeros((h, pad), dtype=img.dtype)
        return np.hstack([img, right])
    else:
        start = (w - target_w) // 2
        return img[:, start:start+target_w]  # center crop

def combine_images_fixed_size(
    image_dir: str,
    image_files: List[str],
    target_w: int,
    target_h: int,
    fit_mode: str = "resize"
) -> Optional[np.ndarray]:
    """Resize each tile to target_h (keep aspect), hstack, then fit to (target_w, target_h)."""
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
    combined = cv2.hconcat(imgs)
    if fit_mode == "padcrop":
        combined = fit_to_size_no_distortion(combined, target_w, target_h)
    else:
        combined = cv2.resize(combined, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return combined

def combine_masks_fixed_size(
    mask_dir: str,
    mask_files: List[str],
    target_w: int,
    target_h: int,
    fit_mode: str = "resize",
    color_to_id: Optional[Dict[Tuple[int,int,int], int]] = None
) -> Optional[np.ndarray]:
    """Same as images, but read as ID maps and use NEAREST for all resizes."""
    ms = []
    for f in mask_files:
        p = os.path.join(mask_dir, f)
        try:
            m = read_mask_id(p, color_to_id=color_to_id)
        except Exception as e:
            print(f"[WARN] Mask read/convert failed: {p} -> {e}")
            return None
        if m is None:
            return None
        h, w = m.shape[:2]
        new_w = max(1, int(round(w * (target_h / float(h)))))
        m = cv2.resize(m, (new_w, target_h), interpolation=cv2.INTER_NEAREST)
        ms.append(m)
    if not ms:
        return None
    combined = cv2.hconcat(ms)
    if fit_mode == "padcrop":
        combined = fit_to_size_no_distortion(combined, target_w, target_h)
    else:
        combined = cv2.resize(combined, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    # Ensure integer dtype for masks
    if combined.dtype == np.float32 or combined.dtype == np.float64:
        combined = combined.astype(np.uint16 if combined.max() > 255 else np.uint8)
    return combined

# ---------------------------
# Main
# ---------------------------

def main(
    original_root: str,
    out_root: str,
    per_subfolder: int,
    target_w: int,
    target_h: int,
    seed: int,
    fit_mode: str,
    manifest_path: Optional[str],
    modalities_str: str,
    allow_jpg_masks: bool,
    color_to_id_json: Optional[str],
):
    random.seed(seed)
    np.random.seed(seed)

    ensure_dir(out_root)
    splits = [s for s in ["train","val","test"] if os.path.isdir(os.path.join(original_root, "input", s))]
    if not splits:
        print("[ERROR] no splits found under original_root/input/")
        return

    modalities = parse_modalities(modalities_str)
    if not modalities:
        print("[ERROR] No modalities provided.")
        return

    color_to_id = None
    if color_to_id_json:
        color_to_id = load_color_to_id_map(color_to_id_json)

    manifest_rows = []
    total_success = 0
    skipped_subs = 0

    for split in splits:
        split_path = os.path.join(original_root, "input", split)
        # deterministic
        subfolders = sorted(
            d for d in os.listdir(split_path)
            if os.path.isdir(os.path.join(split_path, d))
        )

        for sub in tqdm(subfolders, desc=f"[compose] {split}"):
            any_pairs = False
            for mod in modalities:
                in_dir  = os.path.join(original_root, "input", split, sub, mod)
                lab_dir = os.path.join(original_root, "labels", split, sub, mod)
                if not (os.path.isdir(in_dir) and os.path.isdir(lab_dir)):
                    continue

                imgs = list_images(in_dir)
                pairs = []
                for im in imgs:
                    for cand in img_to_mask_candidates(im, allow_jpg_masks=allow_jpg_masks):
                        if os.path.isfile(os.path.join(lab_dir, cand)):
                            pairs.append((im, cand))
                            break
                if len(pairs) < 2:
                    continue
                any_pairs = True

                out_img_dir = os.path.join(out_root, "input", split, sub, mod)
                out_lab_dir = os.path.join(out_root, "labels", split, sub, mod)
                ensure_dir(out_img_dir); ensure_dir(out_lab_dir)

                success = 0
                attempts = 0
                max_attempts = per_subfolder * 6

                used_sets = set()  # avoid exact duplicate combinations
                while success < per_subfolder and attempts < max_attempts:
                    attempts += 1
                    k = random.randint(2, min(5, len(pairs)))
                    sel = tuple(sorted(random.sample(range(len(pairs)), k)))
                    if sel in used_sets:
                        continue
                    used_sets.add(sel)
                    sel_imgs = [pairs[i][0] for i in sel]
                    sel_masks= [pairs[i][1] for i in sel]

                    C = combine_images_fixed_size(in_dir, sel_imgs, target_w, target_h, fit_mode=fit_mode)
                    M = combine_masks_fixed_size(lab_dir, sel_masks, target_w, target_h, fit_mode=fit_mode, color_to_id=color_to_id)
                    if C is None or M is None:
                        continue

                    base = f"{split}_{sub}_{mod}_k{k}_{success+1:05d}"
                    img_out = os.path.join(out_img_dir, base + ".jpg")
                    mask_out = os.path.join(out_lab_dir, base + ".png")

                    # Write with explicit compression params
                    cv2.imwrite(img_out, C, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    # If mask is uint16, PNG will keep it; otherwise uint8 is standard.
                    png_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
                    cv2.imwrite(mask_out, M, png_params)

                    manifest_rows.append({
                        "split": split, "sub": sub, "mod": mod, "k": k,
                        "image": img_out, "mask": mask_out,
                        "components": [{"img": a, "mask": b} for a, b in (pairs[i] for i in sel)],
                        "fit_mode": fit_mode,
                    })
                    success += 1
                    total_success += 1

            if not any_pairs:
                skipped_subs += 1

    # Write manifest if requested
    if manifest_path:
        ensure_dir(os.path.dirname(os.path.abspath(manifest_path)))
        with open(manifest_path, "w", encoding="utf-8") as f:
            for r in manifest_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    print(f"[OK] Semantic composition finished -> {out_root}")
    print(f"     Total composites: {total_success}")
    print(f"     Subfolders with no valid pairs: {skipped_subs}")
    if manifest_path:
        print(f"     Manifest: {manifest_path} (rows: {len(manifest_rows)})")

# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--original_root", default=ORIGINAL_ROOT_DEFAULT)
    ap.add_argument("--out_root", default=OUT_ROOT_DEFAULT)
    ap.add_argument("--per_subfolder", type=int, default=PER_SUBFOLDER_DEFAULT)
    ap.add_argument("--target_w", type=int, default=TARGET_W_DEFAULT)
    ap.add_argument("--target_h", type=int, default=TARGET_H_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--fit_mode", choices=["resize","padcrop"], default="resize",
                    help="Final fit of the hstack to (W,H): resize (default) or no-distortion pad/crop.")
    ap.add_argument("--manifest", default=None, help="Write JSONL manifest with provenance.")
    ap.add_argument("--modalities", default=MODALITIES_DEFAULT,
                    help='Comma-separated list, e.g. "visible,infrared" or just "visible".')
    ap.add_argument("--allow_jpg_masks", action="store_true",
                    help="Allow JPG masks (NOT recommended; can corrupt IDs).")
    ap.add_argument("--color_to_id_json", default=None,
                    help="Optional JSON mapping for color-coded masks {\"r,g,b\": id}.")
    args = ap.parse_args()

    main(
        original_root=args.original_root,
        out_root=args.out_root,
        per_subfolder=args.per_subfolder,
        target_w=args.target_w,
        target_h=args.target_h,
        seed=args.seed,
        fit_mode=args.fit_mode,
        manifest_path=args.manifest,
        modalities_str=args.modalities,
        allow_jpg_masks=args.allow_jpg_masks,
        color_to_id_json=args.color_to_id_json,
    )
