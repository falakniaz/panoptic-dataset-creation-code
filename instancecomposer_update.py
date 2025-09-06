#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Instances from Semantic (COCO Instances JSON)
=============================================
Converts a composed semantic dataset (image + class-ID mask pairs) into
a COCO "instances" dataset with per-instance RLE segmentations.

Input (from semantic composer):
  SEM_ROOT/
    input/<split>/<sub>/<mod>/*.jpg
    labels/<split>/<sub>/<mod>/*.png   # class-id masks (0 = background). 8/16-bit OK.

Output:
  OUT_ROOT/
    <split>/<mod>/*.jpg                # copied (or re-encoded) images
    <split>/annotations.json           # COCO, RLE segmentations

Highlights:
- Deterministic listing, progress bars
- Lossless ID-preserving mask reads (8/16-bit). RGB masks rejected unless mapped
- Optional remap of 255 -> <id> (255 is often 'ignore' in pipelines)
- Per-instance extraction via connected components (8-connectivity)
- Optional allow/ignore ID sets
- Size mismatch policy: error | resize | skip
- Optional category name map JSON (id->name)
"""

import os
import cv2
import json
import argparse
from typing import List, Dict, Optional, Set, Tuple
import numpy as np
from tqdm import tqdm
import shutil

try:
    from pycocotools import mask as maskUtils
except Exception:
    maskUtils = None

# ==== Defaults (edit if desired) ====
SEM_ROOT_DEFAULT       = "/home/falak/UAV_dataset/semantic_composed"
OUT_ROOT_DEFAULT       = "/home/falak/UAV_dataset/instances"
MIN_INSTANCE_DEFAULT   = 50        # min pixel area per instance
REMAP_255_TO_DEFAULT   = 1         # remap label 255 -> this id (set 255 to leave unchanged)
MODALITIES_DEFAULT     = "visible,infrared"
SIZE_MISMATCH_DEFAULT  = "error"   # 'error' | 'resize' | 'skip'
COPY_IMAGES_DEFAULT    = True      # copy (preserve quality) vs re-encode
JPEG_QUALITY_DEFAULT   = 95        # used only if COPY_IMAGES is False

# --------------------------
# Helpers
# --------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_images(folder: str) -> List[str]:
    """Deterministic listing of image filenames (not paths)."""
    if not os.path.isdir(folder):
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(
        f.name for f in os.scandir(folder)
        if f.is_file() and os.path.splitext(f.name)[1].lower() in exts
    )

def parse_modalities(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def parse_int_list(s: Optional[str]) -> Optional[Set[int]]:
    if not s:
        return None
    items = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        items.append(int(tok))
    return set(items) if items else None

def mask_candidates(base_name: str) -> List[str]:
    """Possible mask filenames for an image basename (no extension)."""
    # semantic composer saved masks as <base>.png; keep _mask for flexibility
    return [f"{base_name}_mask.png", f"{base_name}.png"]

def read_mask_id(path: str) -> np.ndarray:
    """
    Read mask preserving ID values.
    Accepts single-channel 8/16-bit PNGs. If RGB (3ch), raise (safer).
    """
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise ValueError(f"Failed to read mask: {path}")
    if m.ndim == 2:
        # 8-bit or 16-bit single-channel: OK
        if m.dtype not in (np.uint8, np.uint16):
            # cast unusual integer types to uint16/8
            if m.max() <= 255:
                m = m.astype(np.uint8)
            else:
                m = m.astype(np.uint16)
        return m
    if m.ndim == 3 and m.shape[2] == 3:
        # This looks like a color-coded mask; better to convert explicitly with a mapping step upstream.
        raise ValueError(f"RGB/color-coded mask detected (3 channels): {path}")
    raise ValueError(f"Unsupported mask shape {m.shape} for: {path}")

def connected_components(bin_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """8-connectivity connected components using OpenCV. Returns (labels, count)."""
    num, labels = cv2.connectedComponents(bin_mask.astype(np.uint8), connectivity=8)
    # OpenCV returns num = number of labels INCLUDING background (label 0)
    return labels, num - 1

def encode_rle(bin_mask: np.ndarray) -> Dict:
    rle = maskUtils.encode(np.asfortranarray(bin_mask.astype(np.uint8)))
    if isinstance(rle["counts"], (bytes, bytearray)):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle

def build_categories(used_ids: Set[int], categories_map: Optional[Dict[int, str]]) -> List[Dict]:
    cats = []
    for cid in sorted(x for x in used_ids if x != 0):
        name = categories_map.get(cid, f"class_{cid}") if categories_map else ( "object" if cid == 1 else f"class_{cid}" )
        cats.append({"id": int(cid), "name": name, "supercategory": name})
    return cats

def load_categories_map(json_path: Optional[str]) -> Optional[Dict[int, str]]:
    if not json_path:
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Supports either {"1":"car", "2":"person"} or {"categories":[{"id":1,"name":"car"},...]}
    if isinstance(raw, dict) and "categories" in raw and isinstance(raw["categories"], list):
        return {int(c["id"]): str(c["name"]) for c in raw["categories"]}
    # else assume simple id->name dict (keys may be str)
    return {int(k): str(v) for k, v in raw.items()}

# --------------------------
# Main conversion
# --------------------------
def main(
    sem_root: str,
    out_root: str,
    min_instance: int,
    remap_255_to: int,
    modalities: str,
    size_mismatch_policy: str,
    copy_images: bool,
    jpeg_quality: int,
    allowed_ids_str: Optional[str],
    ignore_ids_str: Optional[str],
    categories_map_json: Optional[str]
):
    if maskUtils is None:
        raise SystemExit("pycocotools is required: pip install pycocotools")

    ensure_dir(out_root)
    splits = [s for s in ["train", "val", "test"] if os.path.isdir(os.path.join(sem_root, "input", s))]
    if not splits:
        raise SystemExit("[ERROR] No splits found under SEM_ROOT/input/{train,val,test}")

    mods = parse_modalities(modalities)
    if not mods:
        raise SystemExit("[ERROR] No modalities specified.")

    allowed_ids = parse_int_list(allowed_ids_str)
    ignore_ids  = parse_int_list(ignore_ids_str)
    cat_map     = load_categories_map(categories_map_json)

    for split in splits:
        # fresh COCO dict for this split
        coco = {
            "info": {"description": f"Instances {split}", "version": "1.0", "year": 2025},
            "licenses": [],
            "categories": [],
            "images": [],
            "annotations": []
        }
        img_id = 0
        ann_id = 0
        used_category_ids: Set[int] = set()

        split_input_root = os.path.join(sem_root, "input", split)
        subfolders = sorted(d for d in os.listdir(split_input_root)
                            if os.path.isdir(os.path.join(split_input_root, d)))

        total_images = 0
        total_instances = 0
        skipped_due_to_mask = 0
        skipped_size_mismatch = 0

        for mod in mods:
            ensure_dir(os.path.join(out_root, split, mod))
            for sub in tqdm(subfolders, desc=f"[instances] {split}:{mod}"):
                in_dir  = os.path.join(sem_root, "input",  split, sub, mod)
                lab_dir = os.path.join(sem_root, "labels", split, sub, mod)
                if not (os.path.isdir(in_dir) and os.path.isdir(lab_dir)):
                    continue

                for fname in list_images(in_dir):
                    img_path = os.path.join(in_dir, fname)
                    base = os.path.splitext(fname)[0]
                    # find mask
                    mask_path = None
                    for cand in mask_candidates(base):
                        p = os.path.join(lab_dir, cand)
                        if os.path.isfile(p):
                            mask_path = p
                            break
                    if mask_path is None:
                        continue

                    # read
                    img = cv2.imread(img_path)
                    try:
                        mask = read_mask_id(mask_path)
                    except Exception as e:
                        skipped_due_to_mask += 1
                        continue
                    if img is None:
                        continue

                    H, W = img.shape[:2]
                    mH, mW = mask.shape[:2]
                    if (H, W) != (mH, mW):
                        policy = size_mismatch_policy.lower()
                        if policy == "error":
                            raise RuntimeError(f"Size mismatch at {img_path} vs {mask_path}: {(H,W)} != {(mH,mW)}")
                        elif policy == "resize":
                            # nearest for mask (labels), bilinear ok for image (visuals)
                            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                        elif policy == "skip":
                            skipped_size_mismatch += 1
                            continue
                        else:
                            raise ValueError(f"Unknown size_mismatch_policy: {size_mismatch_policy}")

                    # remap 255 if requested
                    if remap_255_to != 255 and (mask == 255).any():
                        mask = mask.copy()
                        mask[mask == 255] = remap_255_to

                    # normalize dtype for safety (keep >=256 IDs if any)
                    if mask.max() <= 255 and mask.dtype != np.uint8:
                        mask = mask.astype(np.uint8)

                    # Write image (copy or encode)
                    out_name = f"{sub}_{fname}".replace("/", "_").replace("\\", "_")
                    dst_img = os.path.join(out_root, split, mod, out_name)
                    if copy_images:
                        try:
                            shutil.copy2(img_path, dst_img)
                        except Exception:
                            # fallback to re-encode
                            cv2.imwrite(dst_img, img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                    else:
                        cv2.imwrite(dst_img, img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

                    img_id += 1
                    total_images += 1
                    coco["images"].append({
                        "id": img_id,
                        "file_name": f"{mod}/{out_name}",
                        "width": int(W),
                        "height": int(H),
                    })

                    # extract per-class instances
                    uniq = np.unique(mask)
                    # Drop background 0
                    uniq = uniq[uniq != 0]
                    # Apply allow/ignore filters
                    if allowed_ids is not None:
                        uniq = np.array([u for u in uniq if int(u) in allowed_ids], dtype=uniq.dtype)
                    if ignore_ids is not None:
                        uniq = np.array([u for u in uniq if int(u) not in ignore_ids], dtype=uniq.dtype)

                    for cid in uniq.tolist():
                        binm = (mask == cid).astype(np.uint8)
                        labels, count = connected_components(binm)
                        if count == 0:
                            continue
                        for inst_label in range(1, count + 1):
                            inst = (labels == inst_label).astype(np.uint8)
                            area = int(inst.sum())
                            if area < min_instance:
                                continue
                            rle = encode_rle(inst)
                            bbox = [float(x) for x in maskUtils.toBbox(rle).tolist()]
                            coco["annotations"].append({
                                "id": int(ann_id := ann_id + 1),
                                "image_id": int(img_id),
                                "category_id": int(cid),
                                "segmentation": rle,
                                "area": float(maskUtils.area(rle)),
                                "bbox": bbox,                 # [x, y, w, h]
                                "iscrowd": 0
                            })
                            total_instances += 1
                        used_category_ids.add(int(cid))

        # fill categories
        coco["categories"] = build_categories(used_category_ids, cat_map)

        # write split json
        split_dir = os.path.join(out_root, split)
        ensure_dir(split_dir)
        ann_path = os.path.join(split_dir, "annotations.json")
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False)
        print(f"[OK] {split}: images={total_images}, instances={total_instances}, "
              f"cats={len(used_category_ids)}, skipped_mask={skipped_due_to_mask}, "
              f"skipped_size_mismatch={skipped_size_mismatch}, -> {ann_path}")

# --------------------------
# CLI
# --------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem_root", default=SEM_ROOT_DEFAULT)
    ap.add_argument("--out_root", default=OUT_ROOT_DEFAULT)
    ap.add_argument("--min_instance", type=int, default=MIN_INSTANCE_DEFAULT,
                    help="Minimum pixel area per instance.")
    ap.add_argument("--remap_255_to", type=int, default=REMAP_255_TO_DEFAULT,
                    help="Remap label 255 to this id (255 means keep as 255).")
    ap.add_argument("--modalities", default=MODALITIES_DEFAULT,
                    help='Comma-separated: e.g. "visible,infrared" or "visible".')
    ap.add_argument("--size_mismatch_policy", choices=["error", "resize", "skip"], default=SIZE_MISMATCH_DEFAULT,
                    help="How to handle image/mask size mismatch.")
    ap.add_argument("--copy_images", action="store_true", default=COPY_IMAGES_DEFAULT,
                    help="Copy images instead of re-encoding (preserves quality).")
    ap.add_argument("--no_copy_images", action="store_false", dest="copy_images",
                    help="Re-encode images (JPEG) instead of copying.")
    ap.add_argument("--jpeg_quality", type=int, default=JPEG_QUALITY_DEFAULT,
                    help="JPEG quality (if re-encoding).")
    ap.add_argument("--allowed_ids", default=None,
                    help="Comma-separated list of class IDs to include. Others dropped.")
    ap.add_argument("--ignore_ids", default=None,
                    help="Comma-separated list of class IDs to exclude.")
    ap.add_argument("--categories_map_json", default=None,
                    help="Optional JSON with id->name mapping or COCO-style categories list.")
    args = ap.parse_args()

    main(
        sem_root=args.sem_root,
        out_root=args.out_root,
        min_instance=args.min_instance,
        remap_255_to=args.remap_255_to,
        modalities=args.modalities,
        size_mismatch_policy=args.size_mismatch_policy,
        copy_images=args.copy_images,
        jpeg_quality=args.jpeg_quality,
        allowed_ids_str=args.allowed_ids,
        ignore_ids_str=args.ignore_ids,
        categories_map_json=args.categories_map_json,
    )
