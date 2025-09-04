#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Instances from Semantic (COCO Instances JSON)
===========================================
Takes the composed semantic dataset and produces a COCO **instances** dataset.

Input (from semantic_composer.py):
  SEM_ROOT/
    input/<split>/<sub>/<mod>/*.jpg
    labels/<split>/<sub>/<mod>/*.png   (class-id masks; 0=background, 255 remapped)

Output:
  INST_OUT/
    <split>/<mod>/*.jpg                (copied images)
    <split>/annotations.json           (COCO, RLE segmentations)

Safety:
- Remaps class id **255 → 1** (safer; 255 is often treated as ignore).
- Background (0) is not included as a category.
- Uses pycocotools RLE with ASCII counts.
"""

import os
import cv2
import sys
import json
import argparse
from typing import List, Dict
import numpy as np
from tqdm import tqdm
from skimage import measure

try:
    from pycocotools import mask as maskUtils
except Exception:
    maskUtils = None

# === PATHS (EDIT THESE) ===
SEM_ROOT_DEFAULT  = "/home/falak/UAV_dataset/semantic_composed"
INST_OUT_DEFAULT  = "/home/falak/UAV_dataset/instances"
MIN_INSTANCE_DEFAULT = 50
REMAP_255_TO_DEFAULT = 1


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts]


def extract_instances(mask: np.ndarray, min_size: int = 50) -> List[Dict]:
    out = []
    classes = [int(v) for v in np.unique(mask) if v != 0]
    for cid in classes:
        binm = (mask == cid).astype(np.uint8)
        lab, num = measure.label(binm, connectivity=2, return_num=True)
        for inst in range(1, num+1):
            m = (lab == inst).astype(np.uint8)
            if m.sum() < min_size:
                continue
            out.append({"class_id": cid, "mask": m})
    return out


def main(sem_root: str, inst_out: str, min_instance: int, remap_255_to: int):
    if maskUtils is None:
        raise SystemExit("pycocotools is required: pip install pycocotools")

    ensure_dir(inst_out)
    splits = [s for s in ["train","val","test"] if os.path.isdir(os.path.join(sem_root, "input", s))]

    coco = None
    for split in splits:
        coco = {
            "info": {"description": f"UAV Instances - {split}", "version": "1.0", "year": 2025},
            "licenses": [], "categories": [], "images": [], "annotations": []
        }
        img_id = 0; ann_id = 0
        for mod in ["visible","infrared"]:
            split_path = os.path.join(sem_root, "input", split)
            subfolders = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
            ensure_dir(os.path.join(inst_out, split, mod))
            for sub in tqdm(subfolders, desc=f"[instances] {split}:{mod}"):
                in_dir  = os.path.join(sem_root, "input", split, sub, mod)
                lab_dir = os.path.join(sem_root, "labels", split, sub, mod)
                if not (os.path.isdir(in_dir) and os.path.isdir(lab_dir)):
                    continue
                for fname in list_images(in_dir):
                    img_path = os.path.join(in_dir, fname)
                    base = os.path.splitext(fname)[0]
                    mask_path1 = os.path.join(lab_dir, base + "_mask.png")
                    mask_path2 = os.path.join(lab_dir, base + ".png")
                    mask_path = mask_path1 if os.path.isfile(mask_path1) else (mask_path2 if os.path.isfile(mask_path2) else "")
                    if not mask_path:
                        continue
                    im = cv2.imread(img_path)
                    m  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if im is None or m is None:
                        continue
                    H, W = im.shape[:2]
                    # remap unsafe 255 -> remap_255_to
                    if (m == 255).any() and remap_255_to != 255:
                        m = m.copy(); m[m == 255] = remap_255_to

                    img_id += 1
                    out_name = f"{sub}_{fname}".replace("/","_").replace("\\","_")
                    dst = os.path.join(inst_out, split, mod, out_name)
                    cv2.imwrite(dst, im)
                    coco["images"].append({"id": img_id, "file_name": f"{mod}/{out_name}", "width": W, "height": H})

                    instances = extract_instances(m, min_instance)
                    for inst in instances:
                        rle = maskUtils.encode(np.asfortranarray(inst["mask"]))
                        if isinstance(rle["counts"], (bytes, bytearray)):
                            rle["counts"] = rle["counts"].decode("ascii")
                        area = float(maskUtils.area(rle))
                        bbox = [float(x) for x in maskUtils.toBbox(rle).tolist()]
                        ann_id += 1
                        coco["annotations"].append({
                            "id": ann_id, "image_id": img_id,
                            "category_id": int(inst["class_id"]),
                            "segmentation": rle, "area": area, "bbox": bbox, "iscrowd": 0
                        })
        # categories from used ids
        used = sorted({a["category_id"] for a in coco["annotations"]})
        name_map = {1: "object"}
        super_map= {1: "object"}
        for cid in used:
            if cid == 0:
                continue
            coco["categories"].append({
                "id": int(cid), "name": name_map.get(cid, f"class_{cid}"), "supercategory": super_map.get(cid, "")
            })
        os.makedirs(os.path.join(inst_out, split), exist_ok=True)
        with open(os.path.join(inst_out, split, "annotations.json"), "w") as f:
            json.dump(coco, f)
        print(f"[OK] Instances -> {inst_out}/{split}  images={len(coco['images'])} anns={len(coco['annotations'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem_root", default=SEM_ROOT_DEFAULT)
    ap.add_argument("--out_root", default=INST_OUT_DEFAULT)
    ap.add_argument("--min_instance", type=int, default=MIN_INSTANCE_DEFAULT)
    ap.add_argument("--remap_255_to", type=int, default=REMAP_255_TO_DEFAULT)
    args = ap.parse_args()
    main(args.sem_root, args.out_root, args.min_instance, args.remap_255_to)
