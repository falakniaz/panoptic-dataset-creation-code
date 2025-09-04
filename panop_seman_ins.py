#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Panoptic Builder (COCO Panoptic from semantic + instances)
=========================================================
- Reads **COCO Instances** JSONs (from instance_from_semantic.py)
- Uses **semantic masks** to fill remaining pixels as **stuff** (incl. background)
- Writes **id2rgb** panoptic PNGs + panoptic JSON with required keys

Input:
  SEM_ROOT/labels/<split>/<sub>/<mod>/*.png      (class-id masks; 0 background)
  INST_ROOT/<split>/annotations.json             (COCO instances file)
  INST_ROOT/<split>/<mod>/*.jpg                  (images copied by instances step)

Output:
  PANO_OUT/<split>/<mod>/images/*.jpg
  PANO_OUT/<split>/<mod>/panoptic_masks/*.png    (COCO id2rgb)
  PANO_OUT/<split>/panoptic_<mod>.json

Registration tip (Detectron2):
  image_root   = PANO_OUT/<split>/<mod>/images
  panoptic_root= PANO_OUT/<split>/<mod>/panoptic_masks
  In JSON, images[i].file_name == just the image filename; annotations[i].file_name == just the mask filename.
"""

import os
import cv2
import sys
import json
import argparse
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
from skimage import measure
from PIL import Image

try:
    from pycocotools import mask as maskUtils
except Exception:
    maskUtils = None

try:
    from panopticapi.utils import id2rgb as _id2rgb
except Exception:
    _id2rgb = None

# === PATHS (EDIT THESE) ===
SEM_ROOT_DEFAULT  = "/home/falak/UAV_dataset/semantic_composed"
INST_ROOT_DEFAULT = "/home/falak/UAV_dataset/instances"
PANO_OUT_DEFAULT  = "/home/falak/UAV_dataset/panoptic"
THING_IDS_DEFAULT = [1]   # extend if you add more thing classes


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def id2rgb(np_id: np.ndarray) -> np.ndarray:
    if _id2rgb is not None:
        return _id2rgb(np_id.astype(np.uint32))
    arr = np_id.astype(np.uint32)
    r = (arr >> 0) & 255
    g = (arr >> 8) & 255
    b = (arr >> 16) & 255
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def build_panoptic_for_image(W: int, H: int, anns: List[dict], stuff_mask: np.ndarray, thing_ids: List[int]) -> Tuple[np.ndarray, List[dict]]:
    if maskUtils is None:
        raise SystemExit("pycocotools is required to decode segmentations")
    segmap = np.zeros((H, W), dtype=np.uint32)
    segments_info = []
    next_sid = 1

    # Things: decode and paint by area (desc), only onto empty pixels
    things = []
    for a in anns:
        if int(a["category_id"]) not in thing_ids:
            continue
        rle = a["segmentation"]
        m = maskUtils.decode(rle).astype(bool)
        area = int(m.sum())
        if area <= 0:
            continue
        rows = np.any(m, axis=1); cols = np.any(m, axis=0)
        if rows.any() and cols.any():
            y0, y1 = np.where(rows)[0][[0, -1]]
            x0, x1 = np.where(cols)[0][[0, -1]]
            bbox = [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]
        else:
            bbox = [0.0, 0.0, 0.0, 0.0]
        things.append((area, int(a["category_id"]), bbox, m))
    things.sort(key=lambda x: x[0], reverse=True)
    for area, cid, bbox, m in things:
        paint = m & (segmap == 0)
        p_area = int(paint.sum())
        if p_area == 0:
            continue
        segmap[paint] = np.uint32(next_sid)
        segments_info.append({"id": int(next_sid), "category_id": int(cid), "area": p_area, "bbox": bbox, "iscrowd": 0})
        next_sid += 1

    # Stuff: fill remaining per class via connected components
    remaining = (segmap == 0)
    sm = stuff_mask.copy()
    sm[~remaining] = 0
    classes = [int(v) for v in np.unique(sm) if v != 0]
    for cid in classes:
        binm = (sm == cid).astype(np.uint8)
        lab, num = measure.label(binm, connectivity=2, return_num=True)
        for cc in range(1, num+1):
            m = (lab == cc)
            p_area = int(m.sum())
            if p_area <= 0:
                continue
            rows = np.any(m, axis=1); cols = np.any(m, axis=0)
            if rows.any() and cols.any():
                y0, y1 = np.where(rows)[0][[0, -1]]
                x0, x1 = np.where(cols)[0][[0, -1]]
                bbox = [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]
            else:
                bbox = [0.0, 0.0, 0.0, 0.0]
            segmap[m] = np.uint32(next_sid)
            segments_info.append({"id": int(next_sid), "category_id": int(cid), "area": p_area, "bbox": bbox, "iscrowd": 0})
            next_sid += 1

    return segmap, segments_info


def category_entry(cid: int, isthing: int) -> dict:
    palette = [
        (220,20,60), (0,0,142), (119,11,32), (0,60,100), (0,80,100),
        (0,0,230), (106,0,228), (70,130,180), (244,35,232), (250,170,30)
    ]
    color = palette[cid % len(palette)] if cid < 1000 else (153,153,153)
    name = "object" if cid == 1 else ("background" if cid == 2 else f"class_{cid}")
    return {"id": int(cid), "name": name, "supercategory": name, "isthing": int(isthing), "color": list(color)}


def main(sem_root: str, inst_root: str, out_root: str, thing_ids: List[int]):
    ensure_dir(out_root)
    splits = [s for s in ["train","val","test"] if os.path.isdir(os.path.join(inst_root, s))]

    for split in splits:
        inst_json = os.path.join(inst_root, split, "annotations.json")
        if not os.path.isfile(inst_json):
            print("[WARN] missing instances JSON:", inst_json)
            continue
        with open(inst_json, "r") as f:
            inst = json.load(f)
        by_img = {}
        for a in inst.get("annotations", []):
            by_img.setdefault(int(a["image_id"]), []).append(a)

        # categories (include background id=2 as stuff)
        seen_cids = {int(a["category_id"]) for a in inst.get("annotations", [])}
        cats = [category_entry(2, 0)]
        for cid in sorted(seen_cids):
            cats.append(category_entry(int(cid), 1 if cid in thing_ids else 0))

        for mod in ["visible","infrared"]:
            images = [im for im in inst.get("images", []) if im.get("file_name", "").startswith(f"{mod}/")]
            out_img_dir = os.path.join(out_root, split, mod, "images")
            out_pan_dir = os.path.join(out_root, split, mod, "panoptic_masks")
            ensure_dir(out_img_dir); ensure_dir(out_pan_dir)

            # Build fast index of semantic masks for this split & modality
            sem_split = os.path.join(sem_root, "labels", split)
            mask_index = {}
            if os.path.isdir(sem_split):
                for sub in os.listdir(sem_split):
                    sub_dir = os.path.join(sem_split, sub, mod)
                    if not os.path.isdir(sub_dir):
                        continue
                    for fname in os.listdir(sub_dir):
                        if not fname.lower().endswith(".png"):
                            continue
                        base_nm = os.path.splitext(fname)[0]
                        path = os.path.join(sub_dir, fname)
                        # keys that might be used by instances step
                        keys = {base_nm, f"{sub}_{base_nm}"}
                        if "combined_" in base_nm:
                            keys.add(base_nm[base_nm.find("combined_"):])
                        for k in keys:
                            mask_index.setdefault(k, path)

            pano = {"info": {"description": f"UAV Panoptic - {split} - {mod}", "version": "1.0", "year": 2025},
                    "licenses": [], "categories": cats, "images": [], "annotations": []}

            for imrec in tqdm(images, desc=f"[panoptic] {split}:{mod}"):
                imid = int(imrec["id"]) ; W = int(imrec["width"]) ; H = int(imrec["height"])
                img_name = os.path.basename(imrec["file_name"])  # stored in INST_ROOT/split/mod/img
                src = os.path.join(inst_root, split, imrec["file_name"])  # e.g., inst_root/train/visible/foo.jpg
                dst = os.path.join(out_img_dir, img_name)
                img = cv2.imread(src)
                if img is None:
                    print("[WARN] missing image:", src)
                    continue
                cv2.imwrite(dst, img)

                # find matching semantic mask (robust to prefixes and "combined_*" names)
                base = os.path.splitext(img_name)[0]
                sem_mask = mask_index.get(base, "")
                if not sem_mask and "combined_" in base:
                    key = base[base.find("combined_"):]
                    sem_mask = mask_index.get(key, "")
                if not sem_mask and "_" in base:
                    sem_mask = mask_index.get(base.split("_", 1)[1], "")
                if not sem_mask:
                    print("[WARN] semantic mask not found for", img_name)
                    continue
                m_sem = cv2.imread(sem_mask, cv2.IMREAD_GRAYSCALE)
                if m_sem is None:
                    print("[WARN] bad semantic mask:", sem_mask)
                    continue
                if m_sem.shape[:2] != (H, W):
                    m_sem = cv2.resize(m_sem, (W, H), interpolation=cv2.INTER_NEAREST)

                anns = by_img.get(imid, [])
                segmap, seginfo = build_panoptic_for_image(W, H, anns, m_sem, thing_ids)

                # save id2rgb png; file names are just basenames for D2
                rgb = id2rgb(segmap)
                mask_name = base + ".png"
                Image.fromarray(rgb).save(os.path.join(out_pan_dir, mask_name))

                pano["images"].append({"id": imid, "width": W, "height": H, "file_name": img_name})
                pano["annotations"].append({"image_id": imid, "file_name": mask_name, "segments_info": seginfo})

            with open(os.path.join(out_root, split, f"panoptic_{mod}.json"), "w") as f:
                json.dump(pano, f)
            print(f"[OK] Panoptic -> {out_root}/{split}/panoptic_{mod}.json  images={len(pano['images'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem_root", default=SEM_ROOT_DEFAULT)
    ap.add_argument("--inst_root", default=INST_ROOT_DEFAULT)
    ap.add_argument("--out_root", default=PANO_OUT_DEFAULT)
    ap.add_argument("--thing_ids", nargs="*", type=int, default=THING_IDS_DEFAULT)
    args = ap.parse_args()
    main(args.sem_root, args.inst_root, args.out_root, args.thing_ids)
