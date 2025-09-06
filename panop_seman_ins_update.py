#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Panoptic Builder (COCO Panoptic from semantic + instances)
=========================================================
- Reads COCO Instances JSONs (from instances step).
- Uses semantic masks to fill remaining pixels as STUFF (keeps VOID=0 as void).
- Writes id2rgb panoptic PNGs + panoptic JSON per split & modality.

Conventions
-----------
VOID (unlabeled) == 0 in the panoptic PNG (encoded as RGB(0,0,0)) and is NOT listed
in segments_info (COCO panoptic spec). Background stays void unless you explicitly
treat it as a stuff class via --stuff_ids.

Input
-----
SEM_ROOT/labels/<split>/<sub>/<mod>/*.png      (class-id masks; single-channel 8/16-bit; 0=void)
INST_ROOT/<split>/annotations.json             (COCO instances file)
INST_ROOT/<split>/<mod>/*.jpg                  (images copied by instances step)

Output
------
PANO_OUT/<split>/<mod>/images/*.jpg            (copied for convenience)
PANO_OUT/<split>/<mod>/panoptic_masks/*.png    (id2rgb panoptic PNG)
PANO_OUT/<split>/panoptic_<mod>.json           (COCO panoptic JSON)

Detectron2 registration tip
---------------------------
image_root    = PANO_OUT/<split>/<mod>/images
panoptic_root = PANO_OUT/<split>/<mod>/panoptic_masks
In JSON: images[i].file_name == image basename; annotations[i].file_name == mask basename.
"""

import os
import cv2
import json
import argparse
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from tqdm import tqdm
from PIL import Image
import shutil

try:
    from pycocotools import mask as maskUtils
except Exception:
    maskUtils = None

try:
    from panopticapi.utils import id2rgb as _id2rgb
except Exception:
    _id2rgb = None

# ===== Defaults =====
SEM_ROOT_DEFAULT   = "/home/falak/UAV_dataset/semantic_composed"
INST_ROOT_DEFAULT  = "/home/falak/UAV_dataset/instances"
PANO_OUT_DEFAULT   = "/home/falak/UAV_dataset/panoptic"

THING_IDS_DEFAULT  = "1"           # comma-separated list of thing category ids
STUFF_IDS_DEFAULT  = ""            # optional comma-separated list of stuff ids (non-zero)
MIN_STUFF_AREA_DEF = 50            # remove tiny stuff fragments
SIZE_MISMATCH_DEF  = "error"       # 'error' | 'resize' | 'skip'
MODALITIES_DEFAULT = "visible,infrared"
COPY_IMAGES_DEF    = True

# ===== Utils =====
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def parse_int_set(s: str) -> Set[int]:
    if not s: return set()
    return {int(x.strip()) for x in s.split(",") if x.strip()}

def id2rgb(np_id: np.ndarray) -> np.ndarray:
    if _id2rgb is not None:
        return _id2rgb(np_id.astype(np.uint32))
    arr = np_id.astype(np.uint32)
    r = (arr >> 0) & 255
    g = (arr >> 8) & 255
    b = (arr >> 16) & 255
    return np.stack([r, g, b], axis=-1).astype(np.uint8)

def list_pngs(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")])

def list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    exts = {".jpg",".jpeg",".png",".bmp"}
    return sorted(
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and os.path.splitext(f)[1].lower() in exts
    )

def read_semantic_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise ValueError(f"Failed to read mask: {path}")
    if m.ndim != 2:
        raise ValueError(f"Mask must be single-channel ID map: {path}, got shape {m.shape}")
    # keep uint8 or uint16; cast others safely
    if m.dtype not in (np.uint8, np.uint16):
        m = m.astype(np.uint16 if m.max() > 255 else np.uint8)
    return m

def encode_bbox_from_mask(m: np.ndarray) -> List[float]:
    rows = np.any(m, axis=1); cols = np.any(m, axis=0)
    if not rows.any() or not cols.any():
        return [0.0, 0.0, 0.0, 0.0]
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]

def build_categories(inst_categories: List[Dict], thing_ids: Set[int], extra_stuff_ids: Set[int]) -> List[Dict]:
    """Merge categories from instances JSON with discovered/extra stuff ids."""
    # base from instances (preserve names if present)
    by_id = {int(c["id"]): {
        "id": int(c["id"]),
        "name": str(c.get("name", f"class_{c['id']}")),
        "isthing": int(1 if int(c["id"]) in thing_ids else 0),
        "color": list(c.get("color", [int(c["id"])%256, 0, 0])),
        "supercategory": str(c.get("supercategory", str(c.get("name",""))))
    } for c in inst_categories}

    # add any extra stuff ids (non-zero) not present
    for sid in sorted(extra_stuff_ids):
        if sid == 0:  # void is not a category
            continue
        if sid not in by_id:
            by_id[sid] = {
                "id": sid,
                "name": f"class_{sid}",
                "isthing": 0,
                "color": [ (sid*37) % 256, (sid*67) % 256, (sid*97) % 256 ],
                "supercategory": f"class_{sid}"
            }
        else:
            by_id[sid]["isthing"] = 0  # ensure marked as stuff

    # finalize ordered list
    return [by_id[k] for k in sorted(by_id.keys())]

# ===== Core panoptic building =====
def build_panoptic_for_image(
    W: int, H: int,
    anns: List[dict],
    stuff_mask: np.ndarray,
    thing_ids: Set[int],
    stuff_ids: Set[int],
    min_stuff_area: int
) -> Tuple[np.ndarray, List[dict]]:
    """Return (segmap uint32 with per-image unique ids >0; segments_info list)."""
    if maskUtils is None:
        raise SystemExit("pycocotools is required to decode segmentations")

    segmap = np.zeros((H, W), dtype=np.uint32)  # 0 stays VOID
    segments_info: List[dict] = []
    next_sid = 1

    # 1) THINGS: sort by area desc, paint onto empty pixels
    things = []
    for a in anns:
        cid = int(a["category_id"])
        if cid not in thing_ids:
            continue
        rle = a["segmentation"]
        m = maskUtils.decode(rle).astype(bool)
        area = int(m.sum())
        if area <= 0:
            continue
        bbox = encode_bbox_from_mask(m)
        things.append((area, cid, bbox, m))

    things.sort(key=lambda x: x[0], reverse=True)
    for area, cid, bbox, m in things:
        paint = m & (segmap == 0)
        p_area = int(paint.sum())
        if p_area == 0:
            continue
        segmap[paint] = np.uint32(next_sid)
        segments_info.append({
            "id": int(next_sid),
            "category_id": int(cid),
            "area": p_area,
            "bbox": bbox,
            "iscrowd": 0
        })
        next_sid += 1

    # 2) STUFF: restrict to stuff_ids if provided; else use all non-zero sem classes not in thing_ids
    remaining = (segmap == 0)
    sm = stuff_mask.copy()
    sm[~remaining] = 0

    unique_ids = np.unique(sm)
    unique_ids = unique_ids[(unique_ids != 0)]  # drop void
    if stuff_ids:
        unique_ids = np.array([u for u in unique_ids if int(u) in stuff_ids], dtype=unique_ids.dtype)
    else:
        unique_ids = np.array([u for u in unique_ids if int(u) not in thing_ids], dtype=unique_ids.dtype)

    for cid in unique_ids.tolist():
        binm = (sm == cid).astype(np.uint8)
        # connected components (8-connectivity)
        num, labels = cv2.connectedComponents(binm, connectivity=8)
        # labels: 0 is background, 1..num-1 are components
        for lab_id in range(1, num):
            comp = (labels == lab_id)
            p_area = int(comp.sum())
            if p_area < min_stuff_area:
                continue
            bbox = encode_bbox_from_mask(comp)
            segmap[comp] = np.uint32(next_sid)
            segments_info.append({
                "id": int(next_sid),
                "category_id": int(cid),
                "area": p_area,
                "bbox": bbox,
                "iscrowd": 0
            })
            next_sid += 1

    return segmap, segments_info

# ===== Main =====
def main(
    sem_root: str,
    inst_root: str,
    out_root: str,
    thing_ids_s: str,
    stuff_ids_s: str,
    min_stuff_area: int,
    size_mismatch_policy: str,
    modalities_s: str,
    copy_images: bool
):
    ensure_dir(out_root)

    splits = [s for s in ["train","val","test"] if os.path.isdir(os.path.join(inst_root, s))]
    if not splits:
        raise SystemExit("[ERROR] No splits found under INST_ROOT/{train,val,test}")

    thing_ids = parse_int_set(thing_ids_s)
    stuff_ids = parse_int_set(stuff_ids_s)  # optional explicit stuff set
    modalities = [m.strip() for m in modalities_s.split(",") if m.strip()]

    for split in splits:
        inst_json = os.path.join(inst_root, split, "annotations.json")
        if not os.path.isfile(inst_json):
            print("[WARN] missing instances JSON:", inst_json)
            continue

        with open(inst_json, "r", encoding="utf-8") as f:
            inst_data = json.load(f)

        # Build index of annotations by image id
        anns_by_img: Dict[int, List[dict]] = {}
        for a in inst_data.get("annotations", []):
            anns_by_img.setdefault(int(a["image_id"]), []).append(a)

        # Start categories from instances JSON (if present)
        inst_categories = inst_data.get("categories", [])
        # We'll add missing stuff classes later, after scanning sem masks below.

        for mod in modalities:
            images = [im for im in inst_data.get("images", []) if im.get("file_name","").startswith(f"{mod}/")]
            out_img_dir = os.path.join(out_root, split, mod, "images")
            out_pan_dir = os.path.join(out_root, split, mod, "panoptic_masks")
            ensure_dir(out_img_dir); ensure_dir(out_pan_dir)

            # Build semantic mask index (basename -> path)
            sem_split_dir = os.path.join(sem_root, "labels", split)
            mask_index: Dict[str, str] = {}
            discovered_sem_ids: Set[int] = set()
            if os.path.isdir(sem_split_dir):
                for sub in sorted(os.listdir(sem_split_dir)):
                    sub_dir = os.path.join(sem_split_dir, sub, mod)
                    if not os.path.isdir(sub_dir):
                        continue
                    for fname in list_pngs(sub_dir):
                        base = os.path.splitext(fname)[0]
                        path = os.path.join(sub_dir, fname)
                        # keys to resolve different naming:
                        keys = {base, f"{sub}_{base}"}
                        if "combined_" in base:
                            keys.add(base[base.find("combined_"):])
                        for k in keys:
                            mask_index.setdefault(k, path)
                        # collect semantic ids present (for category union)
                        try:
                            m = read_semantic_mask(path)
                            vals = np.unique(m)
                            # drop void(0)
                            discovered_sem_ids.update(int(v) for v in vals if int(v) != 0)
                        except Exception:
                            pass

            # Prepare panoptic JSON scaffold
            pano = {
                "info": {"description": f"UAV Panoptic - {split} - {mod}", "version": "1.0", "year": 2025},
                "licenses": [],
                "categories": [],     # filled after
                "images": [],
                "annotations": []
            }

            # Build final categories = instances cats ∪ discovered (stuff) ids
            extra_stuff = set(discovered_sem_ids)
            # if user supplied explicit stuff set, prefer that (ensures disjoint from things)
            if stuff_ids:
                extra_stuff = set(stuff_ids)
            pano["categories"] = build_categories(inst_categories, thing_ids, extra_stuff)

            # Fast map from id -> isthing (for sanity)
            isthing_map = {int(c["id"]): int(c.get("isthing", 0)) for c in pano["categories"]}

            # Per-image processing
            im_count = 0
            seg_count = 0
            mismatches = 0
            missing_sem = 0

            for imrec in tqdm(images, desc=f"[panoptic] {split}:{mod}"):
                im_id = int(imrec["id"])
                W = int(imrec["width"]); H = int(imrec["height"])
                img_name = os.path.basename(imrec["file_name"])
                src_img = os.path.join(inst_root, split, imrec["file_name"])
                dst_img = os.path.join(out_img_dir, img_name)
                if not os.path.isfile(src_img):
                    print("[WARN] missing image:", src_img); continue

                # copy image (preserve quality & EXIF if any)
                if copy_images:
                    try:
                        shutil.copy2(src_img, dst_img)
                    except Exception:
                        img = cv2.imread(src_img)
                        cv2.imwrite(dst_img, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                else:
                    img = cv2.imread(src_img)
                    cv2.imwrite(dst_img, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # locate semantic mask
                base = os.path.splitext(img_name)[0]
                sem_mask_path = mask_index.get(base, "")
                if not sem_mask_path and "combined_" in base:
                    key = base[base.find("combined_"):]
                    sem_mask_path = mask_index.get(key, "")
                if not sem_mask_path and "_" in base:
                    sem_mask_path = mask_index.get(base.split("_", 1)[1], "")
                if not sem_mask_path:
                    missing_sem += 1
                    continue

                try:
                    sm = read_semantic_mask(sem_mask_path)
                except Exception as e:
                    print(f"[WARN] bad semantic mask ({e}): {sem_mask_path}")
                    continue

                if sm.shape[:2] != (H, W):
                    if SIZE_MISMATCH_DEF == "error":
                        raise RuntimeError(f"Size mismatch {img_name}: image {(H,W)} vs mask {sm.shape[:2]}")
                    elif SIZE_MISMATCH_DEF == "resize":
                        sm = cv2.resize(sm, (W, H), interpolation=cv2.INTER_NEAREST)
                        mismatches += 1
                    elif SIZE_MISMATCH_DEF == "skip":
                        mismatches += 1
                        continue

                anns = anns_by_img.get(im_id, [])
                segmap, segments_info = build_panoptic_for_image(
                    W, H, anns, sm, thing_ids=thing_ids, stuff_ids=stuff_ids, min_stuff_area=min_stuff_area
                )

                # Save id2rgb PNG
                mask_name = base + ".png"
                rgb = id2rgb(segmap)
                Image.fromarray(rgb).save(os.path.join(out_pan_dir, mask_name))

                pano["images"].append({"id": im_id, "width": W, "height": H, "file_name": img_name})
                # (Optional) sanity: ensure category ids referenced exist
                for si in segments_info:
                    cid = int(si["category_id"])
                    if cid not in isthing_map:
                        # silently allow; training code may filter unknown categories
                        pass

                pano["annotations"].append({
                    "image_id": im_id,
                    "file_name": mask_name,
                    "segments_info": segments_info
                })

                im_count += 1
                seg_count += len(segments_info)

            # Write JSON for this modality
            out_json = os.path.join(out_root, split, f"panoptic_{mod}.json")
            ensure_dir(os.path.dirname(out_json))
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(pano, f, ensure_ascii=False)
            print(f"[OK] {split}:{mod}  images={im_count}  segments={seg_count}  "
                  f"missing_sem={missing_sem}  mismatches={mismatches}  -> {out_json}")

# ===== CLI =====
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem_root", default=SEM_ROOT_DEFAULT)
    ap.add_argument("--inst_root", default=INST_ROOT_DEFAULT)
    ap.add_argument("--out_root", default=PANO_OUT_DEFAULT)
    ap.add_argument("--thing_ids", default=THING_IDS_DEFAULT,
                    help="Comma-separated thing category IDs (e.g., '1,2,3').")
    ap.add_argument("--stuff_ids", default=STUFF_IDS_DEFAULT,
                    help="Optional comma-separated stuff IDs (non-zero). If empty, use semantic IDs minus thing IDs.")
    ap.add_argument("--min_stuff_area", type=int, default=MIN_STUFF_AREA_DEF,
                    help="Minimum pixel area to keep a stuff component.")
    ap.add_argument("--size_mismatch_policy", choices=["error","resize","skip"], default=SIZE_MISMATCH_DEF,
                    help="How to handle image/mask size mismatch.")
    ap.add_argument("--modalities", default=MODALITIES_DEFAULT,
                    help='Comma-separated (e.g., "visible,infrared").')
    ap.add_argument("--copy_images", action="store_true", default=COPY_IMAGES_DEF,
                    help="Copy images instead of re-encoding.")
    ap.add_argument("--no_copy_images", action="store_false", dest="copy_images")
    args = ap.parse_args()

    main(
        sem_root=args.sem_root,
        inst_root=args.inst_root,
        out_root=args.out_root,
        thing_ids_s=args.thing_ids,
        stuff_ids_s=args.stuff_ids,
        min_stuff_area=args.min_stuff_area,
        size_mismatch_policy=args.size_mismatch_policy,
        modalities_s=args.modalities,
        copy_images=args.copy_images,
    )
