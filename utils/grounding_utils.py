# utils/grounding_utils.py
from __future__ import annotations

import os
import json
import math
import shutil
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from PIL import Image

import torch
from torchvision.ops import box_convert

# ---- GroundingDINO imports ----
import GroundingDINO.groundingdino.datasets.transforms as T  # noqa: F401 (kept in case you extend)
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import load_image, predict

# ---- SAM imports ----
from segment_anything.segment_anything import SamPredictor
from segment_anything.segment_anything.build_sam import build_sam_vit_b

# ---- HF hub for DINO checkpoints ----
from huggingface_hub import hf_hub_download


# ----------------------------- small utils ----------------------------- #

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _list_images(image_dir: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = sorted(
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in exts and f.startswith("image_")
    )
    return [os.path.join(image_dir, f) for f in files]


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")


def _torch_device_str(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _maybe_half(model: torch.nn.Module, fp16: bool) -> torch.nn.Module:
    if fp16:
        return model.half()
    return model


def _dtype_from_fp16(fp16: bool):
    return torch.float16 if fp16 else torch.float32


# ----------------------- payload → entity prompts ----------------------- #
def _entities_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert your text HPG payload to a list of entity prompts.
    Heuristic plural detection (coarse): 'and' in text OR endswith('s') for common nouns.
    You can replace/improve this if you already tag plurality in payload.
    """
    entities = []
    for node in payload.get("nodes", []):
        text = (node.get("text") or "").strip()
        if not text:
            continue
        low = text.lower()
        is_plural = (" and " in f" {low} ") or (low.endswith("s") and not low.endswith("ss"))
        entities.append({
            "id": node.get("id"),
            "text": text,
            "is_plural": bool(is_plural),
        })
    return entities


# ----------------------------- DINO loading ----------------------------- #
def _load_groundingdino_from_hf(
    repo_id: str,
    ckpt_file: str,
    cfg_file: str,
    device: str,
    fp16: bool,
) -> torch.nn.Module:
    cfg_path = hf_hub_download(repo_id=repo_id, filename=cfg_file)
    args = SLConfig.fromfile(cfg_path)
    args.device = _torch_device_str(device)

    model = build_model(args)
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_file)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    _ = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    model = _maybe_half(model, fp16)
    model.to(args.device)
    return model


# ------------------------------ STAGE A -------------------------------- #
@torch.inference_mode()
def run_grounding_pass(
    image_dir: str,
    payload: Dict[str, Any],
    grounding_ckpt_repo: str,
    grounding_ckpt_file: str,
    grounding_cfg_file: str,
    save_dir: str,
    device: str = "cuda",
    fp16: bool = True,
    box_thresh: float = 0.30,
    text_thresh: float = 0.25,
    plural_topk: int = 3,
) -> None:
    """
    Stage A:
      - Load GroundingDINO
      - For each image and each entity prompt, produce candidate boxes/logits
      - Keep top-1 for singular, top-k for plural
      - Save per-image JSON (absolute xyxy pixel boxes, scores, metadata)
      - Free model + VRAM at the end
    """
    device = _torch_device_str(device)
    dtype = _dtype_from_fp16(fp16)

    _ensure_dir(save_dir)
    images = _list_images(image_dir)
    if not images:
        print(f"[Grounding] No images found in: {image_dir}")
        return

    # Build entity prompts from your payload
    ent_prompts = _entities_from_payload(payload)
    if not ent_prompts:
        print("[Grounding] No entities found in payload; nothing to do.")
        return

    # Load DINO
    model = _load_groundingdino_from_hf(
        repo_id=grounding_ckpt_repo,
        ckpt_file=grounding_ckpt_file,
        cfg_file=grounding_cfg_file,
        device=device,
        fp16=fp16,
    )

    # Process images
    for img_path in images:
        # GroundingDINO helper returns:
        #  - image_source (HWC uint8 np array), image (normalized tensor used by DINO)
        image_source, image = load_image(img_path)  # image is CHW tensor on CPU
        H, W, _ = image_source.shape

        per_image = {
            "image_path": img_path,
            "image_size": [int(H), int(W)],
            "entities": {},   # key: entity_id or slug(entity_text)
            "box_threshold": float(box_thresh),
            "text_threshold": float(text_thresh),
            "plural_topk": int(plural_topk),
            "dtype": "fp16" if fp16 else "fp32",
        }

        # For each entity prompt, run DINO
        for ent in ent_prompts:
            ent_text = ent["text"]
            ent_id = ent.get("id") or _slug(ent_text)
            is_plural = bool(ent.get("is_plural", False))

            # Run model
            with (torch.autocast(device_type="cuda", dtype=dtype) if (device == "cuda" and fp16) else torch.cuda.amp.autocast(enabled=False)):
                boxes_cxcywh, logits, phrases = predict(
                    model=model,
                    image=image,                 # DINO expects this preprocessed tensor
                    caption=ent_text,
                    box_threshold=box_thresh,
                    text_threshold=text_thresh,
                    device=device,
                )

            if boxes_cxcywh is None or len(boxes_cxcywh) == 0:
                # No detections; still record an empty entry to keep schema consistent
                per_image["entities"][ent_id] = {
                    "text": ent_text,
                    "is_plural": is_plural,
                    "boxes_xyxy": [],
                    "scores": [],
                }
                continue

            # Convert boxes to absolute xyxy pixel coordinates
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_cxcywh) * torch.tensor([W, H, W, H], device=boxes_cxcywh.device, dtype=boxes_cxcywh.dtype)
            boxes_xyxy = boxes_xyxy.detach().cpu()
            scores = logits.detach().cpu().sigmoid().view(-1)

            # Rank by confidence
            order = torch.argsort(scores, descending=True)
            boxes_xyxy = boxes_xyxy[order]
            scores = scores[order]

            # Keep top-1 for singular, top-k for plural
            k = plural_topk if is_plural else 1
            k = min(k, boxes_xyxy.shape[0])

            top_boxes = boxes_xyxy[:k].numpy().tolist()
            top_scores = scores[:k].numpy().tolist()

            per_image["entities"][ent_id] = {
                "text": ent_text,
                "is_plural": is_plural,
                "boxes_xyxy": [[float(x) for x in b] for b in top_boxes],
                "scores": [float(s) for s in top_scores],
            }

        # Save per-image grounding JSON
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_json = os.path.join(save_dir, f"{base}.grounding.json")
        with open(out_json, "w") as f:
            json.dump(per_image, f, indent=2)

    # Free VRAM
    del model
    if device == "cuda":
        torch.cuda.empty_cache()


# ------------------------------ STAGE B -------------------------------- #
@torch.inference_mode()
def run_segmentation_pass(
    image_dir: str,
    grounding_dir: str,    # per-image JSONs from Stage A
    sam_checkpoint: str,
    save_dir: str,
    device: str = "cuda",
    fp16: bool = True,
) -> None:
    """
    Stage B:
      - Load SAM once
      - For each per-image grounding JSON:
          * load image
          * set SAM image embedding
          * for each entity's kept boxes → get masks
          * save binary masks (PNG) + a manifest per image
      - Free per-image embedding eagerly
    """
    device = _torch_device_str(device)
    dtype = _dtype_from_fp16(fp16)

    # Prepare output
    masks_root = save_dir
    _ensure_dir(masks_root)

    # Collect grounding JSONs
    if not os.path.isdir(grounding_dir):
        print(f"[Segmentation] Grounding dir not found: {grounding_dir}")
        return
    grounding_files = sorted(
        f for f in os.listdir(grounding_dir)
        if f.endswith(".grounding.json")
    )
    if not grounding_files:
        print(f"[Segmentation] No grounding JSONs in: {grounding_dir}")
        return

    # Load SAM
    sam = build_sam_vit_b(checkpoint=sam_checkpoint)
    sam = _maybe_half(sam, fp16)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    for jf in grounding_files:
        jpath = os.path.join(grounding_dir, jf)
        with open(jpath, "r") as f:
            grounding = json.load(f)

        img_path = grounding["image_path"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(image_dir, os.path.basename(img_path))  # fallback if relative

        if not os.path.exists(img_path):
            print(f"[Segmentation] Missing image referenced by JSON: {img_path}")
            continue

        # Load image (uint8 HWC)
        image_source = np.array(Image.open(img_path).convert("RGB"))
        H, W, _ = image_source.shape

        # Set image embedding on GPU (VRAM heavy)
        predictor.set_image(image_source)

        # Per-image output dir
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_dir = os.path.join(masks_root, base)
        _ensure_dir(out_dir)

        # Manifest to record masks written
        manifest = {
            "image_path": img_path,
            "image_size": [int(H), int(W)],
            "entities": {}
        }

        # Iterate entities/boxes
        for ent_id, ent_info in grounding.get("entities", {}).items():
            ent_text = ent_info.get("text", ent_id)
            boxes = ent_info.get("boxes_xyxy", [])
            scores = ent_info.get("scores", [])
            if not boxes:
                manifest["entities"][ent_id] = {
                    "text": ent_text,
                    "masks": []
                }
                continue

            # Prepare boxes tensor (SAM expects tensors on same device as model after transform)
            boxes_xyxy = torch.tensor(boxes, dtype=torch.float32, device="cpu")
            # Transform to SAM coordinate frame
            boxes_trans = predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)

            # Predict masks in a single batch call
            with (torch.autocast(device_type="cuda", dtype=dtype) if (device == "cuda" and fp16) else torch.cuda.amp.autocast(enabled=False)):
                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=boxes_trans,
                    multimask_output=False,
                )  # [N, 1, H, W] bool/bfloat16-ish

            masks = masks.squeeze(1)  # [N, H, W]
            ent_records = []
            for i in range(masks.shape[0]):
                m = masks[i].detach().to("cpu").numpy().astype(np.uint8) * 255  # binary PNG (0/255)

                mask_name = f"entity={_slug(ent_text)}__idx={i}.png"
                mask_path = os.path.join(out_dir, mask_name)
                Image.fromarray(m, mode="L").save(mask_path)

                ent_records.append({
                    "mask_path": mask_path,
                    "box_xyxy": [float(x) for x in boxes[i]],
                    "score": float(scores[i]) if i < len(scores) else float("nan"),
                })

            manifest["entities"][ent_id] = {
                "text": ent_text,
                "masks": ent_records
            }

        # Save per-image manifest JSON (next to masks)
        with open(os.path.join(out_dir, f"{base}.masks.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        # Try to aggressively free the per-image embedding
        # (predictor doesn't have a public reset in all builds; re-setting on the next image
        # overwrites its state. We still prompt CUDA to free cache.)
        if device == "cuda":
            torch.cuda.empty_cache()

    # Free SAM
    del predictor
    del sam
    if device == "cuda":
        torch.cuda.empty_cache()
