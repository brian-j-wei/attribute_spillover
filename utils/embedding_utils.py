import os, json, math, glob
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import CLIPVisionModel, AutoImageProcessor

# -------------------------- Loading / Preprocess -------------------------- #

def load_clip_vision(
    model_id: str = "openai/clip-vit-base-patch16",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """
    Returns (processor, vision) where `vision` outputs patch tokens from the ViT.
    """
    processor = AutoImageProcessor.from_pretrained(model_id)
    vision = CLIPVisionModel.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=True).eval().to(device)
    return processor, vision


def _to_device(x: torch.Tensor, device: str, dtype: torch.dtype) -> torch.Tensor:
    return x.to(device=device, dtype=dtype, non_blocking=True)


def _load_image_tensor(image_path: str, processor, device: str, dtype: torch.dtype) -> Tuple[torch.Tensor, Tuple[int,int]]:
    """
    Load image, preprocess to CLIP input. Returns (pixel_values [1,C,H,W], (H_orig, W_orig))
    """
    img = Image.open(image_path).convert("RGB")
    H0, W0 = img.size[1], img.size[0]
    enc = processor(images=img, return_tensors="pt")
    pixel_values = _to_device(enc["pixel_values"], device, dtype)
    return pixel_values, (H0, W0)


# ----------------------------- Vision Tokens ------------------------------ #

@torch.inference_mode()
def _get_patch_tokens(pixel_values: torch.Tensor, vision: CLIPVisionModel) -> Tuple[torch.Tensor, Tuple[int,int]]:
    """
    Forward CLIP vision; return patch tokens [B, Gh*Gw, D] and (Gh, Gw).
    Sequence = [CLS] + patches; we drop CLS.
    """
    out = vision(pixel_values=pixel_values, output_hidden_states=False)
    # last_hidden_state: [B, 1 + Gh*Gw, D]
    seq = out.last_hidden_state  # [B, L, D]
    cls, patches = seq[:, :1, :], seq[:, 1:, :]
    B, N, D = patches.shape
    G = int(math.sqrt(N))
    assert G * G == N, f"Non-square patch grid? N={N}"
    return patches, (G, G)  # [B, G*G, D], (G,G)


# ------------------------------- Masks I/O -------------------------------- #

def _load_mask_file(fp: str) -> torch.Tensor:
    """
    Load a mask from .pt / .npy / .png. Returns bool tensor [H, W] on CPU.
    """
    ext = os.path.splitext(fp)[1].lower()
    if ext == ".pt":
        m = torch.load(fp, map_location="cpu")
        if m.ndim == 3:
            m = m.squeeze(0)
        # allow float/bool
        if m.dtype != torch.bool:
            m = m > 0.5
        return m.bool()
    elif ext == ".npy":
        arr = np.load(fp)
        if arr.ndim == 3:
            arr = arr.squeeze(0)
        return torch.from_numpy(arr > 0.5) if arr.dtype != np.bool_ else torch.from_numpy(arr).bool()
    elif ext in {".png", ".jpg", ".jpeg"}:
        im = Image.open(fp).convert("L")
        arr = np.array(im) > 127
        return torch.from_numpy(arr).bool()
    else:
        raise ValueError(f"Unsupported mask file: {fp}")


def _load_entity_masks_for_image(masks_root: str, image_stem: str) -> Dict[str, List[torch.Tensor]]:
    """
    Loads all masks for a single image into a dict: {entity_text: [mask_bool_HxW, ...]}.

    Handles SAM-style JSON where we directly read `mask_path` for each entity.

    Directory structure:
        {masks_root}/{image_stem}/image_X.masks.json
    """
    img_dir = os.path.join(masks_root, image_stem)
    if not os.path.isdir(img_dir):
        return {}

    out: Dict[str, List[torch.Tensor]] = {}

    for fp in sorted(glob.glob(os.path.join(img_dir, "*.json"))):
        with open(fp, "r") as f:
            data = json.load(f)

        entities = data.get("entities", {})
        for ent_id, ent_info in entities.items():
            ent_text = ent_info["text"].replace(" ", "_")
            masks = ent_info.get("masks", [])

            for m in masks:
                mask_path = m["mask_path"]
                if not os.path.isfile(mask_path):
                    continue

                # Load mask as binary
                im = Image.open(mask_path).convert("L")
                arr = np.array(im) > 127  # threshold grayscale mask
                mask_tensor = torch.from_numpy(arr).bool()

                out.setdefault(ent_text, []).append(mask_tensor)

    return out



# ---------------------------- Masked Pooling ------------------------------ #

def _resize_mask_to_grid(mask: torch.Tensor, grid_hw: Tuple[int,int], H_img: int, W_img: int, clip_image_hw: Tuple[int,int]) -> torch.Tensor:
    """
    Map a binary mask at original image resolution -> CLIP patch grid (Gh, Gw).
    Steps:
      - resize mask to CLIP input spatial size (processor.size), then to (Gh, Gw), nearest.
    Returns Bool [Gh, Gw]
    """
    Gh, Gw = grid_hw
    # First resize to CLIP input spatial size (from processor) — but we don't have it here.
    # Approximation: directly downsample from original to grid with bilinear then threshold.
    m = mask.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    m_grid = F.interpolate(m, size=(Gh, Gw), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
    return (m_grid > 0.5)


def _pool_tokens_by_mask(patches: torch.Tensor, mask_grid: torch.Tensor) -> torch.Tensor:
    """
    patches: [1, Gh*Gw, D]; mask_grid: [Gh, Gw] bool
    Returns [D] (L2-normalized). Falls back to global avg if mask empty.
    """
    B, N, D = patches.shape
    GhGw = mask_grid.numel()
    assert N == GhGw, "Token/MASK size mismatch."

    sel = mask_grid.flatten().nonzero(as_tuple=False).squeeze(-1)  # [K] or []
    if sel.numel() == 0:
        # fallback: global mean
        v = patches.mean(dim=1).squeeze(0)  # [D]
    else:
        v = patches[:, sel, :].mean(dim=1).squeeze(0)  # [D]
    v = F.normalize(v, dim=-1)
    return v


# --------------------------- Main Extraction API -------------------------- #

@torch.inference_mode()
def extract_embeddings_for_image(
    image_path: str,
    masks_root: str,
    processor,
    vision: CLIPVisionModel,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Dict:
    """
    Returns a dict:
      {
        "image": <filename>,
        "keys": ["entity=a_laptop", "entity=a_backpack", "background"],
        "embeddings": np.ndarray [K, D],
        "meta": {"grid": [Gh, Gw], "model": model_id}
      }
    """
    # Load image and tokens
    pixel_values, (H0, W0) = _load_image_tensor(image_path, processor, device, dtype)
    patches, (Gh, Gw) = _get_patch_tokens(pixel_values, vision)   # [1, Gh*Gw, D]

    # Load masks for this image
    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    ent2masks = _load_entity_masks_for_image(masks_root, image_stem)

    # Pool per-entity (if multiple masks per entity, average after token pooling)
    keys: List[str] = []
    vecs: List[torch.Tensor] = []

    union_mask = torch.zeros((Gh, Gw), dtype=torch.bool)
    for ent, masks in ent2masks.items():
        ent_vecs = []
        for m in masks:
            mg = _resize_mask_to_grid(m, (Gh, Gw), H0, W0, clip_image_hw=None)
            union_mask |= mg
            v = _pool_tokens_by_mask(patches, mg)
            ent_vecs.append(v)
        v_ent = F.normalize(torch.stack(ent_vecs, dim=0).mean(dim=0), dim=-1) if len(ent_vecs) > 1 else ent_vecs[0]
        keys.append(f"entity={ent}")
        vecs.append(v_ent)

    # Background = complement of union of entity masks
    bg_grid = ~union_mask if union_mask.any() else torch.ones((Gh, Gw), dtype=torch.bool)
    v_bg = _pool_tokens_by_mask(patches, bg_grid)
    keys.append("background")
    vecs.append(v_bg)

    # Stack + move to CPU
    V = torch.stack(vecs, dim=0).cpu().numpy()  # [K, D]

    return {
        "image": os.path.basename(image_path),
        "keys": keys,
        "embeddings": V,
        "meta": {"grid": [Gh, Gw]},
    }


@torch.inference_mode()
def extract_dir_embeddings(
    image_dir: str,
    masks_root: str,
    save_dir: str,
    model_id: str = "openai/clip-vit-base-patch16",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    ext: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
) -> List[str]:
    """
    Walks image_dir (expects image_{i}.png), matches masks in masks_root/<image_stem>/,
    writes one NPZ per image to save_dir, returns list of saved file paths.
    """
    os.makedirs(save_dir, exist_ok=True)
    processor, vision = load_clip_vision(model_id=model_id, device=device, dtype=dtype)

    saved = []
    for fp in sorted(glob.glob(os.path.join(image_dir, "*"))):
        if os.path.splitext(fp)[1].lower() not in ext:
            continue
        pack = extract_embeddings_for_image(fp, masks_root, processor, vision, device=device, dtype=dtype)
        out_fp = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(fp))[0]}.npz")
        np.savez_compressed(out_fp, keys=np.array(pack["keys"]), embeddings=pack["embeddings"], meta=json.dumps(pack["meta"]))
        saved.append(out_fp)

    # free VRAM
    del vision
    torch.cuda.empty_cache()
    return saved
