import os
import glob
import json
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import entropy


# -------------------------------------------------------------
# Load Embeddings
# -------------------------------------------------------------
def load_embeddings(embedding_dir: str) -> Dict[str, np.ndarray]:
    """
    Load embeddings from .npz files and group them by entity.
    Returns: { entity: [embeddings ...] }
    """
    entity_dict: Dict[str, List[np.ndarray]] = {}
    for fp in sorted(glob.glob(os.path.join(embedding_dir, "*.npz"))):
        data = np.load(fp, allow_pickle=True)
        keys = data["keys"]
        embs = data["embeddings"]
        for k, v in zip(keys, embs):
            entity = k.item() if isinstance(k, np.ndarray) else str(k)
            entity_dict.setdefault(entity, []).append(v)
    # Stack each entity’s embeddings into arrays
    return {e: np.stack(v, axis=0) for e, v in entity_dict.items()}


# -------------------------------------------------------------
# KL Divergence Computation
# -------------------------------------------------------------
def compute_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute KL divergence between two sets of embeddings using Gaussian KDE histograms.
    """
    # Project embeddings to 1D using PCA (for visualization-friendly KL)
    pca = PCA(n_components=1)
    combined = np.concatenate([p, q], axis=0)
    proj = pca.fit_transform(combined)
    p_proj = proj[: len(p), 0]
    q_proj = proj[len(p):, 0]

    # Build histograms
    bins = 30
    p_hist, bin_edges = np.histogram(p_proj, bins=bins, density=True)
    q_hist, _ = np.histogram(q_proj, bins=bin_edges, density=True)

    # Smooth + normalize
    p_hist = np.clip(p_hist, eps, None)
    q_hist = np.clip(q_hist, eps, None)

    p_hist /= np.sum(p_hist)
    q_hist /= np.sum(q_hist)

    return float(entropy(p_hist, q_hist))  # KL(P || Q)


def compute_entitywise_kl(
    caption1_embs: Dict[str, np.ndarray],
    caption2_embs: Dict[str, np.ndarray],
    skip_entities: List[str] = [],
) -> Dict[str, float]:
    """
    Compute KL divergence per entity between two captions.
    """
    results = {}
    for entity in caption1_embs:
        if entity not in caption2_embs:
            continue
        if entity in skip_entities:
            continue
        kl = compute_kl(caption1_embs[entity], caption2_embs[entity])
        results[entity] = kl
    return results


# -------------------------------------------------------------
# Visualization: KDE Plots
# -------------------------------------------------------------
def plot_entity_kde(entity: str, embs1: np.ndarray, embs2: np.ndarray, label1: str, label2: str, save_dir=None):
    """
    Plot KDE histograms of projected embeddings for one entity.
    """
    pca = PCA(n_components=1)
    combined = np.concatenate([embs1, embs2], axis=0)
    proj = pca.fit_transform(combined)
    e1_proj = proj[: len(embs1), 0]
    e2_proj = proj[len(embs1):, 0]

    plt.figure(figsize=(7, 5))
    sns.kdeplot(e1_proj, fill=True, label=label1, color="orange")
    sns.kdeplot(e2_proj, fill=True, label=label2, color="gray")
    plt.title(f"KDE of Embeddings — {entity}")
    plt.xlabel("PCA 1D Projection")
    plt.ylabel("Density")
    plt.legend()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{entity}_kde.png"), dpi=200)
        plt.close()
    else:
        plt.show()


# -------------------------------------------------------------
# Visualization: t-SNE / PCA Plots
# -------------------------------------------------------------
def plot_tsne_pca(
    entity: str,
    embs1: np.ndarray,
    embs2: np.ndarray,
    label1: str,
    label2: str,
    method: str = "tsne",
    save_dir=None,
):
    """
    Visualize entity embeddings using t-SNE or PCA (2D).
    """
    combined = np.concatenate([embs1, embs2], axis=0)
    labels = np.array([label1] * len(embs1) + [label2] * len(embs2))

    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=10, learning_rate=100, random_state=42)
    else:
        raise ValueError("method must be 'tsne' or 'pca'")

    reduced = reducer.fit_transform(combined)

    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette=["orange", "gray"])
    plt.title(f"{method.upper()} Visualization — {entity}")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{entity}_{method}.png"), dpi=200)
        plt.close()
    else:
        plt.show()
