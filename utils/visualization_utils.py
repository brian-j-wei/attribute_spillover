# hpg_segments_viz.py
import os
import json
import glob
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import networkx as nx


# =========================== Layout / Draw Helpers ===========================

def _prune_for_tree(G: nx.MultiDiGraph, *, drop_cc_labels=("and", "or")) -> nx.DiGraph:
    """
    Collapse to a DiGraph, drop coordination edges (and/or), and break tiny cycles.
    """
    DG = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        lbl = str(data.get("label", "rel")).lower()
        if lbl in drop_cc_labels:
            continue
        if not DG.has_edge(u, v):
            DG.add_edge(u, v, label=lbl)

    # Best-effort cycle breaking
    try:
        while not nx.is_directed_acyclic_graph(DG):
            cycle = next(nx.simple_cycles(DG))
            cand = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]
            edge_to_remove = min(cand, key=lambda e: len(DG[e[0]][e[1]].get("label", "")))
            DG.remove_edge(*edge_to_remove)
    except Exception:
        pass
    return DG


def _hierarchical_positions(
    DG: nx.DiGraph,
    *,
    rankdir: str = "TB",
    layer_gap: float = 0.9,
    node_gap: float = 0.9
):
    """
    Pure-Python layered layout (used if Graphviz isn't available).
    """
    roots = [n for n in DG.nodes if DG.in_degree(n) == 0]
    if not roots:
        min_in = min((DG.in_degree(n) for n in DG.nodes), default=0)
        roots = [n for n in DG.nodes if DG.in_degree(n) == min_in]

    layers, visited, frontier = [], set(), list(roots)
    while frontier:
        layers.append(frontier)
        visited.update(frontier)
        nxt = []
        for u in frontier:
            for v in DG.successors(u):
                if v not in visited and all((p in visited) for p in DG.predecessors(v)):
                    nxt.append(v)
        # unique + stable
        frontier = list(dict.fromkeys(nxt))

    # stragglers
    remaining = [n for n in DG.nodes if n not in visited]
    if remaining:
        layers.append(remaining)

    pos = {}
    for li, layer in enumerate(layers):
        layer_sorted = sorted(layer, key=lambda n: (-DG.out_degree(n), str(n)))
        coords = []
        for j, n in enumerate(layer_sorted):
            if rankdir == "TB":
                x, y = j * node_gap, -li * layer_gap
            else:  # LR
                x, y = li * layer_gap, -j * node_gap
            pos[n] = (x, y)
            coords.append((n, x, y))

        # NEW: center this layer horizontally (TB) or vertically (LR)
        if coords:
            if rankdir == "TB":
                mean_x = sum(x for _, x, _ in coords) / len(coords)
                for n, x, y in coords:
                    pos[n] = (x - mean_x, y)
            else:
                mean_y = sum(y for _, _, y in coords) / len(coords)
                for n, x, y in coords:
                    pos[n] = (x, y - mean_y)

    return pos


def _graphviz_positions_or_none(
    G: nx.DiGraph,
    *,
    rankdir: str = "TB",
    nodesep: float = 0.2,
    ranksep: float = 0.3
):
    """
    Try Graphviz 'dot' for the cleanest hierarchy; otherwise return None.
    """
    try:
        from networkx.drawing.nx_agraph import graphviz_layout  # pygraphviz
        return graphviz_layout(G, prog="dot", args=f"-Grankdir={rankdir} -Gnodesep={nodesep} -Granksep={ranksep}")
    except Exception:
        try:
            from networkx.drawing.nx_pydot import graphviz_layout  # pydot
            return graphviz_layout(G, prog="dot", args=f"-Grankdir={rankdir} -Gnodesep={nodesep} -Granksep={ranksep}")
        except Exception:
            return None


def _wrap(text: str, width: int = 16) -> str:
    out, line = [], []
    for word in str(text).split():
        if sum(len(w) for w in line) + len(line) + len(word) > width:
            out.append(" ".join(line)); line = []
        line.append(word)
    if line:
        out.append(" ".join(line))
    return "\n".join(out)


def _normalize_with_safe_margins(pos, node_size, figsize, ax=None, pad_pts=30):
    """
    Normalize positions to [0,1] and inset by margins derived from node_size so
    nodes never render off-canvas. Returns (pos_safe, mx, my).
    """
    xs = [xy[0] for xy in pos.values()] or [0.0, 1.0]
    ys = [xy[1] for xy in pos.values()] or [0.0, 1.0]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    dx = (maxx - minx) or 1.0
    dy = (maxy - miny) or 1.0
    pos01 = {n: ((x - minx) / dx, (y - miny) / dy) for n, (x, y) in pos.items()}

    # Convert node_size (pts^2) to radius in points
    node_radius_pts = 0.5 * math.sqrt(max(node_size, 1.0))

    if ax is not None:
        dpi = ax.figure.dpi
    else:
        dpi = 100.0
    w_in, h_in = figsize
    ax_w_pts = max(w_in * dpi, 1.0)
    ax_h_pts = max(h_in * dpi, 1.0)

    mx = (node_radius_pts + pad_pts) / ax_w_pts
    my = (node_radius_pts + pad_pts) / ax_h_pts

    pos_safe = {}
    for n, (x, y) in pos01.items():
        xsafe = mx + x * (1 - 2 * mx)
        ysafe = my + y * (1 - 2 * my)
        pos_safe[n] = (min(max(xsafe, mx), 1 - mx), min(max(ysafe, my), 1 - my))
    return pos_safe, mx, my

def _center_squeeze_positions(pos01, *, mx: float, my: float, squeeze_x: float = 0.75, squeeze_y: float = 0.75):
    """
    Pull positions toward the center (0.5, 0.5) after normalization.
    squeeze_x/y in (0,1]; smaller -> stronger pull toward center.
    """
    out = {}
    for n, (x, y) in pos01.items():
        xs = 0.5 + (x - 0.5) * squeeze_x
        ys = 0.5 + (y - 0.5) * squeeze_y
        # keep inside the safe margins
        xs = max(mx, min(1 - mx, xs))
        ys = max(my, min(1 - my, ys))
        out[n] = (xs, ys)
    return out

# ======================== Robust Thumbnail Construction =======================

def _load_mask_array(mask_path: str) -> np.ndarray:
    """
    Return a boolean mask from various mask formats:
    - binary/gray PNG (0..255)
    - RGBA/LA: use alpha channel
    """
    m = Image.open(mask_path)
    if m.mode in ("LA", "RGBA"):
        mask = np.array(m.split()[-1])
    else:
        mask = np.array(m.convert("L"))
    return mask > 0


def _first_valid_thumb(masks_list, img_np, *, min_area: int = 25, thumb_px: int = 96):
    """
    Try each mask in order until we get a non-empty, non-tiny crop.
    Return a PIL.Image (thumb) or None.
    """
    for m in masks_list:
        mask_fp = m.get("mask_path")
        if not mask_fp or not os.path.exists(mask_fp):
            continue
        try:
            mask_arr = _load_mask_array(mask_fp)
        except Exception:
            continue
        if not mask_arr.any():
            continue

        ys, xs = np.where(mask_arr)
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        area = (y2 - y1 + 1) * (x2 - x1 + 1)
        if area < min_area:
            continue

        masked = img_np.copy()
        masked[~mask_arr] = 255  # white out background
        crop = masked[y1:y2+1, x1:x2+1]
        if crop.size == 0:
            continue

        return Image.fromarray(crop).resize((thumb_px, thumb_px))
    return None


# ============================ Main Visualization =============================

def visualize_hpg_with_segments(
    image_path: str,
    masks_root: str,
    G: nx.MultiDiGraph,
    payload: dict,
    figsize: tuple = (14, 10),
    node_size: int = 3200,
    font_size: int = 11,
    *,
    layout: str = "hier",            # "hier" | "spring"
    rankdir: str = "LR",             # "TB" | "LR"
    drop_coordination_edges: bool = True,
    band_by_sentence: bool = True,
    sentence_gap: float = 0.7,
    show_edge_labels: bool = False,
    # Graphviz tuning (if available)
    graphviz_nodesep: float = 0.15,
    graphviz_ranksep: float = 0.20,
    # thumbnails
    thumb_px: int = 96,
    thumb_offset: float = 0.14,      # axes fraction offset away from node
    thumb_box: float = 0.12          # axes fraction size of the thumbnail
):
    """
    Draw a hierarchical HPG with per-entity thumbnails offset from nodes.
    Single figure; node-aware margins to keep everything on-screen; robust
    mask loading; id/text-based thumbnail matching; connector lines.
    """

    # --- Load the base image (for masked thumbnails only) ---
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    # --- Load mask metadata ---
    json_files = glob.glob(os.path.join(masks_root, "*.json"))
    if len(json_files) != 1:
        raise FileNotFoundError(f"Expected exactly one .json in {masks_root}, found {len(json_files)}")
    with open(json_files[0], "r") as f:
        meta = json.load(f)

    # --- Build thumbnails keyed by both entity id and normalized text ---
    ent2thumb_by_id = {}
    ent2thumb_by_text = {}

    for ent_id, ent_info in meta.get("entities", {}).items():
        thumb = _first_valid_thumb(ent_info.get("masks", []), img_np, min_area=25, thumb_px=thumb_px)
        if thumb is None:
            continue
        ent2thumb_by_id[str(ent_id)] = thumb
        key_text = (ent_info.get("text") or "").strip().casefold()
        if key_text:
            ent2thumb_by_text[key_text] = thumb

    # --- Clean graph for tree-like layout ---
    DG = _prune_for_tree(G, drop_cc_labels=("and", "or") if drop_coordination_edges else ())

    # --- Compute positions (hierarchical preferred) ---
    if layout == "hier":
        pos = _graphviz_positions_or_none(DG, rankdir=rankdir, nodesep=graphviz_nodesep, ranksep=graphviz_ranksep)
        if pos is None:
            pos = _hierarchical_positions(DG, rankdir=rankdir, layer_gap=0.85, node_gap=0.85)
    else:
        pos = nx.spring_layout(DG, seed=42, k=0.9)

    # --- Optional: band by sentence to shorten cross-sentence edges ---
    if band_by_sentence:
        sent_ids = {n: G.nodes[n].get("sent_id", 0) for n in DG.nodes}
        remap = {sid: i for i, sid in enumerate(sorted(set(sent_ids.values())))}
        for n, (x, y) in pos.items():
            band = remap[sent_ids[n]]
            if rankdir == "TB":
                pos[n] = (x, y - band * sentence_gap)
            else:  # LR
                pos[n] = (x + band * sentence_gap, y)

    # --- One figure/axes; normalize with node-aware margins ---
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    pos_norm, margin_x, margin_y = _normalize_with_safe_margins(
        pos, node_size=node_size, figsize=figsize, ax=ax, pad_pts=15  # <- slightly larger padding to pull inward
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # --- Draw graph (edges first, then nodes, then labels) ---
    node_labels = {n: _wrap(G.nodes[n].get("text", n), width=16) for n in DG.nodes}
    edge_labels = {(u, v): DG[u][v].get("label", "rel") for u, v in DG.edges}

    nx.draw_networkx_edges(
        DG, pos_norm,
        arrowstyle="-|>", arrowsize=14, width=1.3,
        connectionstyle="arc3,rad=0.06",
        edge_color="#999999",
        ax=ax
    )
    nx.draw_networkx_nodes(
        DG, pos_norm,
        node_size=node_size,
        node_color="#ffffff",
        edgecolors="#2b2b2b",
        linewidths=1.2,
        ax=ax
    )
    nx.draw_networkx_labels(
        DG, pos_norm,
        labels=node_labels,
        font_size=font_size,
        font_color="black",
        ax=ax
    )
    if show_edge_labels:
        nx.draw_networkx_edge_labels(DG, pos_norm, edge_labels=edge_labels, font_size=max(font_size-1, 8), ax=ax)

    # --- Place thumbnails offset from nodes; keep on-screen; add connectors ---
    def _clamp(val, lo, hi): return max(lo, min(hi, val))
    safety = 0.01  # extra safety margin for thumbs
    _missing_logged = False

    for n, (x, y) in pos_norm.items():
        # Prefer id-based lookup; fall back to text. Also try the node key 'n'.
        ent_text_raw = (G.nodes[n].get("text", "") or "").strip()
        ent_text_key = ent_text_raw.casefold()
        id_candidates = [
            G.nodes[n].get("id"),
            G.nodes[n].get("ent_id"),
            str(n)  # node key itself often equals the ent_id from text_hpg
        ]
        thumb = None
        for cid in id_candidates:
            if cid and str(cid) in ent2thumb_by_id:
                thumb = ent2thumb_by_id[str(cid)]
                break
        if thumb is None and ent_text_key in ent2thumb_by_text:
            thumb = ent2thumb_by_text[ent_text_key]

        if thumb is None:
            if not _missing_logged:
                print("[HPG] Warning: missing thumbnail for some nodes. Example:",
                      {"node": str(n), "text": ent_text_raw, "id_candidates": id_candidates[:2]})
                _missing_logged = True
            continue

        if rankdir == "TB":
            # Prefer above; flip below if hitting the top
            left   = x - thumb_box / 2.0
            bottom = y + max(thumb_offset, margin_y + safety)
            if bottom + thumb_box > 1 - safety:
                bottom = y - max(thumb_offset, margin_y + safety) - thumb_box
            left = _clamp(left, safety, 1 - safety - thumb_box)
        else:  # LR
            # Prefer right; flip left if hitting the right edge
            left   = x + max(thumb_offset, margin_x + safety)
            bottom = y - thumb_box / 2.0
            if left + thumb_box > 1 - safety:
                left = x - max(thumb_offset, margin_x + safety) - thumb_box
            bottom = _clamp(bottom, safety, 1 - safety - thumb_box)

        # Connector line from node center to thumbnail center
        cx = left + thumb_box / 2.0
        cy = bottom + thumb_box / 2.0
        ax.plot([x, cx], [y, cy], transform=ax.transAxes,
                linewidth=0.8, alpha=0.55, linestyle="-", zorder=2, color="#888888")

        # Thumbnail box (axes fraction coords)
        t_ax = ax.inset_axes([left, bottom, thumb_box, thumb_box], transform=ax.transAxes)
        t_ax.imshow(thumb)
        t_ax.axis("off")
        t_ax.set_zorder(2.0)

    ax.set_title("HPG + Grounded Segments", fontsize=14, color="black", pad=8)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
