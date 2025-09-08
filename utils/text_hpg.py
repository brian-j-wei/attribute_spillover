"""
text_hpg.py — Build a text Hierarchical Parsing Graph (HPG) from a caption using Stanza

Exports:
  - build_text_hpg(caption: str, lang: str = "en", pipeline: Optional[stanza.Pipeline] = None)
      -> Tuple[nx.MultiDiGraph, Dict]
  - visualize_text_hpg(G: nx.MultiDiGraph, *, title: str | None = None,
                       save_path: str | None = None, node_font_size: int = 10,
                       edge_font_size: int = 9) -> None

Notes
-----
- Nodes represent entities (NER spans and/or noun phrases).
- Directed edges connect entities that are related syntactically; each edge is
  labeled with the governing verb (and preposition, if present), e.g.,
  "sit-on", "look-at", "stand-near".
- If Stanza models are not downloaded yet, call `stanza.download(lang)` once
  in your environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import networkx as nx
import stanza

try:
    import matplotlib.pyplot as plt  # Only needed for visualization
except Exception:  # pragma: no cover
    plt = None


# ----------------------------- Data Structures ----------------------------- #
@dataclass(frozen=True)
class Entity:
    id: str
    text: str
    head_lemma: str
    sent_id: int
    token_indices: Tuple[int, ...]  # 1-based indices within sentence


# ------------------------------ Core Functions ----------------------------- #

def _ensure_pipeline(lang: str, pipeline: Optional[stanza.Pipeline]) -> stanza.Pipeline:
    if pipeline is not None:
        return pipeline
    return stanza.Pipeline(
        lang=lang,
        processors="tokenize,pos,lemma,depparse,ner",
        tokenize_pretokenized=False,
        use_gpu=False  # <-- Force Stanza to run on CPU only
    )



def _span_text(tokens, idxs: List[int]) -> str:
    # tokens are 1-based in Stanza sentence; idxs are 1-based too
    words = [tokens[i - 1].text for i in sorted(idxs)]
    return " ".join(words)


def _collect_ner_entities(sent, sent_id: int) -> List[Entity]:
    ents: List[Entity] = []
    for i, ner in enumerate(sent.ents):
        idxs = list(range(ner.start_char, ner.end_char))  # Character positions, not tokens
        # Stanza's NER spans are char offsets; we need token indices. Map chars→tokens.
        tok_idxs: List[int] = []
        for tok in sent.tokens:
            for w in tok.words:
                # If any character of the word overlaps with the NER span, include token index
                if not (w.end_char <= ner.start_char or w.start_char >= ner.end_char):
                    tok_idxs.append(w.id)
        if not tok_idxs:
            continue
        head_idx = tok_idxs[-1]
        head_lemma = sent.words[head_idx - 1].lemma
        text = sent.text[ner.start_char:ner.end_char]
        ents.append(Entity(id=f"ent_s{sent_id}_{len(ents)}", text=text, head_lemma=head_lemma,
                           sent_id=sent_id, token_indices=tuple(sorted(set(tok_idxs)))))
    return ents


def _collect_np_entities(sent, sent_id: int, occupied_tokens: Set[int]) -> List[Entity]:
    """Backoff: derive simple noun-phrase entities from dependencies.
    We gather heads with UPOS in {NOUN, PROPN, PRON} and attach left modifiers.
    """
    words = sent.words
    np_ents: List[Entity] = []
    for w in words:
        if w.upos not in {"NOUN", "PROPN", "PRON"}:
            continue
        head = w
        # collect modifiers attached to head (compound, amod, det, nummod)
        idxs = {head.id}
        for c in words:
            if c.head == head.id and c.deprel in {"compound", "amod", "det", "nummod", "flat", "fixed"}:
                idxs.add(c.id)
        # Skip if all tokens are already covered by a NER entity
        if set(idxs).issubset(occupied_tokens):
            continue
        text = _span_text(sent.words, list(idxs))
        ent = Entity(id=f"ent_s{sent_id}_{len(np_ents)}_np", text=text, head_lemma=head.lemma,
                     sent_id=sent_id, token_indices=tuple(sorted(idxs)))
        np_ents.append(ent)
    return np_ents


def _index_entities_by_token(entities: List[Entity]) -> Dict[int, Entity]:
    by_tok: Dict[int, Entity] = {}
    for e in entities:
        for tid in e.token_indices:
            by_tok[tid] = e
    return by_tok


def _find_relation_label(head_word, preposition: Optional[str] = None) -> str:
    base = head_word.lemma if head_word.lemma else head_word.text
    if preposition:
        return f"{base}-{preposition}"
    return base


def _add_edge(edges: List[Tuple[str, str, str]], src: Optional[Entity], dst: Optional[Entity], label: str) -> None:
    if src is None or dst is None:
        return
    if src.id == dst.id:
        return
    edges.append((src.id, dst.id, label))


def _build_edges_for_sentence(sent, entities: List[Entity]) -> List[Tuple[str, str, str]]:
    """Return list of (u, v, label) edges for one sentence.
    Heuristics used:
      - Verb-centric: nsubj → obj(iobj) with label verb lemma.
      - Prepositional attachments: nsubj → obl/nmod with label verb+prep.
      - Copulas: subject → complement with the predicate lemma.
      - Conjunctions: ent ↔ ent with label "and"/"or" if coordinated.
      - NEW: Conjunct propagation — if a subject/object/oblique is part of a
        coordination (e.g., "laptop and backpack"), propagate the role to
        the coordinated siblings so both get edges to the same verb.
    """
    edges: List[Tuple[str, str, str]] = []
    tok2ent = _index_entities_by_token(entities)
    words = sent.words

    # Map from token id to word for convenience
    by_id = {w.id: w for w in words}

    # Helper to map any token id to its entity (preferring exact head match)
    def ent_for_token(tid: int) -> Optional[Entity]:
        return tok2ent.get(tid)

    def expand_with_conj(seed_tids: List[int]) -> List[int]:
        """Include conjunct siblings (x conj y) for each seed token id."""
        out: Set[int] = set(seed_tids)
        changed = True
        while changed:
            changed = False
            current = list(out)
            for tid in current:
                # siblings that are conj of this token
                sibs = [w.id for w in words if w.head == tid and w.deprel == "conj"]
                for s in sibs:
                    if s not in out:
                        out.add(s)
                        changed = True
        return sorted(out)

    # Build subject/object maps for each verb head
    for head in words:
        if head.upos not in {"VERB", "AUX"}:
            continue
        subj_tokens = [w.id for w in words if w.head == head.id and w.deprel.startswith("nsubj")]
        obj_tokens = [w.id for w in words if w.head == head.id and w.deprel in {"obj", "iobj"}]
        obl_tokens = [w.id for w in words if w.head == head.id and w.deprel in {"obl", "nmod"}]

        # Propagate roles to conjuncts (e.g., "laptop and backpack" → both are subjects)
        subj_tokens = expand_with_conj(subj_tokens)
        obj_tokens = expand_with_conj(obj_tokens)
        obl_tokens = expand_with_conj(obl_tokens)

        preps: Dict[int, str] = {}
        for tid in obl_tokens:
            # Look for a case-marking adposition under this token
            adps = [w for w in words if w.head == tid and w.deprel == "case"]
            if adps:
                preps[tid] = adps[0].lemma

        for s_tid in subj_tokens:
            s_ent = ent_for_token(s_tid)
            # subject → object (direct)
            for o_tid in obj_tokens:
                o_ent = ent_for_token(o_tid)
                _add_edge(edges, s_ent, o_ent, _find_relation_label(head))
            # subject → oblique (with preposition)
            for o_tid in obl_tokens:
                o_ent = ent_for_token(o_tid)
                prep = preps.get(o_tid)
                _add_edge(edges, s_ent, o_ent, _find_relation_label(head, prep))

    # Copular constructions: x is/on/at y
    for w in words:
        if w.deprel == "cop":
            pred = by_id.get(w.head)  # the predicate (adjectival/noun) that has a copula child
            if not pred:
                continue
            # Find subject of the predicate
            subj = next((x for x in words if x.head == pred.id and x.deprel.startswith("nsubj")), None)
            comp_ent = ent_for_token(pred.id)
            subj_ent = ent_for_token(subj.id) if subj else None
            if comp_ent and subj_ent and subj_ent.id != comp_ent.id:
                _add_edge(edges, subj_ent, comp_ent, pred.lemma or "be")

    # Coordinations (and/or) between entity heads (bidirectional labeled with cc)
    for w in words:
        if w.deprel == "cc":
            conj = by_id.get(w.head)
            if not conj:
                continue
            # Find the coordinated element
            sibling = next((x for x in words if x.head == conj.id and x.deprel == "conj"), None)
            if sibling:
                e1 = ent_for_token(conj.id)
                e2 = ent_for_token(sibling.id)
                if e1 and e2 and e1.id != e2.id:
                    _add_edge(edges, e1, e2, w.lemma)
                    _add_edge(edges, e2, e1, w.lemma)

    return edges


def build_text_hpg(caption: str, lang: str = "en", pipeline: Optional[stanza.Pipeline] = None) -> Tuple[nx.MultiDiGraph, Dict]:
    """Build a text HPG from a single caption.

    Parameters
    ----------
    caption : str
        The input caption/sentence(s).
    lang : str
        Language code for Stanza (default: "en").
    pipeline : stanza.Pipeline | None
        Optional pre-initialized Stanza pipeline.

    Returns
    -------
    (G, payload) : (nx.MultiDiGraph, dict)
        Graph with entity nodes and labeled edges; and a serializable payload:
        {
          "nodes": [{"id", "text", "head_lemma", "sent_id", "token_indices"}, ...],
          "edges": [{"source", "target", "label"}, ...],
          "sentences": [str, ...]
        }
    """
    nlp = _ensure_pipeline(lang, pipeline)
    doc = nlp(caption)

    all_entities: List[Entity] = []
    all_edges: List[Tuple[str, str, str]] = []

    sent_texts: List[str] = []
    for s_id, sent in enumerate(doc.sentences):
        sent_texts.append(sent.text)

        ner_ents = _collect_ner_entities(sent, s_id)
        occupied = {tid for e in ner_ents for tid in e.token_indices}
        np_ents = _collect_np_entities(sent, s_id, occupied)

        entities = ner_ents + np_ents
        all_entities.extend(entities)

        edges = _build_edges_for_sentence(sent, entities)
        all_edges.extend(edges)

    # Build graph
    G = nx.MultiDiGraph()
    for e in all_entities:
        G.add_node(e.id, text=e.text, head_lemma=e.head_lemma, sent_id=e.sent_id, token_indices=e.token_indices)
    for u, v, lbl in all_edges:
        if u in G and v in G:
            G.add_edge(u, v, label=lbl)

    payload = {
        "nodes": [
            {
                "id": e.id,
                "text": e.text,
                "head_lemma": e.head_lemma,
                "sent_id": e.sent_id,
                "token_indices": list(e.token_indices),
            }
            for e in all_entities
        ],
        "edges": [
            {"source": u, "target": v, "label": lbl}
            for (u, v, lbl) in all_edges
        ],
        "sentences": sent_texts,
    }

    return G, payload


# --- NEW: helpers for hierarchical visualization --------------------------------
def _prune_for_tree(G: nx.MultiDiGraph, *, drop_cc_labels=("and", "or")) -> nx.DiGraph:
    """
    Make the HPG more tree-like for visualization:
      - Convert to simple DiGraph (collapse multi-edges).
      - Drop coordination edges (and/or) that criss-cross layers.
      - Break small cycles by removing the back-edge with the shortest label.
    Returns a DAG (best-effort).
    """
    # 1) collapse to DiGraph with representative labels
    DG = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        lbl = str(data.get("label", "rel")).lower()
        if lbl in drop_cc_labels:
            continue
        if DG.has_edge(u, v):
            continue
        DG.add_edge(u, v, label=lbl)

    # 2) try to break cycles conservatively
    try:
        while not nx.is_directed_acyclic_graph(DG):
            cycle = next(nx.simple_cycles(DG))  # list of nodes in a cycle
            # remove one "weak" edge from the cycle (heuristic: shortest label)
            candidate_edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]
            edge_to_remove = min(candidate_edges, key=lambda e: len(DG[e[0]][e[1]].get("label", "")))
            DG.remove_edge(*edge_to_remove)
    except nx.NetworkXNoCycle:
        pass
    except StopIteration:
        pass

    return DG


def _hierarchical_positions(DG: nx.DiGraph, *, rankdir: str = "TB", layer_gap: float = 1.6, node_gap: float = 1.4):
    """
    Compute layered positions without Graphviz.
    rankdir: "TB" (top->bottom) or "LR" (left->right)
    """
    # Roots = nodes with no incoming edges; if none, pick nodes with minimal indegree
    roots = [n for n in DG.nodes if DG.in_degree(n) == 0]
    if not roots:
        min_in = min((DG.in_degree(n) for n in DG.nodes))
        roots = [n for n in DG.nodes if DG.in_degree(n) == min_in]

    # BFS layering from (possibly multiple) roots
    layers = []
    visited = set()
    frontier = list(roots)
    while frontier:
        layers.append(frontier)
        visited.update(frontier)
        next_frontier = []
        for u in frontier:
            for v in DG.successors(u):
                if v not in visited and all((p in visited) for p in DG.predecessors(v)):
                    next_frontier.append(v)
        # add any isolated/unreached nodes as their own layer at the end
        frontier = list(dict.fromkeys(next_frontier))  # unique & stable

    # Add stragglers (disconnected)
    remaining = [n for n in DG.nodes if n not in visited]
    if remaining:
        layers.append(remaining)

    # Lay out
    pos = {}
    for li, layer in enumerate(layers):
        # Stable order: by out-degree descending then by node id
        layer_sorted = sorted(layer, key=lambda n: (-DG.out_degree(n), str(n)))
        for j, n in enumerate(layer_sorted):
            if rankdir == "TB":
                x = j * node_gap
                y = -li * layer_gap
            else:  # LR
                x = li * layer_gap
                y = -j * node_gap
            pos[n] = (x, y)
    return pos


def _graphviz_positions_or_none(G: nx.DiGraph, *, rankdir: str = "TB"):
    """
    Try Graphviz 'dot' if available; otherwise return None.
    """
    try:
        # Prefer pygraphviz
        from networkx.drawing.nx_agraph import graphviz_layout  # type: ignore
        return graphviz_layout(G, prog="dot", args=f"-Grankdir={rankdir}")
    except Exception:
        try:
            # Fallback to pydot
            from networkx.drawing.nx_pydot import graphviz_layout  # type: ignore
            return graphviz_layout(G, prog="dot", args=f"-Grankdir={rankdir}")
        except Exception:
            return None


# --- REPLACE your visualize_text_hpg with this enhanced version -----------------
def visualize_text_hpg(
    G: nx.MultiDiGraph,
    *,
    title: str | None = None,
    save_path: str | None = None,
    node_font_size: int = 11,
    edge_font_size: int = 9,
    node_size: int = 2200,
    layout: str = "hier",           # "hier" | "spring"
    rankdir: str = "TB",            # "TB" (top->bottom) or "LR" (left->right)
    drop_coordination_edges: bool = True,
) -> None:
    """
    Visualize the HPG. Defaults to a hierarchical, tree-like layout.
    - White background, larger nodes, short tidy edges.
    - If Graphviz is installed, uses 'dot' for the cleanest hierarchy;
      otherwise, uses a pure-Python layered layout.

    Parameters
    ----------
    layout : "hier" | "spring"
        "hier" for tree-like layers; "spring" to fall back to force-directed.
    rankdir : "TB" | "LR"
        Direction for hierarchical layers (top-to-bottom or left-to-right).
    drop_coordination_edges : bool
        If True, removes 'and'/'or' edges that tangle the tree.
    """
    if plt is None:
        raise RuntimeError("matplotlib is required for visualization but is not available.")

    # Build a cleaned DAG for tree-like drawing
    DG = _prune_for_tree(G, drop_cc_labels=("and", "or") if drop_coordination_edges else ())

    # Node/edge labels
    node_labels = {n: (G.nodes[n].get("text") or n) for n in G.nodes}
    edge_labels = {(u, v): data.get("label", "rel") for u, v, data in DG.edges(data=True)}

    # Positions
    if layout == "hier":
        pos = _graphviz_positions_or_none(DG, rankdir=rankdir)
        if pos is None:
            pos = _hierarchical_positions(DG, rankdir=rankdir)
    else:
        pos = nx.spring_layout(DG, seed=42, k=0.9)

    # Figure aesthetics: white background, no border
    plt.figure(figsize=(11.5, 8.5), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")

    # Draw nodes (rounded-feel via large size), edges with arrows
    nx.draw_networkx_nodes(DG, pos, node_size=node_size, node_color="#ffffff", edgecolors="#2b2b2b", linewidths=1.2)
    nx.draw_networkx_labels(DG, pos, labels=node_labels, font_size=node_font_size)

    # Short, tidy edges; small curvature helps avoid overlap
    nx.draw_networkx_edges(
        DG, pos,
        arrowstyle="-|>", arrowsize=14, width=1.3,
        connectionstyle="arc3,rad=0.06"
    )

    nx.draw_networkx_edge_labels(DG, pos, edge_labels=edge_labels, font_size=edge_font_size, label_pos=0.5)

    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=220, facecolor="white", bbox_inches="tight")
        plt.close()
    else:
        plt.show()


