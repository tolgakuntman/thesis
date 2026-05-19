"""
scripts/visualize_graphs_v2.py

Visualize a sample of built HeteroData graphs as ego-graph diagrams.

Usage:
    python scripts/visualize_graphs_v2.py                    # 10 random graphs
    python scripts/visualize_graphs_v2.py --n 6              # 6 graphs
    python scripts/visualize_graphs_v2.py --vcc_only          # VCC commits only
    python scripts/visualize_graphs_v2.py --hashes abc123...  # specific commits
    python scripts/visualize_graphs_v2.py --out my_fig.png    # custom output path
"""

from __future__ import annotations

import argparse
import random
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import torch

ROOT       = Path(__file__).resolve().parents[1]
GRAPHS_DIR = ROOT / "outputs" / "graph_ready_v2" / "graphs"
OUT_DEFAULT = ROOT / "outputs" / "graph_ready_v2" / "graph_samples.png"

# ── Visual config ──────────────────────────────────────────────────────────────

NODE_COLORS = {
    "commit":       "#E74C3C",   # red  — VCC
    "commit_neg":   "#3498DB",   # blue — normal/FC
    "function":     "#2ECC71",   # green
    "file":         "#F39C12",   # orange
    "hunk":         "#9B59B6",   # purple
    "developer":    "#F1C40F",   # yellow
    "issue":        "#E91E63",   # pink
    "pull_request": "#00BCD4",   # cyan
    "release_tag":  "#95A5A6",   # grey
}

NODE_LABELS = {
    "commit":       "C",
    "function":     "fn",
    "file":         "f",
    "hunk":         "h",
    "developer":    "dev",
    "issue":        "iss",
    "pull_request": "pr",
    "release_tag":  "rt",
}

EDGE_COLORS = {
    ("commit", "modifies_file",  "file"):         "#F39C12",
    ("commit", "modifies_func",  "function"):     "#2ECC71",
    ("commit", "modifies_hunk",  "hunk"):         "#9B59B6",
    ("commit", "authored_by",    "developer"):    "#F1C40F",
    ("commit", "committed_by",   "developer"):    "#F1C40F",
    ("commit", "has_issue",      "issue"):        "#E91E63",
    ("commit", "has_pr",         "pull_request"): "#00BCD4",
    ("commit", "has_release",    "release_tag"):  "#95A5A6",
    ("file",   "contains",       "function"):     "#27AE60",
    ("developer", "owns",        "file"):         "#E67E22",
}


def hetero_to_nx(data) -> nx.DiGraph:
    """Convert HeteroData to a directed NetworkX graph for layout/drawing."""
    G = nx.DiGraph()

    # Add nodes — global index: "{type}_{local_idx}"
    for ntype in data.node_types:
        n = data[ntype].x.size(0)
        for i in range(n):
            G.add_node(f"{ntype}_{i}", ntype=ntype)

    # Add edges (forward only to avoid clutter)
    forward_rels = {et for et in data.edge_types if et[2] != "commit"}
    for et in forward_rels:
        src_type, rel, dst_type = et
        ei = data[et].edge_index
        if ei.size(1) == 0:
            continue
        for s, d in zip(ei[0].tolist(), ei[1].tolist()):
            G.add_edge(f"{src_type}_{s}", f"{dst_type}_{d}", etype=et)

    return G


def star_layout(G: nx.DiGraph, commit_node: str) -> dict:
    """
    Radial layout: commit at center; each node type on its own ring.
    """
    import math

    type_order = ["file", "function", "hunk", "developer",
                  "issue", "pull_request", "release_tag"]
    by_type: dict[str, list[str]] = {}
    for node, attr in G.nodes(data=True):
        t = attr["ntype"]
        by_type.setdefault(t, []).append(node)

    pos = {commit_node: (0, 0)}
    ring_radius = 1.0

    for ring_i, ntype in enumerate(type_order):
        nodes = by_type.get(ntype, [])
        if not nodes:
            continue
        r = ring_radius * (ring_i + 1) * 0.6
        for j, node in enumerate(nodes):
            angle = 2 * math.pi * j / max(len(nodes), 1)
            pos[node] = (r * math.cos(angle), r * math.sin(angle))

    # Any node not placed yet (shouldn't happen)
    for node in G.nodes():
        if node not in pos:
            pos[node] = (0, 0)

    return pos


def draw_graph(ax: plt.Axes, data, title: str) -> None:
    G = hetero_to_nx(data)
    label = data.y.item()
    commit_type = getattr(data["commit"], "hash", "")[:8]
    repo = str(getattr(data["commit"], "repo_url", "")).split("/")[-2:]
    repo_str = "/".join(repo)

    commit_node = "commit_0"
    pos = star_layout(G, commit_node)

    # Node colors and sizes
    node_colors = []
    node_sizes  = []
    for node, attr in G.nodes(data=True):
        nt = attr["ntype"]
        if nt == "commit":
            color = NODE_COLORS["commit"] if label == 1 else NODE_COLORS["commit_neg"]
        else:
            color = NODE_COLORS.get(nt, "#BDC3C7")
        node_colors.append(color)
        node_sizes.append(400 if nt == "commit" else 150)

    # Edge colors
    edge_colors = []
    for u, v, attr in G.edges(data=True):
        et = attr.get("etype", ())
        # Use forward edge type (no rev)
        col = EDGE_COLORS.get(et, "#BDC3C7")
        edge_colors.append(col)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                           arrows=True, arrowsize=8, alpha=0.6,
                           connectionstyle="arc3,rad=0.1", width=0.8)

    # Node labels: type abbreviation
    labels = {n: NODE_LABELS.get(attr["ntype"], "?")
              for n, attr in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=5,
                            font_color="white", font_weight="bold")

    # Count per type
    type_counts = {}
    for _, attr in G.nodes(data=True):
        type_counts[attr["ntype"]] = type_counts.get(attr["ntype"], 0) + 1

    n_nodes = sum(type_counts.values())
    n_edges = G.number_of_edges()
    tag = "VCC" if label == 1 else "neg"
    ax.set_title(
        f"{tag} | {commit_type}…\n{repo_str}\n"
        f"fn={type_counts.get('function',0)} f={type_counts.get('file',0)} "
        f"h={type_counts.get('hunk',0)} dev={type_counts.get('developer',0)} "
        f"| {n_nodes}N {n_edges}E",
        fontsize=7, pad=4
    )
    ax.axis("off")


def make_legend(fig: plt.Figure) -> None:
    handles = [
        mpatches.Patch(color=NODE_COLORS["commit"],     label="commit (VCC)"),
        mpatches.Patch(color=NODE_COLORS["commit_neg"], label="commit (neg)"),
        mpatches.Patch(color=NODE_COLORS["function"],   label="function"),
        mpatches.Patch(color=NODE_COLORS["file"],       label="file"),
        mpatches.Patch(color=NODE_COLORS["hunk"],       label="hunk"),
        mpatches.Patch(color=NODE_COLORS["developer"],  label="developer"),
        mpatches.Patch(color=NODE_COLORS["issue"],      label="issue"),
        mpatches.Patch(color=NODE_COLORS["pull_request"], label="pull_request"),
        mpatches.Patch(color=NODE_COLORS["release_tag"], label="release_tag"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=7, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))


def load_graphs(pt_files: list[Path]) -> list:
    graphs = []
    for p in pt_files:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = torch.load(p, weights_only=False)
        graphs.append(g)
    return graphs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",        type=int, default=10, help="Number of graphs to show")
    parser.add_argument("--vcc_only", action="store_true",  help="Show only VCC commits")
    parser.add_argument("--neg_only", action="store_true",  help="Show only negative commits")
    parser.add_argument("--hashes",   nargs="+", default=None)
    parser.add_argument("--out",      default=str(OUT_DEFAULT))
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    all_pts = sorted(GRAPHS_DIR.glob("*.pt"))
    if not all_pts:
        raise SystemExit(f"No graphs found in {GRAPHS_DIR}. Run build_graphs_v2.py first.")

    if args.hashes:
        pts = [GRAPHS_DIR / f"{h}.pt" for h in args.hashes if (GRAPHS_DIR / f"{h}.pt").exists()]
    else:
        pts = list(all_pts)

    # Pre-filter by label if requested (peek at y without loading full graph)
    if args.vcc_only or args.neg_only:
        filtered = []
        for p in pts:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g = torch.load(p, weights_only=False)
            if args.vcc_only and g.y.item() == 1:
                filtered.append(p)
            elif args.neg_only and g.y.item() == 0:
                filtered.append(p)
        pts = filtered

    pts = random.sample(pts, min(args.n, len(pts)))
    graphs = load_graphs(pts)

    # Layout: up to 5 per row
    ncols = min(5, len(graphs))
    nrows = (len(graphs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 3.5 + 0.8))
    axes = [axes] if nrows * ncols == 1 else list(axes.flatten() if nrows > 1 else axes)

    for i, (g, ax) in enumerate(zip(graphs, axes)):
        draw_graph(ax, g, title=f"Graph {i+1}")
    for ax in axes[len(graphs):]:
        ax.axis("off")

    make_legend(fig)
    fig.suptitle("graph_ready_v2 — sampled commit ego-graphs", fontsize=11, y=1.01)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
