"""
src/model.py

Heterogeneous GNN for commit-level VCC detection.

Primary model: HeteroSAGE
  - Input projection per node type → shared hidden dimension
  - 2-layer HeteroConv with edge-aware relation operators where edge_attr exists
  - Commit node (index 0 per graph) readout → binary classifier

Ablation model: HeteroRGCN (same interface, swaps SAGEConv for RGCNConv)

Also exports: FocalLoss

Node feature dimensions are intentionally lazy-initialized so the same training
code can consume both the legacy graph-ready package and the finalized package
graphs after migration.

Edge types (all bidirectional):
    commit ↔ file           (modifies_file / in_commit)
    file   ↔ function       (contains / in_file)
    commit ↔ function       (modifies_func / in_commit_fn)
    commit ↔ hunk           (modifies_hunk / in_commit_hunk)
    commit ↔ developer      (authored_by / authored)
    commit ↔ developer      (committed_by / committed)
    developer ↔ file        (owns / owned_by)
    commit ↔ issue          (has_issue / linked_to_commit)
    commit ↔ pull_request   (has_pr / linked_to_commit)
    commit ↔ release_tag    (has_release / release_of)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter
from torch_geometric.nn import GATv2Conv, HeteroConv, SAGEConv, RGCNConv

# ── constants ─────────────────────────────────────────────────────────────────

NODE_TYPES = [
    "commit",
    "function",
    "file",
    "hunk",
    "developer",
    "issue",
    "pull_request",
    "release_tag",
]

EDGE_TYPES: list[tuple[str, str, str]] = [
    ("commit",       "modifies_file",      "file"),
    ("file",         "in_commit",          "commit"),
    ("file",         "contains",           "function"),
    ("function",     "in_file",            "file"),
    ("commit",       "modifies_func",      "function"),
    ("function",     "in_commit_fn",       "commit"),
    ("commit",       "modifies_hunk",      "hunk"),
    ("hunk",         "in_commit_hunk",     "commit"),
    ("commit",       "authored_by",        "developer"),
    ("developer",    "authored",           "commit"),
    ("commit",       "committed_by",       "developer"),
    ("developer",    "committed",          "commit"),
    ("developer",    "owns",               "file"),
    ("file",         "owned_by",           "developer"),
    ("commit",       "has_issue",          "issue"),
    ("issue",        "linked_to_commit",   "commit"),
    ("commit",       "has_pr",             "pull_request"),
    ("pull_request", "linked_to_commit",   "commit"),
    ("commit",       "has_release",        "release_tag"),
    ("release_tag",  "release_of",         "commit"),
]

EDGE_ATTR_DIMS: dict[tuple[str, str, str], int] = {
    ("commit", "modifies_file", "file"): 4,
    ("file", "in_commit", "commit"): 4,
    ("commit", "modifies_func", "function"): 11,
    ("function", "in_commit_fn", "commit"): 11,
    ("commit", "authored_by", "developer"): 3,
    ("developer", "authored", "commit"): 3,
    ("commit", "committed_by", "developer"): 3,
    ("developer", "committed", "commit"): 3,
    ("developer", "owns", "file"): 3,
    ("file", "owned_by", "developer"): 3,
    ("commit", "has_issue", "issue"): 3,
    ("issue", "linked_to_commit", "commit"): 3,
    ("commit", "has_pr", "pull_request"): 3,
    ("pull_request", "linked_to_commit", "commit"): 3,
    ("commit", "has_release", "release_tag"): 1,
    ("release_tag", "release_of", "commit"): 1,
}


def build_relation_convs(hidden: int) -> dict[tuple[str, str, str], nn.Module]:
    convs: dict[tuple[str, str, str], nn.Module] = {}
    for et in EDGE_TYPES:
        edge_dim = EDGE_ATTR_DIMS.get(et)
        if edge_dim is not None:
            convs[et] = GATv2Conv(
                (hidden, hidden),
                hidden,
                heads=1,
                concat=False,
                add_self_loops=False,
                edge_dim=edge_dim,
            )
        else:
            convs[et] = SAGEConv(hidden, hidden, aggr="mean")
    return convs


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_norm_act(x: torch.Tensor, ln: nn.LayerNorm, p: float, training: bool) -> torch.Tensor:
    """LayerNorm + ReLU + Dropout. Safe for empty tensors and single-sample batches."""
    if x.size(0) == 0:
        return x
    return F.dropout(F.relu(ln(x)), p=p, training=training)


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss for binary classification (Lin et al., ICCV 2017).

    Args:
        gamma : focusing parameter (default 2.0) — down-weights easy negatives
        alpha : weight for the positive class (default 0.75)
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : [N] raw (pre-sigmoid) scores
            targets : [N] int64 labels {0, 1}
        Returns:
            scalar mean focal loss
        """
        targets_f = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets_f, reduction="none")
        probs = torch.sigmoid(logits.detach())
        p_t    = probs * targets_f + (1 - probs) * (1 - targets_f)
        alpha_t = self.alpha * targets_f + (1 - self.alpha) * (1 - targets_f)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


# ── Primary model: HeteroSAGE ─────────────────────────────────────────────────

class HeteroSAGE(nn.Module):
    """
    2-layer Heterogeneous GraphSAGE for commit-level VCC detection.

    Architecture:
        1. Per-type input projection: Linear(feat_dim → hidden) + ReLU
        2. HeteroConv Layer 1: SAGEConv(hidden → hidden) per edge type
           + LayerNorm + ReLU + Dropout per node type
        3. HeteroConv Layer 2: same
        4. Commit node readout: Linear(hidden → 1)
           — each graph has exactly 1 commit node, so after batching
             h['commit'] has shape [batch_size, hidden]

    Args:
        hidden  : hidden dimension (default 128)
        dropout : dropout probability (default 0.3)
    """

    def __init__(self, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.hidden  = hidden
        self.dropout_p = dropout

        # Lazy projections let the same model consume both legacy and final-package
        # graphs without hard-coding input dimensions per node type.
        self.input_proj = nn.ModuleDict({
            nt: nn.LazyLinear(hidden) for nt in NODE_TYPES
        })

        # HeteroConv layers
        self.conv1 = HeteroConv(build_relation_convs(hidden), aggr="sum")
        self.conv2 = HeteroConv(build_relation_convs(hidden), aggr="sum")

        # LayerNorm after each conv (per node type, applied per-token → safe for any N)
        self.ln1 = nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in NODE_TYPES})
        self.ln2 = nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in NODE_TYPES})

        # Final classifier
        self.classifier = nn.Linear(hidden, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if isinstance(m.weight, UninitializedParameter):
                    continue
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _project_inputs(self, x_dict: dict) -> dict:
        """Project each node type to hidden dim. Returns zeros for missing/empty types."""
        h = {}
        for nt in NODE_TYPES:
            if nt in x_dict and x_dict[nt].size(0) > 0:
                h[nt] = F.relu(self.input_proj[nt](x_dict[nt]))
            else:
                # Placeholder empty tensor — HeteroConv skips edges to/from empty types
                device = next(self.parameters()).device
                h[nt] = torch.zeros(0, self.hidden, device=device, dtype=torch.float)
        return h

    def _conv_and_norm(
        self,
        h: dict,
        conv: HeteroConv,
        ln: nn.ModuleDict,
        edge_index_dict: dict,
        edge_attr_dict: dict | None = None,
    ) -> dict:
        """Run one HeteroConv layer, apply LayerNorm+ReLU+Dropout, preserve missing node types."""
        # Filter edge_index_dict to only include edge types whose src/dst nodes are non-empty
        active_edges = {
            et: ei for et, ei in edge_index_dict.items()
            if et[0] in h and h[et[0]].size(0) > 0
            and et[2] in h and h[et[2]].size(0) > 0
        }

        active_edge_attr = {}
        if edge_attr_dict:
            for et, ea in edge_attr_dict.items():
                if et in active_edges and ea is not None and ea.numel() > 0:
                    active_edge_attr[et] = ea

        out = conv(h, active_edges, edge_attr_dict=active_edge_attr)  # only updated node types returned

        result = {}
        for nt in NODE_TYPES:
            x_in = h[nt]
            if x_in.size(0) == 0:
                result[nt] = x_in  # stay empty
            elif nt in out:
                result[nt] = _safe_norm_act(out[nt], ln[nt], self.dropout_p, self.training)
            else:
                # No incoming messages in this layer — keep projected input unchanged
                result[nt] = x_in
        return result

    def forward(
        self,
        x_dict: dict,
        edge_index_dict: dict,
        edge_attr_dict: dict | None = None,
        batch: dict | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x_dict          : {node_type: tensor [N_type, feat_dim]}
            edge_index_dict : {(src, rel, dst): tensor [2, E]}
            batch           : unused (kept for API compatibility with PyG Batch)

        Returns:
            logits [batch_size] — one score per graph (pre-sigmoid)
        """
        h = self._project_inputs(x_dict)
        h = self._conv_and_norm(h, self.conv1, self.ln1, edge_index_dict, edge_attr_dict)
        h = self._conv_and_norm(h, self.conv2, self.ln2, edge_index_dict, edge_attr_dict)

        # Readout: each graph has exactly 1 commit node
        # After batching, h['commit'] is [batch_size, hidden]
        commit_emb = h["commit"]                    # [B, hidden]
        logits = self.classifier(commit_emb).squeeze(-1)  # [B]
        return logits


# ── Ablation model: HeteroRGCN ────────────────────────────────────────────────

class HeteroRGCN(nn.Module):
    """
    2-layer R-GCN ablation (Schlichtkrull et al., ESWC 2018).

    Identical interface to HeteroSAGE. Uses RGCNConv with basis decomposition
    instead of SAGEConv — transductive aggregation, no mean-pooling of neighbourhoods.

    Args:
        hidden      : hidden dimension (default 128)
        dropout     : dropout probability (default 0.3)
        num_bases   : R-GCN basis matrices (default 3)
    """

    def __init__(self, hidden: int = 128, dropout: float = 0.3, num_bases: int = 3):
        super().__init__()
        raise NotImplementedError(
            "HeteroRGCN is not currently a real R-GCN ablation in this codebase. "
            "Use HeteroSAGE until a proper relation-specific R-GCN implementation is added."
        )
        self.hidden    = hidden
        self.dropout_p = dropout
        n_rels = len(EDGE_TYPES)

        self.input_proj = nn.ModuleDict({
            nt: nn.LazyLinear(hidden) for nt in NODE_TYPES
        })

        # RGCNConv treats all relations in a flat list — we implement as HeteroConv
        self.conv1 = HeteroConv(build_relation_convs(hidden), aggr="sum")
        self.conv2 = HeteroConv(build_relation_convs(hidden), aggr="sum")
        # NOTE: true R-GCN basis decomposition is applied at the relation-weight level.
        # Full R-GCN with basis decomp across all 14 relations:
        self._basis = nn.Parameter(torch.randn(num_bases, hidden, hidden) * 0.02)
        self._coeff1 = nn.Parameter(torch.randn(n_rels, num_bases) * 0.02)
        self._coeff2 = nn.Parameter(torch.randn(n_rels, num_bases) * 0.02)

        self.ln1 = nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in NODE_TYPES})
        self.ln2 = nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in NODE_TYPES})
        self.classifier = nn.Linear(hidden, 1)

    def _project_inputs(self, x_dict):
        h = {}
        for nt in NODE_TYPES:
            if nt in x_dict and x_dict[nt].size(0) > 0:
                h[nt] = F.relu(self.input_proj[nt](x_dict[nt]))
            else:
                device = next(self.parameters()).device
                h[nt] = torch.zeros(0, self.hidden, device=device, dtype=torch.float)
        return h

    def _conv_and_norm(self, h, conv, ln, edge_index_dict, edge_attr_dict=None):
        active_edges = {
            et: ei for et, ei in edge_index_dict.items()
            if et[0] in h and h[et[0]].size(0) > 0
            and et[2] in h and h[et[2]].size(0) > 0
        }
        active_edge_attr = {}
        if edge_attr_dict:
            for et, ea in edge_attr_dict.items():
                if et in active_edges and ea is not None and ea.numel() > 0:
                    active_edge_attr[et] = ea
        out = conv(h, active_edges, edge_attr_dict=active_edge_attr)
        result = {}
        for nt in NODE_TYPES:
            x_in = h[nt]
            if x_in.size(0) == 0:
                result[nt] = x_in
            elif nt in out:
                result[nt] = _safe_norm_act(out[nt], ln[nt], self.dropout_p, self.training)
            else:
                result[nt] = x_in
        return result

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None, batch=None):
        h = self._project_inputs(x_dict)
        h = self._conv_and_norm(h, self.conv1, self.ln1, edge_index_dict, edge_attr_dict)
        h = self._conv_and_norm(h, self.conv2, self.ln2, edge_index_dict, edge_attr_dict)
        commit_emb = h["commit"]
        return self.classifier(commit_emb).squeeze(-1)
