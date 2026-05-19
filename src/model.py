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

Structural exclusion:
    Pass exclude_node_types and/or exclude_edge_rels to model constructors to
    physically remove node types and edge relations from the message-passing graph.
    'commit' cannot be excluded (assertion). Edges whose src or dst node type is
    excluded are automatically dropped. Edge relations listed in exclude_edge_rels
    are dropped regardless of node availability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter
from torch_geometric.nn import GATv2Conv, HeteroConv, SAGEConv
from torch_geometric.nn.conv import MessagePassing

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


def _resolve_active_types(
    exclude_node_types: list[str] | tuple[str, ...],
    exclude_edge_rels: list[str] | tuple[str, ...],
) -> tuple[list[str], list[tuple[str, str, str]]]:
    """
    Compute the active node types and edge types given exclusion lists.

    Rules:
      - 'commit' cannot be excluded.
      - Edge types are dropped if their src or dst node type is excluded,
        OR if their relation name is in exclude_edge_rels.
    """
    excl_nodes = set(exclude_node_types)
    excl_rels  = set(exclude_edge_rels)
    assert "commit" not in excl_nodes, "'commit' node type cannot be excluded"

    active_nodes = [nt for nt in NODE_TYPES if nt not in excl_nodes]
    active_nodes_set = set(active_nodes)
    active_edges = [
        et for et in EDGE_TYPES
        if et[0] in active_nodes_set
        and et[2] in active_nodes_set
        and et[1] not in excl_rels
    ]
    return active_nodes, active_edges


def build_relation_convs(
    hidden: int,
    edge_types: list[tuple[str, str, str]] | None = None,
) -> dict[tuple[str, str, str], nn.Module]:
    """Build per-relation conv modules. Pass edge_types=None to use all EDGE_TYPES."""
    if edge_types is None:
        edge_types = EDGE_TYPES
    convs: dict[tuple[str, str, str], nn.Module] = {}
    for et in edge_types:
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

    Structural exclusion:
        exclude_node_types : node types to physically remove from the graph.
                             Their projections and LayerNorms are not built.
                             Edges touching excluded types are also removed.
        exclude_edge_rels  : relation names (middle element of edge triple)
                             to exclude, even if src/dst nodes are active.

    Args:
        hidden             : hidden dimension (default 128)
        dropout            : dropout probability (default 0.3)
        feat_dropout       : input feature dropout (default 0.0 = off)
        exclude_node_types : node types to structurally remove (default: none)
        exclude_edge_rels  : edge relations to structurally remove (default: none)
    """

    def __init__(
        self,
        hidden: int = 128,
        dropout: float = 0.3,
        feat_dropout: float = 0.0,
        exclude_node_types: list[str] | tuple[str, ...] = (),
        exclude_edge_rels:  list[str] | tuple[str, ...] = (),
    ):
        super().__init__()
        self.hidden  = hidden
        self.dropout_p = dropout
        self.feat_dropout_p = feat_dropout

        self._active_node_types, self._active_edge_types = _resolve_active_types(
            exclude_node_types, exclude_edge_rels
        )
        self._active_node_set = set(self._active_node_types)
        self._active_edge_set = set(self._active_edge_types)

        # Lazy projections only for active node types
        self.input_proj = nn.ModuleDict({
            nt: nn.LazyLinear(hidden) for nt in self._active_node_types
        })

        # HeteroConv layers — only active edge types
        self.conv1 = HeteroConv(build_relation_convs(hidden, self._active_edge_types), aggr="sum")
        self.conv2 = HeteroConv(build_relation_convs(hidden, self._active_edge_types), aggr="sum")

        # LayerNorm after each conv (only active node types)
        self.ln1 = nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in self._active_node_types})
        self.ln2 = nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in self._active_node_types})

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
        """Project each active node type to hidden dim. Inactive/missing types get empty tensor."""
        h = {}
        device = next(self.parameters()).device
        for nt in NODE_TYPES:
            if nt not in self._active_node_set:
                # Structurally excluded: empty tensor, no projection
                h[nt] = torch.zeros(0, self.hidden, device=device, dtype=torch.float)
            elif nt in x_dict and x_dict[nt].size(0) > 0:
                x = x_dict[nt]
                if self.feat_dropout_p > 0.0:
                    x = F.dropout(x, p=self.feat_dropout_p, training=self.training)
                h[nt] = F.relu(self.input_proj[nt](x))
            else:
                # Active but absent in this graph — placeholder
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
        # Filter to active edge types with non-empty src/dst
        active_edges = {
            et: ei for et, ei in edge_index_dict.items()
            if et in self._active_edge_set
            and et[0] in h and h[et[0]].size(0) > 0
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
                result[nt] = x_in  # stay empty (excluded or absent)
            elif nt in out:
                result[nt] = _safe_norm_act(out[nt], ln[nt], self.dropout_p, self.training)
            else:
                # Active but received no messages — keep projected input
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


# ── R-GCN basis conv ──────────────────────────────────────────────────────────

class RGCNBasisConv(MessagePassing):
    """
    Single-relation R-GCN message-passing layer with basis decomposition.

    W_r = Σ_b (coeff_b · basis_b)   (Schlichtkrull et al., ESWC 2018)

    The shared `basis` parameter is owned by the parent HeteroRGCN module and
    stored here via object.__setattr__ to avoid double-registration in the
    nn.Module parameter tree. Each instance owns its own `coeff` vector.

    Args:
        hidden    : node feature dimension
        num_bases : number of shared basis matrices
    """

    def __init__(self, hidden: int, num_bases: int):
        super().__init__(aggr="mean")
        self.hidden = hidden
        self.coeff = nn.Parameter(torch.empty(num_bases))
        nn.init.normal_(self.coeff, std=0.02)
        # Basis reference set after construction — bypasses nn.Module.__setattr__
        # so the shared tensor is not registered as a parameter of this module.
        object.__setattr__(self, "_basis_ref", None)

    def _set_basis(self, basis: torch.Tensor) -> None:
        object.__setattr__(self, "_basis_ref", basis)

    def forward(
        self,
        x: tuple[torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x_src, x_dst = x

        if edge_index.numel() == 0 or x_src.size(0) == 0:
            return x_dst.new_zeros(x_dst.size(0), self.hidden)

        # Compute relation weight matrix W_r  →  (H, H)
        W_r = torch.einsum("b,bhd->hd", self.coeff, self._basis_ref)
        x_src_t = x_src @ W_r  # (N_src, H)

        return self.propagate(
            edge_index,
            x=(x_src_t, x_dst),
            size=(x_src.size(0), x_dst.size(0)),
        )

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j


# ── Ablation model: HeteroRGCN ────────────────────────────────────────────────

class HeteroRGCN(nn.Module):
    """
    2-layer R-GCN ablation (Schlichtkrull et al., ESWC 2018).

    Identical interface to HeteroSAGE. Uses R-GCN basis decomposition instead of
    GATv2Conv+SAGEConv — all active edge types share `num_bases` weight matrices,
    with per-relation coefficient vectors mixing them: W_r = Σ_b(a_rb · V_b).

    Edge attributes are intentionally ignored — this is the structural ablation
    that measures how much the edge-feature convolutions in HeteroSAGE contribute.

    Structural exclusion:
        Same exclude_node_types / exclude_edge_rels interface as HeteroSAGE.

    Args:
        hidden             : hidden dimension (default 128)
        dropout            : dropout probability (default 0.3)
        num_bases          : R-GCN shared basis matrices (default 4)
        exclude_node_types : node types to structurally remove (default: none)
        exclude_edge_rels  : edge relations to structurally remove (default: none)
    """

    def __init__(
        self,
        hidden: int = 128,
        dropout: float = 0.3,
        num_bases: int = 4,
        exclude_node_types: list[str] | tuple[str, ...] = (),
        exclude_edge_rels:  list[str] | tuple[str, ...] = (),
    ):
        super().__init__()
        self.hidden = hidden
        self.dropout_p = dropout

        self._active_node_types, self._active_edge_types = _resolve_active_types(
            exclude_node_types, exclude_edge_rels
        )
        self._active_node_set = set(self._active_node_types)
        self._active_edge_set = set(self._active_edge_types)

        self.input_proj = nn.ModuleDict({
            nt: nn.LazyLinear(hidden) for nt in self._active_node_types
        })

        # Shared basis parameters owned by this module — not by the individual convs.
        # Shape: (num_bases, hidden, hidden)
        self.basis1 = nn.Parameter(torch.empty(num_bases, hidden, hidden))
        self.basis2 = nn.Parameter(torch.empty(num_bases, hidden, hidden))
        nn.init.normal_(self.basis1, std=0.02)
        nn.init.normal_(self.basis2, std=0.02)

        def _make_convs(basis: nn.Parameter, edge_types: list) -> dict:
            convs = {}
            for et in edge_types:
                c = RGCNBasisConv(hidden, num_bases)
                c._set_basis(basis)
                convs[et] = c
            return convs

        self.conv1 = HeteroConv(_make_convs(self.basis1, self._active_edge_types), aggr="sum")
        self.conv2 = HeteroConv(_make_convs(self.basis2, self._active_edge_types), aggr="sum")

        self.ln1 = nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in self._active_node_types})
        self.ln2 = nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in self._active_node_types})

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
        h = {}
        device = next(self.parameters()).device
        for nt in NODE_TYPES:
            if nt not in self._active_node_set:
                h[nt] = torch.zeros(0, self.hidden, device=device, dtype=torch.float)
            elif nt in x_dict and x_dict[nt].size(0) > 0:
                h[nt] = F.relu(self.input_proj[nt](x_dict[nt]))
            else:
                h[nt] = torch.zeros(0, self.hidden, device=device, dtype=torch.float)
        return h

    def _conv_and_norm(
        self,
        h: dict,
        conv: HeteroConv,
        ln: nn.ModuleDict,
        edge_index_dict: dict,
    ) -> dict:
        """Run one HeteroConv layer (no edge attrs), apply LayerNorm+ReLU+Dropout."""
        active_edges = {
            et: ei for et, ei in edge_index_dict.items()
            if et in self._active_edge_set
            and et[0] in h and h[et[0]].size(0) > 0
            and et[2] in h and h[et[2]].size(0) > 0
        }
        out = conv(h, active_edges)
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

    def forward(
        self,
        x_dict: dict,
        edge_index_dict: dict,
        edge_attr_dict: dict | None = None,  # accepted but ignored — structural ablation
        batch: dict | None = None,
    ) -> torch.Tensor:
        h = self._project_inputs(x_dict)
        h = self._conv_and_norm(h, self.conv1, self.ln1, edge_index_dict)
        h = self._conv_and_norm(h, self.conv2, self.ln2, edge_index_dict)
        commit_emb = h["commit"]
        return self.classifier(commit_emb).squeeze(-1)


# ── No-graph MLP baseline ─────────────────────────────────────────────────────

class CommitMLP(nn.Module):
    """
    No-graph MLP baseline for commit-level VCC detection.

    Uses only commit node features — no message passing, no neighbourhood
    aggregation. Answers whether the GNN is genuinely using graph topology
    or merely acting as a feature propagator.

    Architecture:
        LazyLinear(hidden) + ReLU + Dropout
        Linear(hidden, hidden) + ReLU + Dropout
        Linear(hidden, 1)

    Same interface as HeteroSAGE/HeteroRGCN for drop-in use in train.py.
    All non-commit inputs (edge_index_dict, edge_attr_dict) are ignored.
    """

    def __init__(self, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.hidden    = hidden
        self.dropout_p = dropout

        self.proj   = nn.LazyLinear(hidden)
        self.fc1    = nn.Linear(hidden, hidden)
        self.ln1    = nn.LayerNorm(hidden)
        self.fc2    = nn.Linear(hidden, hidden)
        self.ln2    = nn.LayerNorm(hidden)
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

    def forward(
        self,
        x_dict: dict,
        edge_index_dict: dict,       # ignored
        edge_attr_dict: dict | None = None,  # ignored
        batch: dict | None = None,   # ignored
    ) -> torch.Tensor:
        commit_x = x_dict["commit"]  # [B, feat_dim]
        h = F.relu(self.proj(commit_x))
        h = F.dropout(h, p=self.dropout_p, training=self.training)
        h = F.relu(self.ln1(self.fc1(h)))
        h = F.dropout(h, p=self.dropout_p, training=self.training)
        h = F.relu(self.ln2(self.fc2(h)))
        return self.classifier(h).squeeze(-1)  # [B]
