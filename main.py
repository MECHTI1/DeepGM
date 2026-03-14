#!/usr/bin/env python3
"""
DeepM+ Predictor (UPDATED)
=========================
Goal:
  - pocket-level function prediction from a Zn-centered pocket graph:
      (1) EC top-level class
      (2) metal identity

Update in this version:
  ✅ Adds GHECOM pocket descriptors as a *global pocket vector* (ghecom_vec)
     and concatenates it into the final pocket embedding:
        pocket = f([atom_pool, res_pool, ghecom_vec])

Why this is the safest/cleanest integration:
  - does NOT break equivariance
  - does NOT require mapping grid->atom (optional later)
  - gives strong cavity/accessibility signals for EC classification

Expected additional Data field:
  .ghecom_vec  [B, ghecom_dim]  OR [ghecom_dim] for a single pocket
      - if [ghecom_dim], it will be broadcast to batch size B automatically.

Notes:
  - This script does NOT run GHECOM itself (since your local setup may differ).
    You should compute pocket descriptors externally and store them into Data.ghecom_vec.
  - The smoke test at the bottom generates random ghecom_vec.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from typing import Optional, Tuple, Dict, List


# =============================================================================
# 0.  SYMMETRIC ATOM DEFINITIONS & HELPERS
# =============================================================================
SYMMETRIC_GROUPS: List[Tuple[str, List[str]]] = [
    ("GLU", ["OE1", "OE2"]),
    ("ASP", ["OD1", "OD2"]),
    ("ARG", ["NH1", "NH2"]),
    ("LEU", ["CD1", "CD2"]),
    ("VAL", ["CG1", "CG2"]),
    ("PHE", ["CD1", "CD2"]),
    ("TYR", ["CD1", "CD2"]),
    ("ASN", ["OD1", "ND2"]),
    ("GLN", ["OE1", "NE2"]),
]
_SYM_LOOKUP: Dict[str, List[str]] = {r: atoms for r, atoms in SYMMETRIC_GROUPS}


def build_symmetry_map(
    residue_names: List[str],
    atom_names: List[str],
    residue_idx: List[int],
) -> Tuple[Tensor, int]:
    """
    Build a map: N original atoms → M merged nodes  (M ≤ N).

    Symmetric atoms within the same residue instance share one output node.
    All other atoms map to their own private node.

    Returns
    -------
    group_ids : LongTensor [N]   output node index for each atom
    n_groups  : int              total merged nodes M
    """
    N = len(atom_names)
    group_ids = list(range(N))
    cursor = N
    opened: Dict[Tuple[int, frozenset], int] = {}

    for i, (rname, aname, ridx) in enumerate(zip(residue_names, atom_names, residue_idx)):
        if rname not in _SYM_LOOKUP:
            continue
        sym_atoms = _SYM_LOOKUP[rname]
        if aname not in sym_atoms:
            continue
        key = (ridx, frozenset(sym_atoms))
        if key not in opened:
            opened[key] = cursor
            cursor += 1
        group_ids[i] = opened[key]

    unique_ids = sorted(set(group_ids))
    remap = {old: new for new, old in enumerate(unique_ids)}
    group_ids_t = torch.tensor([remap[g] for g in group_ids], dtype=torch.long)
    return group_ids_t, len(unique_ids)


def find_his_tautomer_pairs(
    residue_names: List[str],
    atom_names: List[str],
    residue_idx: List[int],
) -> Tensor:
    """
    Returns LongTensor [P, 2] — (ND1_idx, NE2_idx) for every HIS in pocket.
    Indices are in ORIGINAL atom space (before merging).
    """
    his_atoms: Dict[int, Dict[str, int]] = {}
    for i, (rname, aname, ridx) in enumerate(zip(residue_names, atom_names, residue_idx)):
        if rname != "HIS" or aname not in ("ND1", "NE2"):
            continue
        his_atoms.setdefault(ridx, {})[aname] = i

    pairs = [[d["ND1"], d["NE2"]] for d in his_atoms.values() if "ND1" in d and "NE2" in d]
    if not pairs:
        return torch.zeros(0, 2, dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long)


# =============================================================================
# 1.  SYMMETRIC ATOM POOLER
# =============================================================================
class SymmetricAtomPooler(nn.Module):
    """
    Merges symmetric atom embeddings (mean pool) into one node per group.
    Non-symmetric atoms pass through as solo nodes unchanged.

    Input  : atom_embed [N, D],  atom_pos [N, 3]
    Output : node_embed [M, D],  node_pos [M, 3]    M ≤ N
    """

    def forward(
        self,
        atom_embed: Tensor,   # [N, D]
        atom_pos: Tensor,     # [N, 3]
        group_ids: Tensor,    # [N]
        n_groups: int,
    ) -> Tuple[Tensor, Tensor]:
        D = atom_embed.size(-1)
        ids = group_ids.unsqueeze(-1)

        node_embed = torch.zeros(n_groups, D, device=atom_embed.device)
        node_embed.scatter_add_(0, ids.expand(-1, D), atom_embed)

        node_pos = torch.zeros(n_groups, 3, device=atom_pos.device)
        node_pos.scatter_add_(0, ids.expand(-1, 3), atom_pos)

        counts = torch.zeros(n_groups, device=atom_embed.device)
        counts.scatter_add_(0, group_ids, torch.ones(group_ids.size(0), device=atom_embed.device))
        counts = counts.clamp(min=1).unsqueeze(-1)

        return node_embed / counts, node_pos / counts


# =============================================================================
# 2.  HIS TAUTOMER CONSISTENCY LOSS
# =============================================================================
class HISTautomerConsistencyLoss(nn.Module):
    """
    HIS ND1 and NE2 are kept separate but embedding directions are pulled together
    using cosine similarity loss.
    """

    def __init__(self, weight: float = 0.3):
        super().__init__()
        self.weight = float(weight)

    def forward(self, atom_embed: Tensor, tautomer_pairs: Tensor) -> Tensor:
        if tautomer_pairs.size(0) == 0:
            return atom_embed.new_zeros(())
        nd1 = atom_embed[tautomer_pairs[:, 0]]
        ne2 = atom_embed[tautomer_pairs[:, 1]]
        cos_sim = F.cosine_similarity(nd1, ne2, dim=-1)
        return self.weight * (1.0 - cos_sim).mean()


# =============================================================================
# 3.  M-MATRIX LAYER
# =============================================================================
class MMatrixLayer(nn.Module):
    """
    Per-atom coordination geometry fingerprint (pairwise agg) + Zn anchor term.
    """

    def __init__(self, n_basis: int = 16, hidden: int = 64, out_dim: int = 128):
        super().__init__()
        self.rbf_centers = nn.Parameter(torch.linspace(0.5, 8.0, n_basis))
        self.rbf_widths = nn.Parameter(torch.ones(n_basis) * 0.5)

        self.sh_proj = nn.Linear(9, n_basis)
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * n_basis, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.zn_proj = nn.Sequential(
            nn.Linear(2 * n_basis, hidden),
            nn.SiLU(),
        )
        self.node_proj = nn.Sequential(
            nn.Linear(2 * hidden, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def rbf(self, d: Tensor) -> Tensor:
        return torch.exp(-self.rbf_widths.abs() * (d.unsqueeze(-1) - self.rbf_centers) ** 2)

    def sh(self, vec: Tensor) -> Tensor:
        x, y, z = vec[:, 0], vec[:, 1], vec[:, 2]
        return torch.stack(
            [
                torch.ones_like(x) / math.sqrt(4 * math.pi),
                math.sqrt(3 / (4 * math.pi)) * y,
                math.sqrt(3 / (4 * math.pi)) * z,
                math.sqrt(3 / (4 * math.pi)) * x,
                math.sqrt(15 / (4 * math.pi)) * x * y,
                math.sqrt(15 / (4 * math.pi)) * y * z,
                math.sqrt(5 / (16 * math.pi)) * (2 * z**2 - x**2 - y**2),
                math.sqrt(15 / (4 * math.pi)) * x * z,
                math.sqrt(15 / (16 * math.pi)) * (x**2 - y**2),
            ],
            dim=-1,
        )

    def forward(self, pos: Tensor, zinc_pos: Tensor, edge_index: Tensor) -> Tensor:
        src, dst = edge_index
        N = pos.size(0)

        r_ij = pos[dst] - pos[src]
        d_ij = r_ij.norm(dim=-1).clamp(min=1e-6)
        u_ij = r_ij / d_ij.unsqueeze(-1)

        pf = self.pair_mlp(torch.cat([self.rbf(d_ij), self.sh_proj(self.sh(u_ij))], -1))
        agg = torch.zeros(N, pf.size(-1), device=pos.device)
        agg.scatter_add_(0, src.unsqueeze(-1).expand_as(pf), pf)

        r_izn = pos - zinc_pos.unsqueeze(0)
        d_izn = r_izn.norm(dim=-1).clamp(min=1e-6)
        u_izn = r_izn / d_izn.unsqueeze(-1)
        zn_f = self.zn_proj(torch.cat([self.rbf(d_izn), self.sh_proj(self.sh(u_izn))], -1))

        return self.node_proj(torch.cat([agg, zn_f], dim=-1))


# =============================================================================
# 4.  ESM3 CROSS-ATTENTION FUSION
# =============================================================================
class ESM3CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusing residue embeddings into atom/node features.
    """

    def __init__(self, atom_dim=128, esm_dim=1280, out_dim=128, n_heads=8):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.d_head = out_dim // n_heads
        self.q_proj = nn.Linear(atom_dim, out_dim, bias=False)
        self.k_proj = nn.Linear(esm_dim, out_dim, bias=False)
        self.v_proj = nn.Linear(esm_dim, out_dim, bias=False)
        self.o_proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.gate = nn.Sequential(nn.Linear(atom_dim + out_dim, out_dim), nn.Sigmoid())

    def forward(self, atom_feat: Tensor, esm_embed: Tensor, atom_res_idx: Tensor) -> Tensor:
        # NOTE: atom_res_idx is currently unused in your fusion (you attend to full L).
        # You can later restrict K/V to a residue window per atom if you want.
        N, L, H, D = atom_feat.size(0), esm_embed.size(0), self.n_heads, self.d_head
        Q = self.q_proj(atom_feat).view(N, H, D)
        K = self.k_proj(esm_embed).view(L, H, D)
        V = self.v_proj(esm_embed).view(L, H, D)

        attn = torch.softmax(torch.einsum("nhd,lhd->nhl", Q, K) / math.sqrt(D), dim=-1)
        out = self.norm(self.o_proj(torch.einsum("nhl,lhd->nhd", attn, V).reshape(N, -1)))

        gate = self.gate(torch.cat([atom_feat, out], -1))
        if atom_feat.size(-1) != out.size(-1):
            return gate * out
        return gate * out + (1.0 - gate) * atom_feat


# =============================================================================
# 5.  PAINN
# =============================================================================
class PaiNNMessage(nn.Module):
    def __init__(self, hidden=128, n_rbf=20, cutoff=8.0):
        super().__init__()
        self.cutoff = cutoff
        self.rbf_centers = nn.Parameter(torch.linspace(0.5, cutoff, n_rbf))
        self.rbf_widths = nn.Parameter(torch.ones(n_rbf) * (cutoff / n_rbf))
        self.phi_s = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3 * hidden))
        self.W = nn.Sequential(nn.Linear(n_rbf, hidden), nn.SiLU(), nn.Linear(hidden, 3 * hidden))

    def envelope(self, d):
        x = d / self.cutoff
        return torch.where(x < 1.0, 0.5 * (torch.cos(math.pi * x) + 1.0), torch.zeros_like(x))

    def rbf(self, d):
        return torch.exp(-self.rbf_widths.abs() * (d.unsqueeze(-1) - self.rbf_centers) ** 2)

    def forward(self, s, v, pos, edge_index):
        src, dst = edge_index
        r = pos[dst] - pos[src]
        d = r.norm(dim=-1).clamp(min=1e-6)
        u = r / d.unsqueeze(-1)

        W = self.W(self.rbf(d) * self.envelope(d).unsqueeze(-1))
        sp = self.phi_s(s[src]) * W
        dS, dV1, dV2 = sp.chunk(3, dim=-1)

        H, N = s.size(-1), s.size(0)
        mv = dV1.unsqueeze(-1) * v[src] + dV2.unsqueeze(-1) * u.unsqueeze(-2)

        s_new = s + torch.zeros(N, H, device=s.device).scatter_add_(0, dst.unsqueeze(-1).expand(-1, H), dS)
        v_new = v + torch.zeros(N, H, 3, device=v.device).scatter_add_(
            0, dst.unsqueeze(-1).unsqueeze(-1).expand(-1, H, 3), mv
        )
        return s_new, v_new


class PaiNNUpdate(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.U = nn.Linear(hidden, hidden, bias=False)
        self.V = nn.Linear(hidden, hidden, bias=False)
        self.net = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3 * hidden))

    def forward(self, s, v):
        Uv = self.U(v.transpose(1, 2)).transpose(1, 2)
        Vv = self.V(v.transpose(1, 2)).transpose(1, 2)
        a = self.net(torch.cat([s, (Vv * Vv).sum(-1)], dim=-1))
        a_ss, a_sv, a_vv = a.chunk(3, dim=-1)
        return s + a_ss + a_sv * (Uv * Vv).sum(-1), v + a_vv.unsqueeze(-1) * Uv


class PaiNN(nn.Module):
    def __init__(self, hidden=128, n_layers=4, cutoff=8.0):
        super().__init__()
        self.layers = nn.ModuleList([PaiNNMessage(hidden, cutoff=cutoff) for _ in range(n_layers)])
        self.updates = nn.ModuleList([PaiNNUpdate(hidden) for _ in range(n_layers)])

    def forward(self, s, pos, edge_index):
        v = torch.zeros(s.size(0), s.size(1), 3, device=s.device)
        for msg, upd in zip(self.layers, self.updates):
            s, v = msg(s, v, pos, edge_index)
            s, v = upd(s, v)
        return s, v


# =============================================================================
# 6.  RESIDUE READOUT
# =============================================================================
class ResidueReadout(nn.Module):
    def __init__(self, atom_dim: int = 128, residue_dim: int = 128):
        super().__init__()
        self.attn_score = nn.Sequential(nn.Linear(atom_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.dist_bias = nn.Linear(1, 1, bias=False)
        self.proj = nn.Sequential(nn.Linear(atom_dim, residue_dim), nn.LayerNorm(residue_dim), nn.SiLU())

    def forward(
        self,
        atom_embed: Tensor,     # [M, H]
        atom_res_idx: Tensor,   # [M]
        atom_pos: Tensor,       # [M, 3]
        zinc_pos: Tensor,       # [3]
        n_residues: int,
    ) -> Tensor:
        dist_to_zn = (atom_pos - zinc_pos).norm(dim=-1, keepdim=True)  # [M,1]
        score = self.attn_score(atom_embed) + self.dist_bias(-dist_to_zn)

        # scatter-softmax within residue
        score_exp = torch.exp(score - score.max())
        denom = torch.zeros(n_residues, 1, device=atom_embed.device)
        denom.scatter_add_(0, atom_res_idx.unsqueeze(-1), score_exp)
        attn_w = score_exp / (denom[atom_res_idx] + 1e-9)

        H = atom_embed.size(-1)
        residue_embed = torch.zeros(n_residues, H, device=atom_embed.device)
        residue_embed.scatter_add_(0, atom_res_idx.unsqueeze(-1).expand(-1, H), attn_w * atom_embed)
        return self.proj(residue_embed)


# =============================================================================
# 7.  SUPERVISED CONTRASTIVE LOSS
# =============================================================================
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.T = float(temperature)

    def forward(self, z: Tensor, labels: Tensor) -> Tensor:
        z = F.normalize(z, dim=-1)
        sim = z @ z.T / self.T
        sim.fill_diagonal_(-1e9)

        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        pos_mask.fill_diagonal_(0)

        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        loss = -(pos_mask * log_prob).sum(1) / pos_mask.sum(1).clamp(min=1)
        return loss.mean()


# =============================================================================
# 8.  FULL MODEL (UPDATED: +GHECOM)
# =============================================================================
class ZincPocketPredictor(nn.Module):
    """
    End-to-end Zn-centered pocket predictor.

    Required Data fields (same as before):
      .x               [N_orig, atom_in_dim]
      .pos             [N_orig, 3]
      .edge_index      [2, E]
      .esm_embed       [L, esm_dim]
      .zinc_pos        [3]
      .group_ids       [N_orig]
      .n_groups        int
      .tautomer_pairs  [P, 2]
      .merged_atom_res [M]
      .n_residues      int
      .batch           [M]
      .res_batch       [R]

    NEW Data field:
      .ghecom_vec      [ghecom_dim] or [B, ghecom_dim]
        where B is the number of pockets in the batch.
    """

    def __init__(
        self,
        atom_in_dim: int = 64,
        esm_dim: int = 1280,
        hidden: int = 128,
        n_painn_layers: int = 4,
        n_ec: int = 7,
        n_metal: int = 8,
        cutoff: float = 8.0,
        contrastive_temp: float = 0.07,
        lambda_his: float = 0.3,
        ghecom_dim: int = 0,          # ✅ set >0 to enable GHECOM features
        ghecom_dropout: float = 0.0,  # optional regularization
    ):
        super().__init__()
        self.lambda_his = float(lambda_his)
        self.ghecom_dim = int(ghecom_dim)

        self.atom_embed = nn.Sequential(nn.Linear(atom_in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.m_matrix = MMatrixLayer(out_dim=hidden)
        self.m_merge = nn.Linear(2 * hidden, hidden)
        self.sym_pooler = SymmetricAtomPooler()
        self.esm_fusion = ESM3CrossAttentionFusion(hidden, esm_dim, hidden)
        self.painn = PaiNN(hidden, n_painn_layers, cutoff)
        self.vec_readout = nn.Linear(hidden, hidden)
        self.his_loss_fn = HISTautomerConsistencyLoss(weight=lambda_his)
        self.res_readout = ResidueReadout(hidden, hidden)

        self.pool_norm = nn.LayerNorm(hidden)
        self.res_norm = nn.LayerNorm(hidden)

        # ✅ NEW: ghecom projection (optional)
        if self.ghecom_dim > 0:
            self.ghecom_proj = nn.Sequential(
                nn.Linear(self.ghecom_dim, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
                nn.Dropout(ghecom_dropout),
            )
            final_in = 3 * hidden   # atom_pool + res_pool + ghecom_pool
        else:
            self.ghecom_proj = None
            final_in = 2 * hidden   # atom_pool + res_pool

        self.final_proj = nn.Sequential(nn.Linear(final_in, hidden), nn.SiLU(), nn.LayerNorm(hidden))

        self.ec_head = nn.Linear(hidden, n_ec)
        self.metal_head = nn.Linear(hidden, n_metal)
        self.ce_loss = nn.CrossEntropyLoss()
        self.sup_con = SupervisedContrastiveLoss(contrastive_temp)

    def _prepare_ghecom_pool(self, data: Data, B: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """
        Returns ghecom_pool: [B, hidden] if ghecom_dim > 0, else raises.
        Accepts:
          data.ghecom_vec shape [ghecom_dim] or [B, ghecom_dim].
        """
        ghe = getattr(data, "ghecom_vec", None)
        if ghe is None:
            raise ValueError("ghecom_dim > 0 but data.ghecom_vec is missing.")
        ghe = ghe.to(device=device, dtype=dtype)

        if ghe.dim() == 1:
            ghe = ghe.unsqueeze(0).expand(B, -1)  # broadcast single-pocket features
        elif ghe.dim() == 2:
            if ghe.size(0) != B:
                raise ValueError(f"data.ghecom_vec has B={ghe.size(0)} but batch has B={B}.")
        else:
            raise ValueError("data.ghecom_vec must be 1D [ghecom_dim] or 2D [B, ghecom_dim].")

        return self.ghecom_proj(ghe)  # [B, hidden]

    def forward(
        self,
        data: Data,
        labels_ec: Optional[Tensor] = None,
        labels_metal: Optional[Tensor] = None,
        lambda_con: float = 0.1,
    ) -> Dict[str, Tensor]:
        pos = data.pos
        edge_index = data.edge_index
        zinc_pos = data.zinc_pos
        tautomer_pairs = data.tautomer_pairs

        # 1) Atom embed
        s = self.atom_embed(data.x)

        # 2) M-matrix geometry
        s = self.m_merge(torch.cat([s, self.m_matrix(pos, zinc_pos, edge_index)], -1))

        # 3) Symmetry pool to merged node space
        s_m, pos_m = self.sym_pooler(s, pos, data.group_ids, data.n_groups)

        # remap edges to merged
        ei_m = data.group_ids[edge_index]
        ei_m = ei_m[:, ei_m[0] != ei_m[1]]
        ei_m = torch.unique(ei_m, dim=1)

        # 4) ESM fusion
        s_m = self.esm_fusion(s_m, data.esm_embed, data.merged_atom_res)

        # 5) PaiNN
        s_m, v_m = self.painn(s_m, pos_m, ei_m)
        s_m = s_m + self.vec_readout(v_m.norm(dim=-1))

        # 6) HIS tautomer loss (compute in original-atom space using group_ids)
        if tautomer_pairs.size(0) > 0:
            s_orig_space = s_m[data.group_ids]
            loss_his = self.his_loss_fn(s_orig_space, tautomer_pairs)
        else:
            loss_his = s_m.new_zeros(())

        # 7) residue embeddings
        residue_embed = self.res_readout(
            atom_embed=s_m,
            atom_res_idx=data.merged_atom_res,
            atom_pos=pos_m,
            zinc_pos=zinc_pos,
            n_residues=data.n_residues,
        )  # [R, H]

        # 8) hierarchical pooling
        batch = data.batch
        res_batch = data.res_batch
        B = int(batch.max().item()) + 1 if batch.numel() else 1

        atom_pool = torch.zeros(B, s_m.size(-1), device=s_m.device)
        atom_pool.scatter_add_(0, batch.unsqueeze(-1).expand_as(s_m), s_m)
        atom_pool = self.pool_norm(atom_pool / batch.bincount().float().unsqueeze(-1).clamp(min=1))

        res_pool = torch.zeros(B, residue_embed.size(-1), device=residue_embed.device)
        res_pool.scatter_add_(0, res_batch.unsqueeze(-1).expand_as(residue_embed), residue_embed)
        res_pool = self.res_norm(res_pool / res_batch.bincount().float().unsqueeze(-1).clamp(min=1))

        # ✅ NEW: add ghecom_pool if enabled
        if self.ghecom_dim > 0:
            ghecom_pool = self._prepare_ghecom_pool(
                data=data, B=B, device=s_m.device, dtype=s_m.dtype
            )  # [B, H]
            pocket = self.final_proj(torch.cat([atom_pool, res_pool, ghecom_pool], dim=-1))
        else:
            pocket = self.final_proj(torch.cat([atom_pool, res_pool], dim=-1))

        # 9) heads
        logits_ec = self.ec_head(pocket)
        logits_metal = self.metal_head(pocket)

        out = {
            "logits_ec": logits_ec,
            "logits_metal": logits_metal,
            "embed": pocket,
            "residue_embed": residue_embed,
        }

        # 10) losses
        if labels_ec is not None and labels_metal is not None:
            loss_ce = self.ce_loss(logits_ec, labels_ec) + self.ce_loss(logits_metal, labels_metal)
            loss_con = self.sup_con(pocket, labels_ec)
            out.update(
                {
                    "loss": loss_ce + lambda_con * loss_con + loss_his,
                    "loss_ce": loss_ce,
                    "loss_con": loss_con,
                    "loss_his": loss_his,
                }
            )

        return out


# =============================================================================
# 9.  PREDICT FUNCTION (UPDATED: carries ghecom_vec through Data)
# =============================================================================
EC_LABELS = {
    0: "Oxidoreductase",
    1: "Transferase",
    2: "Hydrolase",
    3: "Lyase",
    4: "Isomerase",
    5: "Ligase",
    6: "Translocase",
}
METAL_LABELS = {0: "Zn", 1: "Fe", 2: "Cu", 3: "Mn", 4: "Co", 5: "Ni", 6: "Ca", 7: "Mg"}


@torch.inference_mode()
def predict(model: ZincPocketPredictor, data: Data, device: str = "cpu") -> Dict:
    model.eval()
    model.to(device)
    data = data.to(device)

    if not hasattr(data, "batch") or data.batch is None:
        data.batch = torch.zeros(data.n_groups, dtype=torch.long, device=device)
    if not hasattr(data, "res_batch") or data.res_batch is None:
        data.res_batch = torch.zeros(data.n_residues, dtype=torch.long, device=device)

    out = model(data)
    ec_p = out["logits_ec"].softmax(-1)[0]
    metal_p = out["logits_metal"].softmax(-1)[0]

    return {
        "ec_class": EC_LABELS[int(ec_p.argmax())],
        "ec_probs": {EC_LABELS[i]: float(ec_p[i]) for i in range(len(EC_LABELS))},
        "metal_class": METAL_LABELS[int(metal_p.argmax())],
        "metal_probs": {METAL_LABELS[i]: float(metal_p[i]) for i in range(len(METAL_LABELS))},
        "embedding": out["embed"][0].cpu().numpy(),
        "residue_embed": out["residue_embed"].cpu().numpy(),
    }


# =============================================================================
# 10.  TRAINING LOOP
# =============================================================================
def train_epoch(model, loader, optimizer, device, lambda_con=0.1):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch, labels_ec=batch.y_ec, labels_metal=batch.y_metal, lambda_con=lambda_con)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += out["loss"].item()
    return total / max(1, len(loader))


# =============================================================================
# 11.  SMOKE TEST (UPDATED: includes ghecom_vec)
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    # Fake pocket: 1 GLU (OE1/OE2 merge), 1 HIS (ND1/NE2 separate), rest ALA
    N_ORIG, L, H_IN, ESM_DIM = 40, 12, 64, 1280
    res_names = (["GLU"] * 4 + ["HIS"] * 4 + ["ALA"] * (N_ORIG - 8))
    atm_names = (["OE1", "OE2", "CA", "CB"] + ["ND1", "NE2", "CG", "CB"] + ["CA"] * (N_ORIG - 8))
    res_idx = ([0] * 4 + [1] * 4 + list(range(2, N_ORIG - 6)))

    group_ids, n_groups = build_symmetry_map(res_names, atm_names, res_idx)
    tautomer_pairs = find_his_tautomer_pairs(res_names, atm_names, res_idx)

    merged_atom_res = torch.zeros(n_groups, dtype=torch.long)
    for i, g in enumerate(group_ids.tolist()):
        merged_atom_res[g] = res_idx[i]

    pos_orig = torch.randn(N_ORIG, 3)
    edge_index = radius_graph(pos_orig, r=8.0)
    zinc_pos = torch.zeros(3)

    # ✅ Example GHECOM feature vector:
    # Replace this with your real pocket descriptors (e.g., probe1.4 + probe1.0)
    # Suggested dims: 16–32 (start simple)
    GHECOM_DIM = 24
    ghecom_vec = torch.randn(GHECOM_DIM)


    data = Data(
        x=torch.randn(N_ORIG, H_IN),
        pos=pos_orig,
        edge_index=edge_index,
        esm_embed=torch.randn(L, ESM_DIM),
        zinc_pos=zinc_pos,
        group_ids=group_ids,
        n_groups=n_groups,
        tautomer_pairs=tautomer_pairs,
        merged_atom_res=merged_atom_res,
        n_residues=L,
        batch=torch.zeros(n_groups, dtype=torch.long),
        res_batch=torch.zeros(L, dtype=torch.long),
        ghecom_vec=ghecom_vec,  # ✅ NEW
    )

    model = ZincPocketPredictor(atom_in_dim=H_IN, esm_dim=ESM_DIM, ghecom_dim=GHECOM_DIM)
    result = predict(model, data)

    print("── Prediction ──────────────────────────────────────────────────")
    print(f"  EC class           : {result['ec_class']}")
    print(f"  Metal class        : {result['metal_class']}")
    print(f"  Pocket embedding   : {result['embedding'].shape}")
    print(f"  Residue embeddings : {result['residue_embed'].shape}  ← one per residue")
    print(f"  Merged nodes       : {n_groups}/{N_ORIG} original atoms")
    print(f"  HIS tautomer pairs : {tautomer_pairs.tolist()}")
    print("✓ Smoke-test passed (with GHECOM features)")
