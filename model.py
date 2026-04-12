from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import softmax

from data_structures import EDGE_SOURCE_TYPES, INTERACTION_SUMMARIES_OPTIONAL_WITH_RING
from label_schemes import N_EC_CLASSES, N_METAL_CLASSES


class RBFExpansion(nn.Module):
    def __init__(self, n_rbf: int = 16, d_min: float = 0.0, d_max: float = 12.0):
        super().__init__()
        centers = torch.linspace(d_min, d_max, n_rbf)
        self.register_buffer("centers", centers)
        width = (d_max - d_min) / n_rbf
        self.gamma = 1.0 / (width * width + 1e-8)

    def forward(self, d: Tensor) -> Tensor:
        return torch.exp(-self.gamma * (d.unsqueeze(-1) - self.centers) ** 2)


class TinyFeatureGroupMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class NodeScalarEncoder(nn.Module):
    def __init__(self, n_rbf: int = 16, out_dim: int = 128):
        super().__init__()
        self.dist_rbf = RBFExpansion(n_rbf=n_rbf, d_min=0.0, d_max=12.0)
        self.burial_encoder = TinyFeatureGroupMLP(in_dim=4, hidden_dim=8, out_dim=4)
        self.pka_encoder = TinyFeatureGroupMLP(in_dim=4, hidden_dim=8, out_dim=4)

        in_dim = 28 + 2 + 3 + 3 + 3 + 4 + 4 + 3 * n_rbf
        self.out_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        x_reschem: Tensor,
        x_role: Tensor,
        x_dist_raw: Tensor,
        x_misc: Tensor,
        x_env_burial: Tensor,
        x_env_pka: Tensor,
        x_env_conf: Tensor,
        x_env_interactions: Tensor,
    ) -> Tensor:
        d_rbf = self.dist_rbf(x_dist_raw).flatten(start_dim=1)
        burial_latent = self.burial_encoder(x_env_burial)
        pka_latent = self.pka_encoder(x_env_pka)
        x = torch.cat(
            [
                x_reschem,
                x_role,
                x_misc,
                burial_latent,
                pka_latent,
                x_env_conf,
                x_env_interactions,
                d_rbf,
            ],
            dim=-1,
        )
        return self.out_proj(x)


class AttentionPool(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or max(32, in_dim // 2)
        self.score = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        logits = self.score(x).squeeze(-1)
        weights = softmax(logits, batch)
        return global_add_pool(x * weights.unsqueeze(-1), batch)


class ESMGraphEncoder(nn.Module):
    def __init__(self, esm_dim: int, proj_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.attn_pool = AttentionPool(proj_dim)

    def forward(self, x_esm: Tensor, batch: Tensor) -> Tensor:
        z = self.esm_proj(x_esm)
        z_mean = global_mean_pool(z, batch)
        z_attn = self.attn_pool(z, batch)
        return torch.cat([z_mean, z_attn], dim=-1)


class EdgeScalarEncoder(nn.Module):
    def __init__(self, n_rbf: int = 16, out_dim: int = 64):
        super().__init__()
        self.dist_rbf = RBFExpansion(n_rbf=n_rbf, d_min=0.0, d_max=12.0)
        in_dim = 2 * n_rbf + 2 + len(INTERACTION_SUMMARIES_OPTIONAL_WITH_RING) + len(EDGE_SOURCE_TYPES)
        self.out_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        edge_dist_raw: Tensor,
        edge_seqsep: Tensor,
        edge_same_chain: Tensor,
        edge_interaction_type: Tensor,
        edge_source_type: Tensor,
    ) -> Tensor:
        d_rbf = self.dist_rbf(edge_dist_raw).flatten(start_dim=1)
        x = torch.cat(
            [d_rbf, edge_seqsep, edge_same_chain, edge_interaction_type, edge_source_type],
            dim=-1,
        )
        return self.out_proj(x)


def vector_norm(v: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.sqrt(torch.clamp((v * v).sum(dim=-1), min=eps))


class SimpleGVP(nn.Module):
    def __init__(self, s_in: int, v_in: int, s_out: int, v_out: int):
        super().__init__()
        self.scalar_mlp = nn.Sequential(
            nn.Linear(s_in + v_in, s_out),
            nn.SiLU(),
            nn.Linear(s_out, s_out),
        )
        self.vector_linear = nn.Linear(v_in, v_out, bias=False)
        self.vector_gate = nn.Linear(s_out, v_out)

    def forward(self, s: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        v_norm = vector_norm(v)
        s_cat = torch.cat([s, v_norm], dim=-1)
        s_out = self.scalar_mlp(s_cat)

        v_t = v.transpose(1, 2)
        v_proj = self.vector_linear(v_t).transpose(1, 2)
        gate = torch.sigmoid(self.vector_gate(s_out)).unsqueeze(-1)
        v_out = v_proj * gate
        return s_out, v_out


class SimpleGVPLayer(nn.Module):
    def __init__(self, s_dim: int, v_dim: int, e_dim: int):
        super().__init__()

        self.message_gvp = SimpleGVP(
            s_in=2 * s_dim + e_dim + 1,
            v_in=2 * v_dim + 1,
            s_out=s_dim,
            v_out=v_dim,
        )
        self.update_gvp = SimpleGVP(
            s_in=2 * s_dim,
            v_in=2 * v_dim,
            s_out=s_dim,
            v_out=v_dim,
        )
        self.norm_s = nn.LayerNorm(s_dim)

    def forward(self, s: Tensor, v: Tensor, edge_index: Tensor, edge_s: Tensor, edge_v: Tensor) -> Tuple[Tensor, Tensor]:
        src, dst = edge_index

        s_src = s[src]
        s_dst = s[dst]
        v_src = v[src]
        v_dst = v[dst]

        edge_len = vector_norm(edge_v)
        m_s_in = torch.cat([s_src, s_dst, edge_s, edge_len], dim=-1)
        m_v_in = torch.cat([v_src, v_dst, edge_v], dim=1)

        m_s, m_v = self.message_gvp(m_s_in, m_v_in)

        agg_s = torch.zeros_like(s)
        agg_s.index_add_(0, dst, m_s)

        agg_v = torch.zeros_like(v)
        agg_v.index_add_(0, dst, m_v)

        u_s_in = torch.cat([s, agg_s], dim=-1)
        u_v_in = torch.cat([v, agg_v], dim=1)
        ds, dv = self.update_gvp(u_s_in, u_v_in)

        s_out = self.norm_s(s + ds)
        v_out = v + dv
        return s_out, v_out


class GVPPocketClassifier(nn.Module):
    def __init__(
        self,
        esm_dim: int,
        hidden_s: int = 128,
        hidden_v: int = 16,
        edge_hidden: int = 64,
        n_layers: int = 4,
        n_metal: int = N_METAL_CLASSES,
        n_ec: int = N_EC_CLASSES,
        esm_fusion_dim: int = 128,
        metal_loss_weight: float = 1.0,
        ec_loss_weight: float = 1.0,
        metal_class_weights: Optional[Tensor] = None,
        ec_class_weights: Optional[Tensor] = None,
    ):
        super().__init__()
        # Current supervised targets:
        # - EC head: first EC digit only, mapped from EC 1..7 to class ids 0..6.
        # - Metal classifier: 3 classes -> Zn, Cu, and a merged Co/Fe/Ni class.

        self.node_scalar_encoder = NodeScalarEncoder(n_rbf=16, out_dim=hidden_s)
        self.esm_graph_encoder = ESMGraphEncoder(esm_dim=esm_dim, proj_dim=esm_fusion_dim, dropout=0.1)
        self.edge_scalar_encoder = EdgeScalarEncoder(n_rbf=16, out_dim=edge_hidden)
        self.gvp_attn_pool = AttentionPool(hidden_s)
        self.init_vec_proj = nn.Linear(2, hidden_v, bias=False)

        self.layers = nn.ModuleList(
            [SimpleGVPLayer(s_dim=hidden_s, v_dim=hidden_v, e_dim=edge_hidden) for _ in range(n_layers)]
        )

        gvp_graph_dim = 2 * hidden_s
        esm_graph_dim = 2 * esm_fusion_dim
        self.gvp_fusion_proj = nn.Sequential(
            nn.Linear(gvp_graph_dim, hidden_s),
            nn.LayerNorm(hidden_s),
            nn.SiLU(),
        )
        self.esm_fusion_proj = nn.Sequential(
            nn.Linear(esm_graph_dim, hidden_s),
            nn.LayerNorm(hidden_s),
            nn.SiLU(),
        )
        self.site_feature_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * hidden_s, hidden_s),
            nn.Sigmoid(),
        )
        fused_dim = 2 * hidden_s + 32

        self.head_metal = nn.Sequential(
            nn.Linear(fused_dim, hidden_s),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_s, n_metal),
        )
        self.head_ec = nn.Sequential(
            nn.Linear(fused_dim, hidden_s),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_s, n_ec),
        )

        self.metal_loss_weight = float(metal_loss_weight)
        self.ec_loss_weight = float(ec_loss_weight)
        self.register_buffer(
            "metal_class_weights",
            metal_class_weights.float() if metal_class_weights is not None else torch.empty(0),
        )
        self.register_buffer(
            "ec_class_weights",
            ec_class_weights.float() if ec_class_weights is not None else torch.empty(0),
        )

    def _init_vector_channels(self, x_vec: Tensor) -> Tensor:
        # x_vec arrives as two explicit geometric channels per residue; project them
        # into the hidden vector width used by the GVP layers.
        x_t = x_vec.transpose(1, 2)
        x_proj = self.init_vec_proj(x_t)
        return x_proj.transpose(1, 2)

    def _prepare_edge_vectors(self, data: Data) -> Tensor:
        if hasattr(data, "edge_vector_raw"):
            rel = data.edge_vector_raw.float()
        else:
            src, dst = data.edge_index
            rel = (data.pos[dst] - data.pos[src]).float()
        return rel.unsqueeze(1)

    def forward(self, data: Data) -> Dict[str, Tensor]:
        s = self.node_scalar_encoder(
            data.x_reschem,
            data.x_role,
            data.x_dist_raw,
            data.x_misc,
            data.x_env_burial,
            data.x_env_pka,
            data.x_env_conf,
            data.x_env_interactions,
        )
        v = self._init_vector_channels(data.x_vec)

        edge_s = self.edge_scalar_encoder(
            data.edge_dist_raw,
            data.edge_seqsep,
            data.edge_same_chain,
            data.edge_interaction_type,
            data.edge_source_type,
        )
        edge_v = self._prepare_edge_vectors(data)

        for layer in self.layers:
            s, v = layer(s, v, data.edge_index, edge_s, edge_v)

        # Structural branch: pool the residue-level GVP states into one graph embedding.
        pooled_mean = global_mean_pool(s, data.batch)
        pooled_attn = self.gvp_attn_pool(s, data.batch)
        gvp_graph_embed = torch.cat([pooled_mean, pooled_attn], dim=-1)

        # Sequence branch: pool residue ESM embeddings separately, then fuse late.
        esm_graph_embed = self.esm_graph_encoder(data.x_esm, data.batch)
        gvp_fused = self.gvp_fusion_proj(gvp_graph_embed)
        esm_fused = self.esm_fusion_proj(esm_graph_embed)
        if hasattr(data, "site_metal_stats"):
            site_stats = data.site_metal_stats.float()
        else:
            batch_size = int(data.batch.max().item()) + 1
            site_stats = torch.zeros(batch_size, 4, dtype=torch.float32, device=gvp_fused.device)
        site_fused = self.site_feature_encoder(site_stats)
        # The gate lets the model decide how much ESM information to inject per pocket.
        fusion_gate = self.fusion_gate(torch.cat([gvp_fused, esm_fused], dim=-1))
        pocket_embed = torch.cat([gvp_fused, fusion_gate * esm_fused, site_fused], dim=-1)

        logits_metal = self.head_metal(pocket_embed)
        logits_ec = self.head_ec(pocket_embed)

        outputs = {
            "logits_metal": logits_metal,
            "logits_ec": logits_ec,
            "embed": pocket_embed,
            "gvp_embed": gvp_graph_embed,
            "esm_embed": esm_graph_embed,
            "fusion_gate": fusion_gate,
        }

        if hasattr(data, "y_metal") and hasattr(data, "y_ec"):
            # TODO- HERE I need to add the final loss policy after seeing real-data results:
            # keep weighted CE as-is, or change task weighting / loss design based on baseline validation.
            metal_weights = self.metal_class_weights if self.metal_class_weights.numel() > 0 else None
            ec_weights = self.ec_class_weights if self.ec_class_weights.numel() > 0 else None
            loss = (
                self.metal_loss_weight * F.cross_entropy(logits_metal, data.y_metal, weight=metal_weights)
                + self.ec_loss_weight * F.cross_entropy(logits_ec, data.y_ec, weight=ec_weights)
            )
            outputs["loss"] = loss

        return outputs
