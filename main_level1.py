#!/usr/bin/env python3
"""
Stage-1 Minimal Zn-pocket classifier
===================================
What it does (in order):

1) Takes a pocket graph centered on Zn:
   - atom features: data.x         [N, atom_in_dim]
   - atom coords  : data.pos       [N, 3]
   - zinc coord   : data.zinc_pos  [3]
   - labels       : data.y_ec, data.y_metal (for training)

2) Adds simple geometric feature:
   - RBF(dist(atom, Zn))  -> appended to x

3) Pools atom features to one pocket vector:
   - mean over atoms (per graph in batch)

4) Predicts:
   - EC top-level class
   - metal identity

No PaiNN, no ESM, no symmetry merge, no GHECOM.
This is your "make sure training works" model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from typing import Optional, Dict


class ZnRBF(nn.Module):
    """RBF expansion of Zn distance: d -> [n_rbf]."""
    def __init__(self, n_rbf: int = 16, d_min: float = 0.0, d_max: float = 10.0):
        super().__init__()
        centers = torch.linspace(d_min, d_max, n_rbf)
        self.register_buffer("centers", centers)
        self.gamma = nn.Parameter(torch.ones(n_rbf) * (1.0 / ((d_max - d_min) / n_rbf + 1e-6) ** 2))

    def forward(self, d: Tensor) -> Tensor:
        # d: [N]
        return torch.exp(-self.gamma.abs() * (d.unsqueeze(-1) - self.centers) ** 2)


class PocketBaseline(nn.Module):
    """
    Minimal pocket classifier:
      x (+ Zn-RBF) -> MLP -> pooled pocket embedding -> dual heads
    """
    def __init__(self, atom_in_dim: int = 64, hidden: int = 128, n_ec: int = 7, n_metal: int = 8, n_rbf: int = 16):
        super().__init__()
        self.rbf = ZnRBF(n_rbf=n_rbf, d_min=0.0, d_max=10.0)

        self.atom_mlp = nn.Sequential(
            nn.Linear(atom_in_dim + n_rbf, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

        self.ec_head = nn.Linear(hidden, n_ec)
        self.metal_head = nn.Linear(hidden, n_metal)

        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        data: Data,
        labels_ec: Optional[Tensor] = None,
        labels_metal: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:

        x = data.x                      # [N, atom_in_dim]
        pos = data.pos                  # [N, 3]
        zinc_pos = data.zinc_pos        # [3] OR [B,3] (we handle both)
        batch = data.batch              # [N]

        # --- handle zinc_pos shape ---
        if zinc_pos.dim() == 1:
            # single pocket (or same Zn for all in batch): broadcast to atoms via batch
            zinc_atom = zinc_pos.unsqueeze(0).expand(pos.size(0), 3)  # [N,3]
        else:
            # [B,3] -> map each atom to its pocket Zn via batch
            zinc_atom = zinc_pos[batch]  # [N,3]

        # --- Zn distance features ---
        d = (pos - zinc_atom).norm(dim=-1)          # [N]
        d_rbf = self.rbf(d)                         # [N, n_rbf]
        x2 = torch.cat([x, d_rbf], dim=-1)          # [N, atom_in_dim+n_rbf]

        # --- per-atom transform ---
        h = self.atom_mlp(x2)                       # [N, hidden]

        # --- pool to pocket embedding ---
        pocket = global_mean_pool(h, batch)         # [B, hidden]

        logits_ec = self.ec_head(pocket)            # [B, n_ec]
        logits_metal = self.metal_head(pocket)      # [B, n_metal]

        out = {"logits_ec": logits_ec, "logits_metal": logits_metal, "embed": pocket}

        if labels_ec is not None and labels_metal is not None:
            loss = self.ce(logits_ec, labels_ec) + self.ce(logits_metal, labels_metal)
            out["loss"] = loss

        return out


def train_epoch(model, loader, optimizer, device="cpu"):
    model.train()
    tot = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch, labels_ec=batch.y_ec, labels_metal=batch.y_metal)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tot += out["loss"].item()
    return tot / max(1, len(loader))


@torch.inference_mode()
def predict(model, data: Data, device="cpu"):
    model.eval().to(device)
    data = data.to(device)
    if not hasattr(data, "batch") or data.batch is None:
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

    out = model(data)
    ec_p = out["logits_ec"].softmax(-1)[0]
    metal_p = out["logits_metal"].softmax(-1)[0]
    return ec_p, metal_p


if __name__ == "__main__":
    # Smoke test
    torch.manual_seed(0)
    N = 50
    atom_in = 64

    data = Data(
        x=torch.randn(N, atom_in),
        pos=torch.randn(N, 3),
        zinc_pos=torch.zeros(3),
        batch=torch.zeros(N, dtype=torch.long),
        y_ec=torch.tensor([2]),      # dummy
        y_metal=torch.tensor([0]),   # dummy
    )

    model = PocketBaseline(atom_in_dim=atom_in, hidden=128, n_ec=7, n_metal=8, n_rbf=16)
    ec_p, metal_p = predict(model, data)
    print("EC probs shape:", ec_p.shape)
    print("Metal probs shape:", metal_p.shape)
    print("OK")