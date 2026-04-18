from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class ResidueEdgeRecord:
    src: int
    dst: int
    dist_raw: Tensor
    seqsep: float
    same_chain: float
    vector_raw: Tensor
    interaction_type: Tensor
    source_type: Tensor
    geometry_label: str

    def clone(self) -> "ResidueEdgeRecord":
        return ResidueEdgeRecord(
            src=int(self.src),
            dst=int(self.dst),
            dist_raw=self.dist_raw.clone(),
            seqsep=float(self.seqsep),
            same_chain=float(self.same_chain),
            vector_raw=self.vector_raw.clone(),
            interaction_type=self.interaction_type.clone(),
            source_type=self.source_type.clone(),
            geometry_label=self.geometry_label,
        )

    def reversed_copy(self) -> "ResidueEdgeRecord":
        return ResidueEdgeRecord(
            src=int(self.dst),
            dst=int(self.src),
            dist_raw=self.dist_raw.clone(),
            seqsep=float(self.seqsep),
            same_chain=float(self.same_chain),
            vector_raw=(-self.vector_raw).clone(),
            interaction_type=self.interaction_type.clone(),
            source_type=self.source_type.clone(),
            geometry_label=self.geometry_label,
        )


@dataclass
class ResidueMetalEdgeRecord:
    residue_idx: int
    metal_idx: int
    dist_raw: Tensor
    vector_raw: Tensor
    interaction_type: Tensor
    source_type: Tensor
    geometry_label: str

    def clone(self) -> "ResidueMetalEdgeRecord":
        return ResidueMetalEdgeRecord(
            residue_idx=int(self.residue_idx),
            metal_idx=int(self.metal_idx),
            dist_raw=self.dist_raw.clone(),
            vector_raw=self.vector_raw.clone(),
            interaction_type=self.interaction_type.clone(),
            source_type=self.source_type.clone(),
            geometry_label=self.geometry_label,
        )
