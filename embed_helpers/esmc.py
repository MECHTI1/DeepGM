import os
from pathlib import Path

import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

from Bio.PDB import PDBParser, MMCIFParser
from Bio.Data.PDBData import protein_letters_3to1


def extract_chain_sequences(structure_file):
    structure_file = Path(structure_file)

    if structure_file.suffix.lower() in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    elif structure_file.suffix.lower() == ".pdb":
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported file format: {structure_file.suffix}")

    structure = parser.get_structure("model", str(structure_file))
    sequences = {}

    first_model = next(structure.get_models())

    for chain in first_model:
        seq = []

        for residue in chain:
            if residue.id[0] != " ":
                continue

            resname = residue.get_resname().upper()
            aa = protein_letters_3to1.get(resname, "X")
            seq.append(aa)

        chain_seq = "".join(seq)
        if chain_seq:
            chain_id = chain.id.strip() if chain.id.strip() else "_"
            sequences[chain_id] = chain_seq

    if not sequences:
        raise ValueError(f"No protein sequences found in {structure_file}")

    return sequences


def clean_embedding_length(emb, sequence_length):
    if emb.dim() == 3:
        emb = emb[0]

    if emb.shape[0] == sequence_length + 2:
        emb = emb[1:-1]
    elif emb.shape[0] != sequence_length:
        raise ValueError(
            f"Unexpected embedding length: got {emb.shape[0]}, "
            f"expected {sequence_length} or {sequence_length + 2}"
        )

    return emb


def get_default_embeddings_dir() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    return project_root / ".data" / "embeddings"


def resolve_embeddings_dir(configured_dir: str | None) -> Path:
    if configured_dir:
        embeddings_dir = Path(configured_dir).expanduser()
        if not embeddings_dir.is_absolute():
            embeddings_dir = Path(__file__).resolve().parent.parent / embeddings_dir
    else:
        embeddings_dir = get_default_embeddings_dir()

    embeddings_dir.mkdir(parents=True, exist_ok=True)
    return embeddings_dir


def create_resi_embed_pt(structure_file, out_dir: Path | None = None):
    structure_file = Path(structure_file)
    chain_sequences = extract_chain_sequences(structure_file)

    if out_dir is None:
        out_dir = get_default_embeddings_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
    print(f"embeddings dir: {out_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model = ESMC.from_pretrained("esmc_300m").to(device)
    model.eval()

    with torch.no_grad():
        for chain_id, sequence in chain_sequences.items():
            print(f"\nProcessing chain {chain_id} | sequence length = {len(sequence)}")

            protein = ESMProtein(sequence=sequence)
            protein_tensor = model.encode(protein)

            output = model.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True)
            )

            emb = clean_embedding_length(output.embeddings, len(sequence))
            print("embedding shape:", emb.shape)

            out_file = out_dir / f"{structure_file.stem}_chain_{chain_id}_esmc.pt"
            torch.save(emb.cpu(), out_file)
            print(f"saved: {out_file}")


if __name__ == "__main__":
    structure_file_path = "/home/mechti/Downloads/AF-P06965-F1-model_v6.cif"
    configured_embeddings_dir = os.getenv("EMBEDDINGS_DIR")
    output_dir = resolve_embeddings_dir(configured_embeddings_dir)
    create_resi_embed_pt(structure_file_path, out_dir=output_dir)
