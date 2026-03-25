import time
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANL"
device = "cuda" if torch.cuda.is_available() else "cpu"

t0 = time.time()
model = ESMC.from_pretrained("esmc_300m").to(device)
model.eval()
print(f"model load time: {time.time() - t0:.2f} s")

protein = ESMProtein(sequence=sequence)

with torch.no_grad():
    t1 = time.time()
    protein_tensor = model.encode(protein)
    print(f"encode time: {time.time() - t1:.2f} s")

    t2 = time.time()
    output = model.logits(
        protein_tensor,
        LogitsConfig(sequence=True, return_embeddings=True)
    )
    print(f"logits/embedding time: {time.time() - t2:.2f} s")

emb = output.embeddings

if emb.dim() == 3:
    emb = emb[0]

if emb.shape[0] == len(sequence) + 2:
    emb = emb[1:-1]

print("embedding shape:", emb.shape)

t3 = time.time()
torch.save(emb.cpu(), "residue_embeddings.pt")
print(f"save time: {time.time() - t3:.2f} s")
print("total time done")