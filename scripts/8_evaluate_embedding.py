import os
import pickle

import polars as pl
import torch

from syncabel.embeddings import TextEncoder

torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"


# Load Encoder (can be SapBERT local or CODER from hub)
# Example local SapBERT: model_name = "SapBERT-from-PubMedBERT-fulltext"
# Example CODER from hub: model_name = "GanjinZero/coder-large-v3"
model_name = "SapBERT-from-PubMedBERT-fulltext"
local_model_path = f"models/{model_name}"
encoder = TextEncoder(
    model_name=local_model_path if os.path.isdir(local_model_path) else model_name
)


# Create Embedding
def get_bert_embed(
    phrase_list,
    normalize=True,
    summary_method="CLS",
    tqdm_bar=True,
    batch_size=128,
):
    """Compatibility wrapper that forwards to TextEncoder.encode with CODER-like behavior."""
    embs = encoder.encode(
        list(phrase_list),
        batch_size=batch_size,
        normalize=normalize,
        summary_method=summary_method,
        tqdm_bar=tqdm_bar,
        max_length=32,
    )
    # Return a torch tensor on CPU to match previous behavior
    return torch.from_numpy(embs)


# UMLS Embeddings
legal_umls_token_all = pl.read_parquet(
    "../data/legal_umls_token_2017_short_syn_all_main.parquet"
)
Syn_to_annotation = legal_umls_token_all.with_columns(
    Syn=pl.col("Entity").str.split(" of type ").list.get(0)
)
Syn_to_CUI = {}
for category in Syn_to_annotation["SEM_NAME_MM"].unique():
    Syn_to_CUI[category] = dict(
        Syn_to_annotation.filter(pl.col("SEM_NAME_MM") == category)
        .group_by("Syn")
        .agg([pl.col("CUI").unique()])
        .iter_rows()
    )

# Process
umls_embedding = {}
for category in Syn_to_CUI.keys():
    print(category)
    cat_syn = list(Syn_to_CUI[category].keys())
    umls_embedding[category] = get_bert_embed(cat_syn)

out_dir = f"../data/UMLS_embeddings/{model_name}"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "medmentions_umls_2017_embeddings.pkl"), "wb") as file:
    pickle.dump(umls_embedding, file, protocol=-1)
with open(os.path.join(out_dir, "medmentions_umls_2017_syn_to_cui.pkl"), "wb") as file:
    pickle.dump(Syn_to_CUI, file, protocol=-1)
