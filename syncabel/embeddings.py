from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer


def _mean_pooling(token_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).type_as(token_embeddings)
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


@dataclass
class TextEncoder:
    model_name: str = "GanjinZero/coder-large-v3"
    device = None

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        embs: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)
            pooled = _mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])  # type: ignore
            # Normalize to unit vectors
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embs.append(pooled.detach().cpu().numpy())
        return np.vstack(embs) if embs else np.empty((0,))


def save_embeddings_parquet(
    texts: list[str],
    embeddings: np.ndarray,
    out_path: Path,
    extra_cols=None,
) -> None:
    data = {"text": texts, "embedding": embeddings.tolist()}
    if extra_cols:
        data.update(extra_cols)
    df = pl.DataFrame(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)


def load_embeddings_parquet(path: Path) -> dict[str, np.ndarray]:
    df = pl.read_parquet(path)
    # Expect columns: text, embedding (list[float])
    mapping: dict[str, np.ndarray] = {}
    for row in df.iter_rows(named=True):
        text = str(row["text"]).strip()
        vec = np.asarray(row["embedding"], dtype=np.float32)
        # Ensure normalized vectors
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        mapping[text.lower()] = vec
    return mapping


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # Assumes normalized inputs; if not normalized, will compute cosine anyway
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def best_by_cosine(
    mention: str,
    candidates: list[str],
    mention_embeds: dict[str, np.ndarray],
    syn_embeds: dict[str, np.ndarray],
    encoder=None,
):
    key = mention.lower().strip()
    m_vec = mention_embeds.get(key)
    if m_vec is None and encoder is not None:
        m_vec = encoder.encode([mention])
        if m_vec.size:
            m_vec = m_vec[0]
    if m_vec is None or (isinstance(m_vec, np.ndarray) and m_vec.size == 0):
        return None
    best_syn = None
    best_score = -1.0
    for cand in candidates:
        c_vec = syn_embeds.get(cand.lower().strip())
        if c_vec is None:
            continue
        score = float(np.dot(m_vec, c_vec))  # both normalized
        if score > best_score:
            best_score = score
            best_syn = cand
    return best_syn
