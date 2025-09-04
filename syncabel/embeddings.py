from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tqdm
from transformers import AutoModel, AutoTokenizer


def _mean_pooling(token_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).type_as(token_embeddings)
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


@dataclass
class TextEncoder:
    model_name: str = "GanjinZero/coder-all"
    device = None

    def __post_init__(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
        except Exception as hub_err:  # pragma: no cover - network / availability branch
            local_dir = Path("models") / str(self.model_name)
            if local_dir.exists():
                print(
                    f"⚠️ Hub load failed for '{self.model_name}' ({hub_err}); falling back to local path {local_dir}."
                )
                self.tokenizer = AutoTokenizer.from_pretrained(str(local_dir))
                self.model = AutoModel.from_pretrained(str(local_dir))
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
        summary_method: str = "CLS",
        tqdm_bar: bool = True,
        max_length: int = 32,
    ) -> np.ndarray:
        """
        CODER-like encoder with batching and optional progress bar.

        - summary_method: "CLS" uses pooler_output or first token as fallback; "MEAN" uses masked mean pooling.
        - normalize: L2-normalize each embedding to unit length.
        - max_length: truncation length for tokenizer.
        """
        if not texts:
            return np.empty((0,), dtype=np.float32)

        self.model.eval()

        out_parts: list[np.ndarray] = []
        total = len(texts)
        pbar = tqdm.tqdm(total=total, disable=not tqdm_bar)
        try:
            for start in range(0, total, batch_size):
                batch = texts[start : start + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(**inputs)

                if summary_method.upper() == "CLS":
                    # Prefer pooler_output; fallback to first token if missing
                    embed = getattr(outputs, "pooler_output", None)
                    if embed is None or (
                        isinstance(embed, torch.Tensor) and embed.numel() == 0
                    ):
                        embed = outputs.last_hidden_state[:, 0, :]
                else:  # MEAN
                    embed = _mean_pooling(
                        outputs.last_hidden_state, inputs["attention_mask"]
                    )  # type: ignore

                if normalize:
                    embed = torch.nn.functional.normalize(embed, p=2, dim=1)

                out_parts.append(embed.detach().cpu().numpy())
                pbar.update(len(batch))
        finally:
            pbar.close()

        return np.vstack(out_parts) if out_parts else np.empty((0,), dtype=np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # Assumes normalized inputs; if not normalized, will compute cosine anyway
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def best_by_cosine(
    encoder: TextEncoder,
    mention: str,
    candidates: list[str],
):
    m_vec = encoder.encode([mention])
    if m_vec.size:
        m_vec = m_vec[0]
    else:
        return None
    best_syn = None
    best_score = -1.0
    for cand in candidates:
        c_vec = encoder.encode([cand])
        if c_vec.size:
            c_vec = c_vec[0]
        else:
            continue
        score = cosine_sim(m_vec, c_vec)  # both normalized
        if score > best_score:
            best_score = score
            best_syn = cand
    return best_syn, best_score
