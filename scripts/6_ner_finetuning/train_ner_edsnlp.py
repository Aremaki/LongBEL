"""Train and evaluate EDS-NLP NER models on several datasets.

The script is designed for experiments such as:

    - MedMentions
    - QUAERO-EMEA
    - QUAERO-MEDLINE
    - SPACCC

It keeps a single reusable config file. The dataset-specific paths and labels are
centralized in DATASETS below, so you can change only `vars.dataset` and
`vars.base_model` from the command line.

Typical usage
-------------

python train_ner_edsnlp.py train configs/ner/train_ner.cfg \
  --vars.dataset quaero_emea \
  --vars.base_model ../../models/deberta-v3-large

python train_ner_edsnlp.py evaluate configs/ner/train_ner.cfg \
  --vars.dataset quaero_emea \
  --vars.base_model ../../models/deberta-v3-large
"""

from __future__ import annotations

import itertools
import json
import math
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Sized
from copy import deepcopy
from itertools import chain, repeat
from pathlib import Path
from typing import Any

import nltk
import polars as pl
import torch
from accelerate import Accelerator
from confit import Cli, validate_arguments
from confit.utils.random import set_seed
from datasets import load_dataset
from edsnlp.core.pipeline import Pipeline
from edsnlp.core.registries import registry
from edsnlp.pipes.trainable.embeddings.transformer.transformer import Transformer
from edsnlp.training.optimizer import LinearSchedule, ScheduledOptimizer
from edsnlp.utils.collections import batchify
from rich_logger import RichTablePrinter
from spacy.tokens import Doc, Span
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

try:
    import edsnlp
except ImportError:  # pragma: no cover
    edsnlp = None


if not Doc.has_extension("note_id"):
    Doc.set_extension("note_id", default=None)

BASE_DIR = Path.cwd()
app = Cli(pretty_exceptions_show_locals=False)


# -----------------------------------------------------------------------------
# Dataset registry
# -----------------------------------------------------------------------------
# Edit these paths and label lists once, then reuse the same config for all runs.
# The QUAERO labels below follow your previous EMEA config.

QUAERO_LABELS = [
    "Procedures",
    "Objects",
    "Phenomena",
    "Geographic Areas",
    "Devices",
    "Living Beings",
    "Physiology",
    "Disorders",
    "Chemicals & Drugs",
    "Anatomy",
]
MEDMENTIONS_ST21PV_LABELS = [
    "Living Beings",
    "Disorders",
    "Geographic Areas",
    "Physiology",
    "Objects",
    "Procedures",
    "Occupations",
    "Phenomena",
    "Chemicals & Drugs",
    "Organizations",
    "Anatomy",
    "Concepts & Ideas",
    "Genes & Molecular Sequences",
    "Devices",
]
SPACCC_LABELS = [
    "SINTOMA",
    "PROCEDIMIENTO",
    "ENFERMEDAD",
]

DATASETS: dict[str, dict[str, Any]] = {
    "EMEA": {
        "hf_dataset": "Aremaki/EMEA",
        "hf_config": "",
        "local_dir": "../../data/final_data/EMEA/bigbio_dataset/processed_data",
        "span_getter": {"gold_spans": QUAERO_LABELS},
    },
    "MEDLINE": {
        "hf_dataset": "Aremaki/MEDLINE",
        "hf_config": "",
        "local_dir": "../../data/final_data/MEDLINE/bigbio_dataset/processed_data",
        "span_getter": {"gold_spans": QUAERO_LABELS},
    },
    "MedMentions": {
        "hf_dataset": "Aremaki/MedMentions",
        "hf_config": "",
        "local_dir": "../../data/final_data/MedMentions/bigbio_dataset/processed_data",
        "span_getter": {"gold_spans": MEDMENTIONS_ST21PV_LABELS},
    },
    "SPACCC": {
        "hf_dataset": "Aremaki/SPACCC",
        "hf_config": "",
        "local_dir": "../../data/final_data/SPACCC/bigbio_dataset/processed_data",
        "span_getter": {"gold_spans": SPACCC_LABELS},
    },
}


def dataset_config(dataset: str) -> dict[str, Any]:
    """Return paths and span configuration for a dataset."""
    if dataset not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available datasets: "
            f"{', '.join(sorted(DATASETS))}"
        )
    return DATASETS[dataset]


def load_dataset_with_fallback(
    dataset: str,
    config_name: str,
    split: str,
    local_dir: Path,
):
    """
    Load a Hugging Face dataset split.

    Priority:
    1. Try loading from Hugging Face Hub.
    2. If this fails, load from a local HF dataset folder with load_dataset().
    3. As final fallback, try load_from_disk().
    """

    config_name = config_name or None  # type: ignore

    try:
        if config_name is None:
            print(f"Loading {dataset} [{split}] from Hugging Face...")
            return load_dataset(dataset, split=split)
        print(f"Loading {dataset} ({config_name}) [{split}] from Hugging Face...")
        return load_dataset(dataset, config_name, split=split)

    except Exception as e:
        print(f"Could not load from Hugging Face: {e}")
        print(f"Loading local dataset from {local_dir}")

    return load_dataset(str(local_dir), split=split)


def reconstruct_bigbio_text(page: dict[str, Any]) -> str:
    """
    Reconstruct the document text while preserving BigBio character offsets.
    """
    passages = sorted(
        page.get("passages", []),
        key=lambda p: p["offsets"][0][0],
    )

    text = ""
    for passage in passages:
        start = passage["offsets"][0][0]
        passage_text = passage["text"][0]

        if len(text) < start:
            text += " " * (start - len(text))

        text += passage_text

    return text


@registry.readers.register("bigbio_hf")
def read_bigbio_hf(
    dataset_name: str,
    split_name: str,
    randomize: bool = False,
    seed: int = 42,
):
    """
    Read a BigBio Hugging Face dataset and yield spaCy Doc objects.

    randomize:
        Shuffle documents. Use true for train, false for val/test.
    """

    cfg = dataset_config(dataset_name)

    dataset = cfg["hf_dataset"]
    config_name = cfg["hf_config"]
    local_dir = Path(cfg["local_dir"])
    span_setter = cfg["span_getter"]

    def generator(nlp) -> Iterable[Doc]:
        ds = load_dataset_with_fallback(
            dataset=dataset,
            config_name=config_name,
            split=split_name,
            local_dir=local_dir,
        )

        pages = list(ds)
        if randomize:
            rng = random.Random(seed)
            rng.shuffle(pages)

        for page_idx, page in enumerate(pages):
            doc = nlp.make_doc(reconstruct_bigbio_text(page))

            doc._.note_id = str(page.get("id", page_idx))

            all_spans = []

            for entity in page.get("entities", []):
                offsets = entity.get("offsets", [])
                if not offsets:
                    continue

                # Important: do not merge discontinuous entities.
                # A flat CRF/BIO model cannot represent them.
                if len(offsets) > 1:
                    continue

                start_char = min(start for start, _ in offsets)
                end_char = max(end for _, end in offsets)

                label = entity.get("type", "UNKNOWN")
                if isinstance(label, list):
                    label = label[0] if label else "UNKNOWN"

                span = doc.char_span(
                    start_char,
                    end_char,
                    label=label,
                    alignment_mode="strict",
                )

                if span is None:
                    continue

                accepted = False
                for _group_name, accepted_labels in span_setter.items():
                    if accepted_labels is True or label in accepted_labels:
                        accepted = True
                        break

                if accepted:
                    all_spans.append(span)

            doc.spans["gold_spans"] = all_spans

            spans_by_label = defaultdict(list)
            for span in all_spans:
                spans_by_label[span.label_].append(span)

            for label, label_spans in spans_by_label.items():
                doc.spans[label] = label_spans

            yield doc

    return generator


# -----------------------------------------------------------------------------
# Logging / metrics utilities
# -----------------------------------------------------------------------------

LOGGER_FIELDS = {
    "step": {},
    "(.*)loss": {
        "goal": "lower_is_better",
        "format": "{:.2e}",
        "goal_wait": 2,
    },
    "(.*)exact_ner/micro/(p|r|f)$": {
        "goal": "higher_is_better",
        "format": "{:.2%}",
        "goal_wait": 1,
        "name": r"\1ner_\2",
    },
    "lr": {"format": "{:.2e}"},
    "speed/(.*)": {"format": "{:.2f}", "name": r"\1"},
    "labels": {"format": "{:.2f}"},
}


def flatten_dict(d: Any, path: str = "") -> dict[str, Any]:
    if not isinstance(d, dict):
        return {path: d}
    return {
        k: v
        for key, val in d.items()
        for k, v in flatten_dict(val, f"{path}/{key}" if path else key).items()
    }


def get_metric(metrics: dict[str, Any], metric_name: str) -> float:
    flat = flatten_dict(metrics)
    if metric_name not in flat:
        available = "\n".join(f"- {k}" for k in sorted(flat))
        raise KeyError(
            f"Metric '{metric_name}' was not found. Available flattened metrics:\n"
            f"{available}"
        )
    value = flat[metric_name]
    if isinstance(value, torch.Tensor):
        value = value.item()
    return float(value)


def normalize_scores(scores: dict[str, Any]) -> dict[str, Any]:
    """
    Keep the old metric path:
    ner/exact_ner/micro/f

    even if the scorer itself returns:
    micro/f
    """
    if "ner" in scores:
        return scores
    return {"ner": {"exact_ner": scores}}


def score_docs(
    nlp: Pipeline,  # type: ignore
    docs: list[Doc],
    scorer: Any,
    pipe_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Score the model on gold docs.

    EDS-NLP scorers expect:
        scorer(gold_docs, predicted_docs)

    not:
        scorer(nlp, docs)
    """

    # Make prediction copies so we do not overwrite gold annotations
    pred_docs = deepcopy(docs)

    # Remove gold annotations from the prediction copies
    for doc in pred_docs:
        doc.ents = []
        doc.spans.clear()

    # Run only the requested pipe if provided
    if pipe_names is None:
        pred_docs = list(nlp.pipe(pred_docs))
    else:
        with nlp.select_pipes(enable=pipe_names):
            pred_docs = list(nlp.pipe(pred_docs))

    return normalize_scores(scorer(docs, pred_docs))


def clean_model_name(model_name: str) -> str:
    """Create a filesystem-friendly model tag from a HF name or local path."""
    model_name = str(model_name).rstrip("/")
    name = Path(model_name).name if "/" in model_name else model_name
    name = name.replace("microsoft/", "")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).replace("-", "_")


def compute_output_dir(output_root: Path | str, base_model: str, dataset: str) -> Path:
    return Path(output_root) / clean_model_name(base_model) / dataset


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# -----------------------------------------------------------------------------
# Batching utilities
# -----------------------------------------------------------------------------


class BatchSizeArg:
    """Confit/pydantic caster for values such as `2000 words` or `8 samples`."""

    @classmethod
    def validate(cls, value: Any, config: Any = None) -> tuple[int, str]:
        value = str(value)
        parts = value.split()
        num = int(parts[0])
        unit = parts[1] if len(parts) == 2 else "samples"
        if len(parts) in (1, 2) and unit in ("words", "samples", "spans"):
            return num, unit
        raise ValueError(
            f"Invalid batch size: {value}, must be '<int> samples|words|spans'"
        )

    @classmethod
    def __get_validators__(cls):
        yield cls.validate


class LengthSortedBatchSampler:
    """Batch samples of similar length together to reduce transformer padding waste."""

    def __init__(
        self,
        dataset: Sized,
        batch_size: int,
        batch_unit: str,
        noise: int = 1,
        drop_last: bool = False,
        buffer_size: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_unit = batch_unit
        self.noise = noise
        self.drop_last = drop_last
        self.buffer_size = buffer_size

    def __iter__(self):
        if self.batch_unit == "words":

            def sample_len(idx: int, noise: bool = True) -> int:
                count = sum(
                    len(x)
                    for x in next(
                        v
                        for k, v in self.dataset[idx].items()  # type: ignore
                        if k.endswith("word_lengths")
                    )
                )
                return (
                    count + random.randint(-self.noise, self.noise) if noise else count
                )

        elif self.batch_unit == "spans":

            def sample_len(idx: int, noise: bool = True) -> int:
                return len(
                    next(
                        v
                        for k, v in self.dataset[idx].items()  # type: ignore
                        if k.endswith("begins")
                    )
                )

        else:

            def sample_len(idx: int, noise: bool = True) -> int:
                return 1

        total_count = sum(sample_len(i, False) for i in range(len(self.dataset)))
        buffer_size = self.buffer_size or max(
            1, math.ceil(total_count / self.batch_size)
        )

        sorted_sequences = chain.from_iterable(
            sorted((sample_len(i), i) for i in range(len(self.dataset)))
            for _ in repeat(None)
        )

        def make_batches():
            total = 0
            batch: list[int] = []
            for seq_size, idx in sorted_sequences:
                if total and total + seq_size > self.batch_size:
                    if batch or not self.drop_last:
                        yield batch
                    total = 0
                    batch = []
                total += seq_size
                batch.append(idx)

        for buffer in batchify(make_batches(), buffer_size):
            random.shuffle(buffer)
            yield from buffer


class SubBatchCollater:
    """Split a batch into token-budgeted sub-batches for gradient accumulation."""

    def __init__(
        self,
        nlp: Pipeline,  # type: ignore
        embedding: Transformer,
        grad_accumulation_max_tokens: int,
    ) -> None:
        self.nlp = nlp
        self.embedding = embedding
        self.grad_accumulation_max_tokens = grad_accumulation_max_tokens

    def __call__(self, seq):
        total = 0
        mini_batches: list[list[Any]] = [[]]
        for sample_features in seq:
            num_tokens = sum(
                math.ceil(len(p) / self.embedding.stride) * self.embedding.window
                for key in sample_features
                if key.endswith("/input_ids")
                for p in sample_features[key]
            )
            if total and total + num_tokens > self.grad_accumulation_max_tokens:
                total = 0
                mini_batches.append([])
            total += num_tokens
            mini_batches[-1].append(sample_features)
        return [self.nlp.collate(b) for b in mini_batches if b]


@validate_arguments
class TrainingDataLoaderFactory:
    """Factory used by the config to build the training dataloader."""

    def __init__(
        self,
        data: list[Any],
        batch_size: BatchSizeArg,
        grad_accumulation_max_tokens: int,
        pipe_names: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        self.data = data
        self.seed = seed
        self.batch_size = batch_size
        self.grad_accumulation_max_tokens = grad_accumulation_max_tokens
        self.pipe_names = pipe_names

    def __call__(self, nlp: Pipeline) -> DataLoader:  # type: ignore
        with nlp.select_pipes(enable=self.pipe_names), set_seed(self.seed):
            trf_pipe = next(
                module
                for _pipe_name, pipe in nlp.torch_components()
                for _module_name, module in pipe.named_component_modules()
                if isinstance(module, Transformer)
            )
            train_docs = [d for reader in self.data for d in reader(nlp)]
            nlp.post_init(train_docs)
            preprocessed = list(nlp.preprocess_many(train_docs, supervision=True))
            print(
                "Training sample count"
                f" for {', '.join(self.pipe_names or [])}: {len(preprocessed)}"
            )
            return DataLoader(
                preprocessed,  # type: ignore
                batch_sampler=LengthSortedBatchSampler(
                    preprocessed,
                    batch_size=self.batch_size[0],  # type: ignore
                    batch_unit=self.batch_size[1],  # type: ignore
                ),
                collate_fn=SubBatchCollater(
                    nlp,
                    trf_pipe,
                    grad_accumulation_max_tokens=self.grad_accumulation_max_tokens,
                ),
            )


def connected_pipes(pipeline: Iterable[tuple[str, Any]]) -> list[list[str]]:
    """Group trainable pipes that share parameters.

    In this NER-only setup this will usually return a single phase, but keeping it
    makes the script robust to the EDS-NLP transformer/CNN/CRF structure.
    """
    pipe_to_params: dict[str, set[int]] = {}
    for name, pipe in pipeline:
        pipe_to_params[name] = {id(p) for p in pipe.parameters()}

    remaining = list(pipe_to_params)
    results: list[list[str]] = []
    while remaining:
        current = [remaining.pop(0)]
        i = 0
        while i < len(current):
            a = current[i]
            i += 1
            for j, b in enumerate(list(remaining)):
                if a != b and pipe_to_params[a] & pipe_to_params[b]:
                    current.append(b)
                    remaining[j] = None  # type: ignore
            remaining = [p for p in remaining if p is not None]
        results.append(current)
    return results


# -----------------------------------------------------------------------------
# Prediction export
# -----------------------------------------------------------------------------


def span_to_dict(span: Span) -> dict[str, Any]:
    return {
        "start": span.start_char,
        "end": span.end_char,
        "text": span.text,
        "label": span.label_,
    }


def span_to_bigbio_entity(span: Span, entity_id: int) -> dict[str, Any]:
    return {
        "id": f"T{entity_id}",
        "type": span.label_,
        "text": [span.text],
        "offsets": [[span.start_char, span.end_char]],
    }


def get_prediction_spans(doc: Doc, span_key: str = "gold_spans") -> list[Span]:
    """
    Get predicted spans from doc.spans.

    In your setup, predictions are written to doc.spans["gold_spans"]
    after prediction on cleaned documents.
    """
    return list(doc.spans.get(span_key, []))


def get_doc_id(doc: Doc, index: int) -> str:
    """
    Return the original document id.

    We require doc._.note_id because gold doc_id must align exactly
    with the original test dataframe.
    """
    if not Doc.has_extension("note_id"):
        raise ValueError("Doc extension note_id is not registered.")

    note_id = getattr(doc._, "note_id", None)

    if note_id is None:
        raise ValueError(
            f"Missing doc._.note_id for document at position {index}. "
            "Set doc._.note_id in the reader before evaluation."
        )

    return str(note_id)


def doc_to_bigbio_record(
    doc: Doc,
    spans: list[Span],
    index: int,
    doc_id: str | None = None,
) -> dict[str, Any]:
    """
    Convert one predicted doc to a BigBio-like record.
    """
    if doc_id is None:
        doc_id = get_doc_id(doc, index)

    doc_id = str(doc_id)

    return {
        "id": doc_id,
        "document_id": doc_id,
        "passages": [
            {
                "id": f"{doc_id}_passage_0",
                "type": "abstract",
                "text": [doc.text],
                "offsets": [[0, len(doc.text)]],
            }
        ],
        "entities": [
            span_to_bigbio_entity(span, entity_id=i) for i, span in enumerate(spans)
        ],
        "events": [],
        "coreferences": [],
        "relations": [],
    }


def write_predictions_jsonl_and_parquet(
    pred_docs: list[Doc],
    output: Path,
    span_key: str = "gold_spans",
    pipe_names: list[str] | None = None,
    gold_test_df: pl.DataFrame | None = None,
    match_on_label: bool = True,
    sent_tokenizer: Any | None = None,
) -> None:
    """
    Run NER prediction on clean copies of docs and export predictions.

    Exports:
        - predictions_test.jsonl: simple readable format
        - predictions_test.parquet: BigBio-like format
        - normalization_input_test.parquet: mixed dataframe for normalization/evaluation

    The mixed dataframe contains:
        - TP: predicted mention matched with gold
        - FP: predicted mention not in gold
        - FN: gold mention missed by NER

    Tracking columns:
        - is_predicted
        - is_gold
        - ner_status
        - pred_mention_id
        - gold_mention_id
    """

    output.mkdir(parents=True, exist_ok=True)

    jsonl_path = output / "predictions_test.jsonl"
    parquet_path = output / "predictions_test.parquet"
    norm_input_path = output / "normalization_input_test.parquet"

    # ------------------------------------------------------------------
    # Prepare gold dataframe
    # ------------------------------------------------------------------
    gold_info = None

    if gold_test_df is not None:
        gold = gold_test_df.clone()

        rename_map = {
            "doc_id": "filename",
            "semantic_group": "label",
            "mention": "span",
            "gold_concept_code": "code",
            "gold_concept_name": "annotation",
        }

        gold = gold.rename({
            old_name: new_name
            for old_name, new_name in rename_map.items()
            if old_name in gold.columns
        })

        for col in ["code", "semantic_rel", "annotation", "mention_id"]:
            if col not in gold.columns:
                gold = gold.with_columns(pl.lit("").alias(col))

        gold = gold.with_columns([
            pl.col("filename").cast(pl.Utf8),
            pl.col("label").cast(pl.Utf8),
            pl.col("start_span").cast(pl.Int64),
            pl.col("end_span").cast(pl.Int64),
            pl.col("span").fill_null("").cast(pl.Utf8),
            pl.col("code").fill_null("").cast(pl.Utf8),
            pl.col("semantic_rel").fill_null("").cast(pl.Utf8),
            pl.col("annotation").fill_null("").cast(pl.Utf8),
            pl.col("mention_id").fill_null("").cast(pl.Utf8),
        ])

        if match_on_label:
            merge_cols = ["filename", "label", "start_span", "end_span"]
        else:
            merge_cols = ["filename", "start_span", "end_span"]

        gold_info = (
            gold.select([
                "filename",
                "label",
                "start_span",
                "end_span",
                "mention_id",
                "span",
                "code",
                "semantic_rel",
                "annotation",
            ])
            .unique(subset=merge_cols, keep="first")
            .rename({
                "mention_id": "gold_mention_id",
                "span": "gold_span",
                "code": "gold_code",
                "semantic_rel": "gold_semantic_rel",
                "annotation": "gold_annotation",
            })
            .with_columns(pl.lit(True).alias("is_gold"))
        )

    # ------------------------------------------------------------------
    # Write JSONL, BigBio-like parquet, and collect prediction rows
    # ------------------------------------------------------------------
    bigbio_records = []
    pred_rows = []
    doc_ids = [get_doc_id(doc, i) for i, doc in enumerate(pred_docs)]

    with jsonl_path.open("w", encoding="utf8") as f:
        for i, doc in enumerate(pred_docs):
            # Important: use the document index to stay aligned with gold doc_id
            filename = doc_ids[i]

            spans = get_prediction_spans(doc, span_key=span_key)

            spans = sorted(
                spans,
                key=lambda s: (s.start_char, s.end_char, s.label_),
            )

            simple_item = {
                "id": filename,
                "text": doc.text,
                "entities": [span_to_dict(span) for span in spans],
            }

            f.write(json.dumps(simple_item, ensure_ascii=False) + "\n")

            bigbio_records.append(
                doc_to_bigbio_record(
                    doc=doc,
                    spans=spans,
                    index=i,
                    doc_id=filename,
                )
            )

            for entity_id, span in enumerate(spans, start=1):
                pred_rows.append({
                    "filename": filename,
                    "pred_mention_id": f"{filename}.{entity_id}",
                    "label": span.label_,
                    "start_span": int(span.start_char),
                    "end_span": int(span.end_char),
                    "span": span.text,
                    "sentence": make_marked_sentence_from_doc(
                        text=doc.text,
                        start=span.start_char,
                        end=span.end_char,
                        label=span.label_,
                        sent_tokenizer=sent_tokenizer,
                    ),
                    "is_predicted": True,
                })
    # ------------------------------------------------------------------
    # Save BigBio-like Parquet
    # ------------------------------------------------------------------
    if len(bigbio_records) == 0:
        bigbio_df = pl.DataFrame()
    else:
        bigbio_df = pl.DataFrame(bigbio_records)

    bigbio_df.write_parquet(parquet_path)

    # ------------------------------------------------------------------
    # Build prediction dataframe
    # ------------------------------------------------------------------
    pred_schema = {
        "filename": pl.Utf8,
        "pred_mention_id": pl.Utf8,
        "label": pl.Utf8,
        "start_span": pl.Int64,
        "end_span": pl.Int64,
        "span": pl.Utf8,
        "is_predicted": pl.Boolean,
        "sentence": pl.Utf8,
    }

    if len(pred_rows) == 0:
        pred_df = pl.DataFrame(schema=pred_schema)
    else:
        pred_df = pl.DataFrame(pred_rows).with_columns([
            pl.col("filename").cast(pl.Utf8),
            pl.col("pred_mention_id").cast(pl.Utf8),
            pl.col("label").cast(pl.Utf8),
            pl.col("start_span").cast(pl.Int64),
            pl.col("end_span").cast(pl.Int64),
            pl.col("span").cast(pl.Utf8),
            pl.col("is_predicted").cast(pl.Boolean),
            pl.col("sentence").cast(pl.Utf8),
        ])

    # ------------------------------------------------------------------
    # Full join with gold dataframe
    # ------------------------------------------------------------------
    if gold_info is not None:
        norm_df = pred_df.join(
            gold_info,
            on=merge_cols,  # type: ignore
            how="full",
            coalesce=True,
        )

        # If match_on_label=False, Polars may keep gold label as label_right.
        # We keep predicted label when available, otherwise gold label.
        if "label_right" in norm_df.columns:
            norm_df = norm_df.with_columns(
                pl.col("label").fill_null(pl.col("label_right")).alias("label")
            ).drop("label_right")

        norm_df = norm_df.with_columns([
            pl.col("is_predicted").fill_null(False),
            pl.col("is_gold").fill_null(False),
            pl.col("pred_mention_id").fill_null(""),
            pl.col("gold_mention_id").fill_null(""),
            pl.col("span").fill_null(pl.col("gold_span")).alias("span"),
            pl.col("sentence").fill_null("").alias("sentence"),
            pl.col("gold_code").fill_null("").alias("code"),
            pl.col("gold_semantic_rel").fill_null("").alias("semantic_rel"),
            pl.col("gold_annotation").fill_null("").alias("annotation"),
        ])

        norm_df = norm_df.with_columns([
            pl.when(pl.col("is_predicted") & pl.col("is_gold"))
            .then(pl.lit("TP"))
            .when(pl.col("is_predicted") & ~pl.col("is_gold"))
            .then(pl.lit("FP"))
            .otherwise(pl.lit("FN"))
            .alias("ner_status"),
            pl.when(pl.col("pred_mention_id") != "")
            .then(pl.col("pred_mention_id"))
            .otherwise(pl.col("gold_mention_id"))
            .alias("mention_id"),
        ])

        norm_columns = [
            "filename",
            "mention_id",
            "pred_mention_id",
            "gold_mention_id",
            "label",
            "start_span",
            "end_span",
            "span",
            "code",
            "semantic_rel",
            "annotation",
            "is_predicted",
            "is_gold",
            "ner_status",
        ]

        norm_df = norm_df.select(norm_columns)

    else:
        norm_df = pred_df.with_columns([
            pl.col("pred_mention_id").alias("mention_id"),
            pl.lit("").alias("gold_mention_id"),
            pl.lit("").alias("code"),
            pl.lit("").alias("semantic_rel"),
            pl.lit("").alias("annotation"),
            pl.lit(False).alias("is_gold"),
            pl.lit("PRED").alias("ner_status"),
        ]).select([
            "filename",
            "mention_id",
            "pred_mention_id",
            "gold_mention_id",
            "label",
            "start_span",
            "end_span",
            "span",
            "code",
            "semantic_rel",
            "annotation",
            "is_predicted",
            "is_gold",
            "ner_status",
        ])

    norm_df = norm_df.sort([
        "filename",
        "start_span",
        "end_span",
        "label",
        "ner_status",
    ])

    norm_df.write_parquet(norm_input_path)

    print(f"Saved JSONL predictions to {jsonl_path}")
    print(f"Saved BigBio-like Parquet predictions to {parquet_path}")
    print(f"Saved normalization/evaluation dataframe to {norm_input_path}")


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------


@app.command(name="train", registry=registry)
def train(
    *,
    nlp: Pipeline,  # type: ignore
    train_dataloader: list[TrainingDataLoaderFactory],  # type: ignore
    val_data: list[Any],
    test_data: list[Any],
    scorer: Any,
    dataset: str,
    base_model: str,
    output_root: Path,
    seed: int = 42,
    max_steps: int = 1000,
    transformer_lr: float = 1e-5,
    task_lr: float = 5e-5,
    validation_interval: int = 100,
    early_stopping_metric: str = "ner/exact_ner/micro/f",
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 0.0,
    max_grad_norm: float = 1.0,
    warmup_rate: float = 0.1,
    cpu: bool = False,
) -> Pipeline:  # type: ignore
    """Train one independent NER model for one dataset/model pair."""

    output_dir = compute_output_dir(output_root, base_model, dataset)
    model_last_path = output_dir / "model-last"
    model_best_path = output_dir / "model-best"
    train_metrics_path = output_dir / "train_metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    val_docs: list[Doc] = [d for reader in val_data for d in reader(nlp)]
    print(f"Dataset: {dataset}")
    print(f"Base model: {base_model}")
    print(f"Output dir: {output_dir}")
    print(f"Validation documents: {len(val_docs)}")

    trainable_pipe_names = {name for name, _pipe in nlp.torch_components()}
    print("Trainable components: " + ", ".join(sorted(trainable_pipe_names)))
    phases = connected_pipes(nlp.torch_components())
    print(
        "Training phases:"
        + "".join(f"\n - {i + 1}: {', '.join(names)}" for i, names in enumerate(phases))
    )

    all_metrics: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_step = -1
    bad_validations = 0
    stopped_early = False

    for phase_i, pipe_names in enumerate(phases):
        logger = RichTablePrinter(LOGGER_FIELDS, auto_refresh=False)
        with logger, nlp.select_pipes(disable=trainable_pipe_names - set(pipe_names)):
            print(f"Phase {phase_i + 1}: training {', '.join(pipe_names)}")
            set_seed(seed)

            trained_pipes = [nlp.get_pipe(name) for name in pipe_names]
            trf_pipe = next(
                (
                    module
                    for pipe in trained_pipes
                    for _name, module in pipe.named_component_modules()
                    if isinstance(module, Transformer)
                ),
                None,
            )

            print("Preprocessing training data")
            dataloaders = [
                dl(nlp)
                for dl in train_dataloader
                if not dl.pipe_names or set(dl.pipe_names) & set(pipe_names)
            ]

            params = {p for pipe in trained_pipes for p in pipe.parameters()}
            trf_params = params & set(trf_pipe.parameters() if trf_pipe else ())
            task_params = params - trf_params

            optimizer_groups = [
                {
                    "params": list(task_params),
                    "lr": task_lr,
                    "schedules": LinearSchedule(
                        total_steps=max_steps,
                        warmup_rate=warmup_rate,
                        start_value=task_lr,
                    ),
                }
            ]
            if transformer_lr and trf_params:
                optimizer_groups.append({
                    "params": list(trf_params),
                    "lr": transformer_lr,
                    "schedules": LinearSchedule(
                        total_steps=max_steps,
                        warmup_rate=warmup_rate,
                        start_value=0,
                    ),
                })

            optim = ScheduledOptimizer(torch.optim.AdamW(optimizer_groups))
            grad_params = {p for group in optim.param_groups for p in group["params"]}

            print(
                "Optimizing groups:"
                + "".join(
                    f"\n - {len(group['params'])} tensors "
                    f"({sum(p.numel() for p in group['params']):,} parameters), "
                    f"lr={group['lr']}"
                    for group in optim.param_groups
                )
            )
            print(f"Not optimizing {len(params - grad_params)} tensors")
            for param in params - grad_params:
                param.requires_grad_(False)

            accelerator = Accelerator(cpu=cpu)
            print("Device:", accelerator.device)
            prep = accelerator.prepare(optim, *dataloaders, *trained_pipes)
            optim = prep[0]
            dataloaders = prep[1 : 1 + len(dataloaders)]
            trained_pipes = prep[1 + len(dataloaders) :]

            cumulated_data = defaultdict(lambda: 0.0)
            iterators = [
                itertools.chain.from_iterable(itertools.repeat(dl))
                for dl in dataloaders
            ]

            nlp.train(True)
            set_seed(seed)

            for step in trange(
                max_steps + 1,
                desc=f"Training {dataset}",
                leave=True,
                mininterval=5.0,
            ):
                if step % validation_interval == 0:
                    nlp.train(False)
                    with torch.no_grad():
                        scores = score_docs(
                            nlp=nlp,
                            docs=val_docs,
                            scorer=scorer,
                        )
                    nlp.train(True)

                    metrics = {
                        "step": step,
                        "phase": phase_i + 1,
                        "dataset": dataset,
                        "base_model": base_model,
                        "lr": optim.param_groups[0]["lr"],
                        **dict(cumulated_data),
                        **scores,
                    }
                    all_metrics.append(metrics)
                    save_json(train_metrics_path, all_metrics)
                    logger.log_metrics(flatten_dict(metrics))
                    cumulated_data.clear()

                    current_score = get_metric(metrics, early_stopping_metric)
                    if current_score > best_score + early_stopping_min_delta:
                        best_score = current_score
                        best_step = step
                        bad_validations = 0
                        print(
                            f"New best {early_stopping_metric}: "
                            f"{best_score:.4f} at step {best_step}. Saving model-best."
                        )
                        nlp.to_disk(model_best_path)
                    else:
                        bad_validations += 1
                        print(
                            f"No improvement on {early_stopping_metric}. "
                            f"Bad validations: {bad_validations}/"
                            f"{early_stopping_patience}."
                        )

                    if bad_validations >= early_stopping_patience:
                        print(
                            f"Early stopping at step {step}. Best step: {best_step}, "
                            f"best {early_stopping_metric}: {best_score:.4f}."
                        )
                        stopped_early = True
                        break

                if step == max_steps:
                    break

                optim.zero_grad()
                for mini_batch in (b for iterator in iterators for b in next(iterator)):
                    loss = torch.zeros((), device=accelerator.device)
                    with nlp.cache():
                        for name, pipe in zip(pipe_names, trained_pipes, strict=False):
                            if name not in mini_batch:
                                continue
                            res = dict(pipe.module_forward(mini_batch[name]))
                            if "loss" in res:
                                loss = loss + res["loss"]
                                res[f"{name}_loss"] = res["loss"]
                            for key, value in res.items():
                                if key.endswith("loss"):
                                    cumulated_data[key] += float(value)
                            if torch.isnan(loss):
                                raise ValueError(f"NaN loss at component {name}")

                    accelerator.backward(loss)
                    del loss, mini_batch

                torch.nn.utils.clip_grad_norm_(grad_params, max_grad_norm)
                optim.step()

            if stopped_early:
                break

    print(f"Saving final model-last to {model_last_path}")
    nlp.to_disk(model_last_path)
    summary = {
        "dataset": dataset,
        "base_model": base_model,
        "output_dir": str(output_dir),
        "best_step": best_step,
        "best_metric": early_stopping_metric,
        "best_score": best_score,
        "stopped_early": stopped_early,
    }
    save_json(output_dir / "training_summary.json", summary)
    return nlp


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Trim leading/trailing whitespace while keeping offsets relative to text."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _break_inside_entity(
    point: int, entity_spans: list[tuple[int, int]] | None
) -> bool:
    if not entity_spans:
        return False
    return any(start < point < end for start, end in entity_spans)


def _process_chunk(
    chunk: str,
    offset: int,
    full_text: str,
    sent_spans: list[tuple[int, int]],
    sent_tokenizer=None,
) -> None:
    """
    Add sentence spans from a chunk.

    If sent_tokenizer has span_tokenize(), use it.
    Otherwise fallback to a simple punctuation-based splitter.
    """

    if not chunk.strip():
        return

    if sent_tokenizer is not None and hasattr(sent_tokenizer, "span_tokenize"):
        local_spans = list(sent_tokenizer.span_tokenize(chunk))
    else:
        local_spans = []
        start = 0
        for match in re.finditer(r"[.!?]+(?:\s+|$)", chunk):
            end = match.end()
            local_spans.append((start, end))
            start = end
        if start < len(chunk):
            local_spans.append((start, len(chunk)))

    for local_start, local_end in local_spans:
        global_start = offset + local_start
        global_end = offset + local_end

        global_start, global_end = _trim_span(
            text=full_text,
            start=global_start,
            end=global_end,
        )

        if global_start < global_end:
            sent_spans.append((global_start, global_end))


def span_tokenize_with_trailing_newlines(
    text: str,
    sent_tokenizer=None,
    entity_spans: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """
    Tokenize text into sentence spans.

    Splits on:
        - sentence punctuation if sent_tokenizer is provided
        - newlines
    Avoids splitting inside entity spans.
    """

    sent_spans: list[tuple[int, int]] = []
    last_break = 0

    newline_matches = list(re.finditer(r"\n+", text))

    for match in newline_matches:
        break_start = match.start()
        break_end = match.end()

        if _break_inside_entity(break_start, entity_spans):
            continue

        chunk = text[last_break:break_start]

        _process_chunk(
            chunk=chunk,
            offset=last_break,
            full_text=text,
            sent_spans=sent_spans,
            sent_tokenizer=sent_tokenizer,
        )

        # Important: skip the newline characters
        last_break = break_end

    if last_break < len(text):
        chunk = text[last_break:]

        _process_chunk(
            chunk=chunk,
            offset=last_break,
            full_text=text,
            sent_spans=sent_spans,
            sent_tokenizer=sent_tokenizer,
        )

    return sent_spans


def make_marked_sentence_from_doc(
    text: str,
    start: int,
    end: int,
    label: str,
    sent_tokenizer=None,
    start_entity: str = "[",
    end_entity: str = "]",
    start_group: str = "{",
    end_group: str = "}",
) -> str:
    """
    Create SynCABEL input:
        sentence with [mention]{semantic_group}
    """

    start = int(start)
    end = int(end)

    entity_spans = [(start, end)]

    sent_spans = span_tokenize_with_trailing_newlines(
        text=text,
        sent_tokenizer=sent_tokenizer,
        entity_spans=entity_spans,
    )

    # Find sentence containing the entity start
    sent_start, sent_end = 0, len(text)
    for s_start, s_end in sent_spans:
        if s_start <= start < s_end:
            sent_start, sent_end = s_start, s_end
            break

    sent_text = text[sent_start:sent_end]

    start_in_sent = start - sent_start
    end_in_sent = end - sent_start

    if (
        start_in_sent < 0
        or end_in_sent > len(sent_text)
        or start_in_sent >= end_in_sent
    ):
        mention = text[start:end]
        return f"{start_entity}{mention}{end_entity}{start_group}{label}{end_group}"

    marked_sentence = (
        sent_text[:start_in_sent]
        + start_entity
        + sent_text[start_in_sent:end_in_sent]
        + end_entity
        + start_group
        + str(label)
        + end_group
        + sent_text[end_in_sent:]
    )

    return " ".join(marked_sentence.split())


def predict_docs(
    nlp: Pipeline,  # type: ignore
    docs: list[Doc],
    pipe_names: list[str] | None = None,
    desc: str = "Running inference",
) -> list[Doc]:
    """
    Create clean prediction docs and run inference once.

    Important:
        - gold annotations are removed from prediction copies
        - doc._.note_id is preserved for alignment with gold dataframe
    """
    pred_docs = deepcopy(docs)

    for doc in pred_docs:
        doc.ents = []
        doc.spans.clear()

    if pipe_names is None:
        pred_docs = list(
            tqdm(
                nlp.pipe(pred_docs),
                total=len(pred_docs),
                desc=desc,
            )
        )
    else:
        with nlp.select_pipes(enable=pipe_names):
            pred_docs = list(
                tqdm(
                    nlp.pipe(pred_docs),
                    total=len(pred_docs),
                    desc=desc,
                )
            )

    return pred_docs


@app.command(name="evaluate", registry=registry)
def evaluate(
    *,
    scorer: Any,
    data: list[Any],
    dataset: str,
    base_model: str,
    output_root: Path,
    prediction_root: Path,
    model_name: str = "model-best",
) -> dict[str, Any]:
    """Evaluate a trained model on the test set and export JSON metrics/predictions."""

    if edsnlp is None:
        raise ImportError("Could not import edsnlp. Please install edsnlp first.")

    if dataset in ["EMEA", "MEDLINE"]:  # type: ignore
        sent_tokenizer = nltk.data.load("tokenizers/punkt/french.pickle")
    elif dataset == "MedMentions":  # type: ignore
        sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    elif dataset == "SPACCC":  # type: ignore
        sent_tokenizer = nltk.data.load("tokenizers/punkt/spanish.pickle")
    else:
        sent_tokenizer = None
    output_dir = compute_output_dir(output_root, base_model, dataset)
    model_path = output_dir / model_name
    metrics_path = output_dir / "test_metrics.json"
    prediction_dir = Path(prediction_root) / clean_model_name(base_model) / dataset

    print(f"Loading model from {model_path}")
    nlp = edsnlp.load(model_path)

    # ------------------------------------------------------------------
    # Move model to GPU for inference
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nlp.to(device)
    print(f"Inference device: {device}")

    test_docs: list[Doc] = [d for reader in data for d in reader(nlp)]
    print(f"Test documents: {len(test_docs)}")

    nlp.train(False)
    with torch.no_grad():
        pred_docs = predict_docs(
            nlp=nlp,
            docs=test_docs,
            pipe_names=None,
            desc=f"Predicting {dataset} test set",
        )
        scores = normalize_scores(scorer(test_docs, pred_docs))

    metrics = {
        "dataset": dataset,
        "base_model": base_model,
        "model_path": str(model_path),
        **scores,
    }
    save_json(metrics_path, metrics)
    print(f"Saved test metrics to {metrics_path}")

    gold_test_df = pl.read_csv(
        Path(f"../../data/final_data/{dataset}/test_tfidf_annotations_short.tsv"),
        separator="\t",
        has_header=True,
        schema_overrides={
            "gold_concept_code": str,
            "mention_id": str,
            "doc_id": str,  # force as string
        },  # type: ignore
    )

    write_predictions_jsonl_and_parquet(
        pred_docs=pred_docs,
        output=prediction_dir,
        span_key="gold_spans",
        pipe_names=None,
        gold_test_df=gold_test_df,
        match_on_label=True,
        sent_tokenizer=sent_tokenizer,
    )
    print(f"Saved predictions to {prediction_dir / 'predictions_test.jsonl'}")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


if __name__ == "__main__":
    app()
