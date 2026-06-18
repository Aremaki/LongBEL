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
from itertools import chain, repeat
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from confit import Cli, validate_arguments
from confit.utils.random import set_seed
from edsnlp.core.pipeline import Pipeline
from edsnlp.core.registries import registry
from edsnlp.pipes.trainable.embeddings.transformer.transformer import Transformer
from edsnlp.training.optimizer import LinearSchedule, ScheduledOptimizer
from edsnlp.utils.collections import batchify
from rich_logger import RichTablePrinter
from spacy.tokens import Doc, Span
from torch.utils.data import DataLoader
from tqdm import trange

try:
    import edsnlp
except ImportError:  # pragma: no cover
    edsnlp = None


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
    "quaero_emea": {
        "hf_dataset": "Aremaki/EMEA",
        "hf_config": "",
        "train_split": "train",
        "val_split": "validation",
        "test_split": "test",
        "span_getter": {"gold_spans": QUAERO_LABELS},
    },
    "quaero_medline": {
        "hf_dataset": "Aremaki/MEDLINE",
        "hf_config": "",
        "train_split": "train",
        "val_split": "validation",
        "test_split": "test",
        "span_getter": {"gold_spans": QUAERO_LABELS},
    },
    "medmentions": {
        "hf_dataset": "Aremaki/MedMentions",
        "hf_config": "",
        "train_split": "train",
        "val_split": "validation",
        "test_split": "test",
        "span_getter": {"gold_spans": MEDMENTIONS_ST21PV_LABELS},
    },
    "spaccc": {
        "hf_dataset": "Aremaki/SPACCC",
        "hf_config": "",
        "train_split": "train",
        "val_split": "validation",
        "test_split": "test",
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


# This lets the TOML config contain:
#
# [dataset_config]
# @misc = "ner_dataset_config"
# dataset = ${vars.dataset}
try:
    registry.misc.register("ner_dataset_config", func=dataset_config)
except Exception:
    # Some registry implementations do not accept func=... and expect decorator use.
    @registry.misc.register("ner_dataset_config")
    def _registered_dataset_config(dataset: str) -> dict[str, Any]:  # type: ignore
        return dataset_config(dataset)


@registry.readers.register("bigbio_hf")
def read_bigbio_hf(
    dataset: str,
    config_name: str,
    split: str,
    span_setter: dict[str, Any],
):
    """
    Read BigBio HF dataset and yield spaCy Doc objects.
    """
    from datasets import load_dataset

    def generator(nlp) -> Iterable[Doc]:
        print(f"Loading {dataset} ({config_name}) split {split} via HuggingFace...")
        ds = load_dataset(dataset, config_name, split=split)

        for page in ds:
            # Reconstruct document text by sorting passages by offsets
            passages = sorted(
                page.get("passages", []),  # type: ignore
                key=lambda p: p["offsets"][0][0],
            )
            doc_text = ""
            for passage in passages:
                start_offset = passage["offsets"][0][0]
                if len(doc_text) < start_offset:
                    doc_text += " " * (start_offset - len(doc_text))
                doc_text += passage["text"][0]

            doc = nlp.make_doc(doc_text)

            spans_dict = (
                {k: [] for k in span_setter.keys()} if span_setter else {"ents": []}
            )

            for entity in page.get("entities", []):  # type: ignore
                if not entity.get("offsets"):
                    continue
                start_char = min(off[0] for off in entity["offsets"])
                end_char = max(off[1] for off in entity["offsets"])

                label = entity["type"]
                if isinstance(label, list):
                    label = label[0] if label else "UNKNOWN"

                span = doc.char_span(
                    start_char, end_char, label=label, alignment_mode="expand"
                )
                if span is not None:
                    if span_setter:
                        for k, v in span_setter.items():
                            if v is True or (isinstance(v, list) and label in v):
                                spans_dict[k].append(span)
                    else:
                        spans_dict["ents"].append(span)

            for k, spans in spans_dict.items():
                if k == "ents":
                    doc.ents = spans
                else:
                    doc.spans[k] = spans
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
        "text": span.text,
        "start": span.start_char,
        "end": span.end_char,
        "label": span.label_,
    }


def write_predictions_jsonl(nlp: Pipeline, docs: list[Doc], output: Path) -> None:  # type: ignore
    output.mkdir(parents=True, exist_ok=True)
    pred_path = output / "predictions_test.jsonl"
    with pred_path.open("w", encoding="utf8") as f:
        for doc in nlp.pipe(docs):
            item = {
                "text": doc.text,
                "entities": [span_to_dict(ent) for ent in doc.ents],
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


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
                        scores = scorer(nlp, val_docs)
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
    cpu: bool = False,
) -> dict[str, Any]:
    """Evaluate a trained model on the test set and export JSON metrics/predictions."""

    if edsnlp is None:
        raise ImportError("Could not import edsnlp. Please install edsnlp first.")

    output_dir = compute_output_dir(output_root, base_model, dataset)
    model_path = output_dir / model_name
    metrics_path = output_dir / "test_metrics.json"
    prediction_dir = Path(prediction_root) / clean_model_name(base_model) / dataset

    print(f"Loading model from {model_path}")
    nlp = edsnlp.load(model_path)
    if cpu:
        nlp.to("cpu")

    test_docs: list[Doc] = [d for reader in data for d in reader(nlp)]
    print(f"Test documents: {len(test_docs)}")

    nlp.train(False)
    with torch.no_grad():
        scores = scorer(nlp, test_docs)

    metrics = {
        "dataset": dataset,
        "base_model": base_model,
        "model_path": str(model_path),
        **scores,
    }
    save_json(metrics_path, metrics)
    print(f"Saved test metrics to {metrics_path}")

    write_predictions_jsonl(nlp, test_docs, prediction_dir)
    print(f"Saved predictions to {prediction_dir / 'predictions_test.jsonl'}")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


if __name__ == "__main__":
    app()
