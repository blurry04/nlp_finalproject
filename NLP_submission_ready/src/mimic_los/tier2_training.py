from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from .constants import LABELS
from .metrics import compute_classification_metrics


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _read_metadata(prepared_dir: Path) -> dict[str, Any]:
    return json.loads((prepared_dir / "metadata.json").read_text(encoding="utf-8"))


def _build_offsets(path: Path) -> list[int]:
    offsets: list[int] = []
    offset = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                offsets.append(offset)
            offset += len(line)
    return offsets


class LazyPreparedJsonlDataset(Dataset):
    def __init__(
        self,
        path: Path,
        text_field: str,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        self.path = path
        self.text_field = text_field
        offsets = _build_offsets(path)
        if max_samples is not None and max_samples < len(offsets):
            rng = np.random.default_rng(seed)
            chosen = np.sort(rng.choice(len(offsets), size=max_samples, replace=False))
            offsets = [offsets[idx] for idx in chosen]
        self.offsets = offsets
        self._handle = None

    def __len__(self) -> int:
        return len(self.offsets)

    def _get_handle(self):
        if self._handle is None:
            self._handle = self.path.open("r", encoding="utf-8", errors="replace")
        return self._handle

    def __getitem__(self, index: int) -> dict[str, Any]:
        handle = self._get_handle()
        handle.seek(self.offsets[index])
        record = json.loads(handle.readline())
        return {
            "text": record.get(self.text_field, ""),
            "structured_vector": record.get("structured_vector", []),
            "label_id": int(record["label_id"]),
            "subject_id": record.get("subject_id", ""),
            "hadm_id": record.get("hadm_id", ""),
            "stay_id": record.get("stay_id", ""),
        }


class HeadTailCollator:
    def __init__(
        self,
        tokenizer: Any,
        max_length: int,
        with_structured: bool,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_structured = with_structured

    def _encode_single(self, text: str) -> dict[str, list[int]]:
        token_ids = self.tokenizer.encode(text or "", add_special_tokens=False)
        inner_max = self.max_length - self.tokenizer.num_special_tokens_to_add(pair=False)
        if len(token_ids) > inner_max:
            front = inner_max // 2
            back = inner_max - front
            token_ids = token_ids[:front] + token_ids[-back:]
        prefix_ids: list[int] = []
        suffix_ids: list[int] = []
        cls_token_id = getattr(self.tokenizer, "cls_token_id", None)
        sep_token_id = getattr(self.tokenizer, "sep_token_id", None)
        if cls_token_id is not None:
            prefix_ids.append(int(cls_token_id))
        if sep_token_id is not None:
            suffix_ids.append(int(sep_token_id))
        input_ids = prefix_ids + token_ids + suffix_ids
        input_ids = input_ids[: self.max_length]
        attention_mask = [1] * len(input_ids)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + ([pad_id] * pad_len)
            attention_mask = attention_mask + ([0] * pad_len)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encodings = [self._encode_single(item["text"]) for item in batch]
        result = {
            "input_ids": torch.tensor([enc["input_ids"] for enc in encodings], dtype=torch.long),
            "attention_mask": torch.tensor([enc["attention_mask"] for enc in encodings], dtype=torch.long),
            "labels": torch.tensor([item["label_id"] for item in batch], dtype=torch.long),
            "subject_ids": [str(item.get("subject_id", "")) for item in batch],
            "hadm_ids": [str(item.get("hadm_id", "")) for item in batch],
            "stay_ids": [str(item.get("stay_id", "")) for item in batch],
        }
        if self.with_structured:
            result["structured_features"] = torch.tensor(
                [item["structured_vector"] for item in batch],
                dtype=torch.float32,
            )
        return result


def masked_mean(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def freeze_transformer_layers(model: AutoModel, freeze_layers: int) -> None:
    encoder = getattr(model, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None:
        return
    for layer in list(layers)[:freeze_layers]:
        for param in layer.parameters():
            param.requires_grad = False


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_factor = (1.0 - target_probs).pow(self.gamma)
        loss = -focal_factor * target_log_probs
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            loss = loss * alpha[targets]
        return loss.mean()


class LateFusionClinicalBert(nn.Module):
    def __init__(
        self,
        model_name: str,
        structured_dim: int,
        num_labels: int = 3,
        dropout: float = 0.2,
        freeze_layers: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        freeze_transformer_layers(self.encoder, freeze_layers)
        hidden_size = self.encoder.config.hidden_size
        self.struct_proj = nn.Sequential(
            nn.LayerNorm(structured_dim),
            nn.Linear(structured_dim, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2 + hidden_size // 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        structured_features: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]
        mean_token = masked_mean(outputs.last_hidden_state, attention_mask)
        structured = self.struct_proj(structured_features)
        logits = self.classifier(torch.cat([cls_token, mean_token, structured], dim=-1))
        return logits


class NarrativeClinicalBert(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int = 3,
        dropout: float = 0.2,
        freeze_layers: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        freeze_transformer_layers(self.encoder, freeze_layers)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]
        mean_token = masked_mean(outputs.last_hidden_state, attention_mask)
        return self.classifier(torch.cat([cls_token, mean_token], dim=-1))


@dataclass
class Tier2Config:
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    dropout: float
    freeze_layers: int
    focal_gamma: float
    warmup_ratio: float = 0.10


def _class_weights_from_split(path: Path) -> torch.Tensor:
    labels: list[int] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                labels.append(int(json.loads(line)["label_id"]))
    classes = np.arange(len(LABELS))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.asarray(labels))
    return torch.tensor(weights, dtype=torch.float32)


def _make_loaders(
    split_paths: dict[str, Path],
    tokenizer: Any,
    text_field: str,
    with_structured: bool,
    batch_size: int,
    max_length: int,
    seed: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = LazyPreparedJsonlDataset(split_paths["train"], text_field=text_field, max_samples=max_train_samples, seed=seed)
    val_dataset = LazyPreparedJsonlDataset(split_paths["val"], text_field=text_field, max_samples=max_val_samples, seed=seed)
    test_dataset = LazyPreparedJsonlDataset(split_paths["test"], text_field=text_field, max_samples=max_val_samples, seed=seed)
    collator = HeadTailCollator(tokenizer=tokenizer, max_length=max_length, with_structured=with_structured)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collator)
    return train_loader, val_loader, test_loader


def _dataset_size(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _run_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    collect_predictions: bool = False,
    tier_name: str = "",
    split_name: str = "",
    progress_label: str | None = None,
    log_every: int = 500,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    is_train = optimizer is not None
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda" and is_train)
    if is_train:
        model.train()
    else:
        model.eval()
    running_loss = 0.0
    probs_list: list[np.ndarray] = []
    preds_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    prediction_records: list[dict[str, Any]] = []

    total_batches = len(loader)
    if progress_label:
        _log(f"{progress_label}: starting {total_batches:,} batches")
    for batch_idx, batch in enumerate(loader, start=1):
        labels = batch.pop("labels").to(device)
        subject_ids = batch.pop("subject_ids")
        hadm_ids = batch.pop("hadm_ids")
        stay_ids = batch.pop("stay_ids")
        batch = {key: value.to(device) for key, value in batch.items()}
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(**batch)
                loss = loss_fn(logits, labels)
            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
        probabilities = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
        predictions = probabilities.argmax(axis=1)
        labels_np = labels.detach().cpu().numpy()
        probs_list.append(probabilities)
        preds_list.append(predictions)
        labels_list.append(labels_np)
        if collect_predictions:
            for idx, predicted_id in enumerate(predictions):
                actual_id = int(labels_np[idx])
                prediction_records.append(
                    {
                        "subject_id": subject_ids[idx],
                        "hadm_id": hadm_ids[idx],
                        "stay_id": stay_ids[idx],
                        "tier": tier_name,
                        "split": split_name,
                        "actual": LABELS[actual_id],
                        "predicted": LABELS[int(predicted_id)],
                        "correct": int(predicted_id) == actual_id,
                        "confidence_raw": float(probabilities[idx, int(predicted_id)]),
                        "probabilities": {label: float(probabilities[idx, label_idx]) for label_idx, label in enumerate(LABELS)},
                        "shap_top_features": [],
                    }
                )
        running_loss += float(loss.item()) * len(predictions)
        if progress_label and (
            batch_idx == 1
            or batch_idx == total_batches
            or (log_every > 0 and batch_idx % log_every == 0)
        ):
            _log(f"{progress_label}: batch {batch_idx:,}/{total_batches:,}")

    y_true = np.concatenate(labels_list)
    y_pred = np.concatenate(preds_list)
    y_proba = np.concatenate(probs_list)
    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    mean_loss = running_loss / max(len(y_true), 1)
    return mean_loss, metrics, prediction_records


def _train_single_trial(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    config: Tier2Config,
    device: torch.device,
    trial_label: str,
) -> tuple[nn.Module, dict[str, Any]]:
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn = FocalLoss(alpha=class_weights, gamma=config.focal_gamma)

    best_state = None
    best_metrics = None
    best_score = -1.0
    patience = 0
    _log(f"{trial_label}: training started with config={json.dumps(asdict(config), sort_keys=True)}")
    for epoch in range(config.epochs):
        epoch_label = f"{trial_label} epoch {epoch + 1}/{config.epochs}"
        train_loss, train_metrics, _ = _run_loader(
            model,
            train_loader,
            device,
            loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            progress_label=f"{epoch_label} train",
            log_every=500,
        )
        val_loss, val_metrics, _ = _run_loader(
            model,
            val_loader,
            device,
            loss_fn,
            optimizer=None,
            scheduler=None,
            progress_label=f"{epoch_label} val",
            log_every=500,
        )
        score = val_metrics["macro_f1"]
        _log(
            f"{epoch_label}: train_loss={train_loss:.4f} "
            f"train_macro_f1={train_metrics['macro_f1']:.4f} "
            f"val_loss={val_loss:.4f} val_macro_f1={score:.4f} "
            f"val_weighted_f1={val_metrics['weighted_f1']:.4f}"
        )
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_metrics": train_metrics,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
            }
            patience = 0
            _log(f"{epoch_label}: new best validation macro_f1={best_score:.4f}")
        else:
            patience += 1
            if patience >= 2:
                _log(f"{trial_label}: early stopping after epoch {epoch + 1}; best_macro_f1={best_score:.4f}")
                break

    assert best_state is not None and best_metrics is not None
    model.load_state_dict(best_state)
    _log(f"{trial_label}: finished; best_epoch={best_metrics['epoch']} best_val_macro_f1={best_score:.4f}")
    return model, best_metrics


def _write_prediction_file(path: Path, records: list[dict[str, Any]]) -> str:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(path)


def _search_configs(
    tier_name: str,
    builder: Any,
    split_paths: dict[str, Path],
    structured_dim: int | None,
    output_dir: Path,
    model_name: str,
    text_field: str,
    with_structured: bool,
    max_length: int,
    seed: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
    configs: list[Tier2Config],
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = _class_weights_from_split(split_paths["train"]).to(device)
    leaderboard: list[dict[str, Any]] = []
    best_bundle = None
    best_score = -1.0
    _log(
        f"{tier_name}: search started on device={device} text_field={text_field} "
        f"with_structured={with_structured} max_length={max_length} configs={len(configs)}"
    )

    for idx, config in enumerate(configs, start=1):
        trial_label = f"{tier_name} config {idx}/{len(configs)} seed={seed}"
        _log(f"{trial_label}: preparing loaders")
        train_loader, val_loader, test_loader = _make_loaders(
            split_paths=split_paths,
            tokenizer=tokenizer,
            text_field=text_field,
            with_structured=with_structured,
            batch_size=config.batch_size,
            max_length=max_length,
            seed=seed + idx,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
        )
        _log(
            f"{trial_label}: loader sizes train={len(train_loader.dataset):,} "
            f"val={len(val_loader.dataset):,} test={len(test_loader.dataset):,} "
            f"batch_size={config.batch_size}"
        )
        model = builder(config, structured_dim).to(device)
        model, best_metrics = _train_single_trial(
            model,
            train_loader,
            val_loader,
            class_weights,
            config,
            device,
            trial_label=trial_label,
        )
        loss_fn = FocalLoss(alpha=class_weights, gamma=config.focal_gamma)
        _, test_metrics, _ = _run_loader(
            model,
            test_loader,
            device,
            loss_fn,
            optimizer=None,
            scheduler=None,
            progress_label=f"{trial_label} test",
            log_every=500,
        )
        _log(
            f"{trial_label}: test_macro_f1={test_metrics['macro_f1']:.4f} "
            f"test_weighted_f1={test_metrics['weighted_f1']:.4f}"
        )
        bundle = {
            "config": asdict(config),
            "best_epoch": best_metrics["epoch"],
            "validation_macro_f1": best_metrics["val_metrics"]["macro_f1"],
            "validation_metrics": best_metrics["val_metrics"],
            "test_metrics": test_metrics,
        }
        leaderboard.append(bundle)
        if best_metrics["val_metrics"]["macro_f1"] > best_score:
            best_score = best_metrics["val_metrics"]["macro_f1"]
            best_bundle = {
                "model_state": copy.deepcopy(model.state_dict()),
                "result": bundle,
                "tokenizer": tokenizer,
                "config": config,
            }
            _log(f"{trial_label}: new best config with val_macro_f1={best_score:.4f}")

    assert best_bundle is not None
    best_config: Tier2Config = best_bundle["config"]
    _log(f"{tier_name}: collecting final predictions with best_config={json.dumps(asdict(best_config), sort_keys=True)}")
    best_model = builder(best_config, structured_dim).to(device)
    best_model.load_state_dict(best_bundle["model_state"])
    train_loader, val_loader, test_loader = _make_loaders(
        split_paths=split_paths,
        tokenizer=tokenizer,
        text_field=text_field,
        with_structured=with_structured,
        batch_size=best_config.batch_size,
        max_length=max_length,
        seed=seed + 999,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )
    loss_fn = FocalLoss(alpha=class_weights, gamma=best_config.focal_gamma)
    _, _, train_predictions = _run_loader(
        best_model,
        train_loader,
        device,
        loss_fn,
        optimizer=None,
        scheduler=None,
        collect_predictions=True,
        tier_name=tier_name,
        split_name="train",
        progress_label=f"{tier_name} final train predictions",
        log_every=500,
    )
    _, _, val_predictions = _run_loader(
        best_model,
        val_loader,
        device,
        loss_fn,
        optimizer=None,
        scheduler=None,
        collect_predictions=True,
        tier_name=tier_name,
        split_name="val",
        progress_label=f"{tier_name} final val predictions",
        log_every=500,
    )
    _, final_test_metrics, test_predictions = _run_loader(
        best_model,
        test_loader,
        device,
        loss_fn,
        optimizer=None,
        scheduler=None,
        collect_predictions=True,
        tier_name=tier_name,
        split_name="test",
        progress_label=f"{tier_name} final test predictions",
        log_every=500,
    )

    torch.save(best_bundle["model_state"], output_dir / f"{tier_name}_best.pt")
    best_bundle["tokenizer"].save_pretrained(output_dir / f"{tier_name}_tokenizer")
    prediction_paths = {
        "train": _write_prediction_file(output_dir / "predictions_train.jsonl", train_predictions),
        "val": _write_prediction_file(output_dir / "predictions_val.jsonl", val_predictions),
        "test": _write_prediction_file(output_dir / "predictions_test.jsonl", test_predictions),
    }
    best_bundle["result"]["test_metrics"] = final_test_metrics
    result = {
        "tier": tier_name,
        "model_name": model_name,
        "best_result": best_bundle["result"],
        "prediction_paths": prediction_paths,
        "leaderboard": leaderboard,
    }
    (output_dir / f"{tier_name}_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _log(f"{tier_name}: wrote results to {output_dir / f'{tier_name}_results.json'}")
    return result


def train_tier2a_late_fusion(
    prepared_dir: Path,
    output_dir: Path,
    seed: int = 42,
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _read_metadata(prepared_dir)
    split_paths = {key: Path(value) for key, value in metadata["prepared_split_paths"].items()}
    structured_dim = int(metadata["structured_dim"])
    train_size = _dataset_size(split_paths["train"])
    configs = [
        Tier2Config(learning_rate=2e-5, weight_decay=0.01, batch_size=12, epochs=3, dropout=0.20, freeze_layers=8, focal_gamma=2.0),
        Tier2Config(learning_rate=3e-5, weight_decay=0.02, batch_size=10, epochs=3, dropout=0.20, freeze_layers=6, focal_gamma=1.5),
        Tier2Config(learning_rate=2e-5, weight_decay=0.03, batch_size=8, epochs=4, dropout=0.10, freeze_layers=4, focal_gamma=2.0),
        Tier2Config(learning_rate=1.5e-5, weight_decay=0.02, batch_size=8, epochs=4, dropout=0.10, freeze_layers=2, focal_gamma=2.0),
        Tier2Config(learning_rate=2.5e-5, weight_decay=0.015, batch_size=10, epochs=4, dropout=0.15, freeze_layers=2, focal_gamma=1.5),
    ]
    if train_size <= 50000:
        configs.extend(
            [
                Tier2Config(learning_rate=2e-5, weight_decay=0.01, batch_size=6, epochs=5, dropout=0.10, freeze_layers=0, focal_gamma=2.0),
                Tier2Config(learning_rate=1.5e-5, weight_decay=0.03, batch_size=6, epochs=5, dropout=0.15, freeze_layers=0, focal_gamma=2.0),
            ]
        )

    def builder(config: Tier2Config, dim: int | None) -> nn.Module:
        assert dim is not None
        return LateFusionClinicalBert(
            model_name=model_name,
            structured_dim=dim,
            num_labels=len(LABELS),
            dropout=config.dropout,
            freeze_layers=config.freeze_layers,
        )

    return _search_configs(
        tier_name="tier2a_late_fusion",
        builder=builder,
        split_paths=split_paths,
        structured_dim=structured_dim,
        output_dir=output_dir,
        model_name=model_name,
        text_field="tier2a_text",
        with_structured=True,
        max_length=512,
        seed=seed,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        configs=configs,
    )


def train_tier2b_early_fusion(
    prepared_dir: Path,
    output_dir: Path,
    seed: int = 42,
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _read_metadata(prepared_dir)
    split_paths = {key: Path(value) for key, value in metadata["prepared_split_paths"].items()}
    configs = [
        Tier2Config(learning_rate=2e-5, weight_decay=0.01, batch_size=16, epochs=3, dropout=0.20, freeze_layers=4, focal_gamma=2.0),
        Tier2Config(learning_rate=3e-5, weight_decay=0.02, batch_size=12, epochs=3, dropout=0.20, freeze_layers=2, focal_gamma=1.5),
        Tier2Config(learning_rate=2e-5, weight_decay=0.03, batch_size=8, epochs=4, dropout=0.10, freeze_layers=0, focal_gamma=2.0),
    ]

    def builder(config: Tier2Config, dim: int | None) -> nn.Module:
        return NarrativeClinicalBert(
            model_name=model_name,
            num_labels=len(LABELS),
            dropout=config.dropout,
            freeze_layers=config.freeze_layers,
        )

    return _search_configs(
        tier_name="tier2b_early_fusion",
        builder=builder,
        split_paths=split_paths,
        structured_dim=None,
        output_dir=output_dir,
        model_name=model_name,
        text_field="tier2_narrative",
        with_structured=False,
        max_length=512,
        seed=seed,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        configs=configs,
    )
