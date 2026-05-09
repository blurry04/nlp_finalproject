from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.svm import LinearSVC

from .constants import (
    CATEGORICAL_FEATURES,
    LABELS,
    NUMERIC_FEATURES,
    RAW_TEXT_FIELDS,
    STRUCTURED_FEATURE_LABELS,
)
from .metrics import compute_classification_metrics
from .text_utils import tier2_handoff_narrative
from .training_data import iter_jsonl, load_split_frame


def _read_metadata(prepared_dir: Path) -> dict[str, Any]:
    metadata_path = prepared_dir / "metadata.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _raw_columns() -> list[str]:
    columns = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "los_bucket",
        "label_id",
        "tagged_text",
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
        *RAW_TEXT_FIELDS,
    ]
    return list(dict.fromkeys(columns))


def _load_raw_splits(prepared_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = _read_metadata(prepared_dir)
    needed_columns = _raw_columns()
    train_df = load_split_frame(Path(metadata["raw_split_paths"]["train"]), columns=needed_columns)
    val_df = load_split_frame(Path(metadata["raw_split_paths"]["val"]), columns=needed_columns)
    test_df = load_split_frame(Path(metadata["raw_split_paths"]["test"]), columns=needed_columns)
    return train_df, val_df, test_df


def _class_sample_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=len(LABELS))
    weights = np.zeros_like(counts, dtype=np.float32)
    total = counts.sum()
    for idx, count in enumerate(counts):
        weights[idx] = total / (len(LABELS) * max(count, 1))
    return weights[y]


def _to_dense(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _xgboost_device() -> str:
    try:
        build_info = xgb.build_info()
        use_cuda = str(build_info.get("USE_CUDA", "")).lower()
        if use_cuda in {"1", "true", "yes", "on"}:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("subject_id", "")).strip(),
        str(record.get("hadm_id", "")).strip(),
        str(record.get("stay_id", "")).strip(),
    )


def _format_number(value: Any, decimals: int = 1) -> str:
    if value in ("", None):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    if number.is_integer():
        return str(int(number))
    return f"{number:.{decimals}f}".rstrip("0").rstrip(".")


def _split_feature_name(feature_name: str) -> tuple[str, str, str]:
    if feature_name.startswith("num__"):
        return ("num", feature_name.split("__", 1)[1], "")
    if feature_name.startswith("cat__"):
        remainder = feature_name.split("__", 1)[1]
        for feature in CATEGORICAL_FEATURES:
            prefix = f"{feature}_"
            if remainder.startswith(prefix):
                return ("cat", feature, remainder[len(prefix) :])
        return ("cat", remainder, "")
    return ("unknown", feature_name, "")


def _feature_label(base_feature: str) -> str:
    return STRUCTURED_FEATURE_LABELS.get(base_feature, base_feature.replace("_", " ").title())


def _feature_display(feature_name: str, row: dict[str, Any]) -> tuple[str, str]:
    kind, base_feature, category_value = _split_feature_name(feature_name)
    label = _feature_label(base_feature)
    if kind == "num":
        raw_value = row.get(base_feature)
        value = _format_number(raw_value, decimals=1)
        if base_feature == "o2sat" and value:
            value = f"{value}%"
        elif base_feature == "ed_dwell_minutes" and value:
            value = f"{value} min"
        return label, value
    display_value = str(row.get(base_feature, "") or "").strip() or category_value.replace("_", " ").strip()
    return label, display_value


def _evidence_text(feature_name: str, row: dict[str, Any]) -> str:
    kind, base_feature, _ = _split_feature_name(feature_name)
    if kind == "num":
        value = _format_number(row.get(base_feature), decimals=1)
        if not value:
            return ""
        templates = {
            "anchor_age": f"age {value}",
            "acuity": f"triage acuity {value}",
            "temperature": f"temperature {value}",
            "heartrate": f"heart rate {value}",
            "resprate": f"respiratory rate {value}",
            "o2sat": f"O2 saturation {value}%",
            "sbp": f"systolic BP {value}",
            "dbp": f"diastolic BP {value}",
            "pain": f"pain score {value}",
            "ed_dwell_minutes": f"ED dwell {value} minutes",
            "med_count": f"{value} home medications documented",
            "radiology_note_count": f"{value} radiology notes in the first 48 hours",
        }
        return templates.get(base_feature, f"{_feature_label(base_feature).lower()} {value}")
    label, display_value = _feature_display(feature_name, row)
    if not display_value:
        return ""
    return f"{label.lower()} {display_value}"


def _feature_manifest_entry(feature_name: str, score: float) -> dict[str, Any]:
    kind, base_feature, category_value = _split_feature_name(feature_name)
    feature = _feature_label(base_feature)
    if kind == "cat" and category_value:
        feature = f"{feature}: {category_value.replace('_', ' ')}"
    return {
        "feature_name": feature_name,
        "feature": feature,
        "mean_abs_shap": float(score),
    }


def _extract_pred_contribs(model: xgb.XGBClassifier, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)
    contribs = booster.predict(dmatrix, pred_contribs=True, strict_shape=True)
    array = np.asarray(contribs, dtype=np.float32)
    if array.ndim == 2:
        array = array[:, np.newaxis, :]
    return array


def _top_contributions_for_row(
    row: dict[str, Any],
    feature_names: list[str],
    pred_contribs: np.ndarray,
    selected_feature_names: set[str],
    top_k: int = 6,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for feature_idx, feature_name in enumerate(feature_names):
        if feature_name not in selected_feature_names:
            continue
        score = float(pred_contribs[feature_idx])
        if abs(score) < 1e-8:
            continue
        label, display_value = _feature_display(feature_name, row)
        feature = label if not display_value else f"{label}: {display_value}"
        evidence_text = _evidence_text(feature_name, row)
        entries.append(
            {
                "feature_name": feature_name,
                "feature": feature,
                "value": score,
                "evidence_text": evidence_text,
            }
        )
    entries.sort(key=lambda item: abs(float(item["value"])), reverse=True)
    return entries[:top_k]


def _prediction_record(
    row: dict[str, Any],
    scores: np.ndarray,
    tier: str,
    split: str,
    shap_top_features: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    predicted_id = int(np.argmax(scores))
    actual_id = int(row["label_id"])
    predicted = LABELS[predicted_id]
    actual = LABELS[actual_id]
    return {
        "subject_id": str(row.get("subject_id", "")),
        "hadm_id": str(row.get("hadm_id", "")),
        "stay_id": str(row.get("stay_id", "")),
        "tier": tier,
        "split": split,
        "actual": actual,
        "predicted": predicted,
        "correct": predicted_id == actual_id,
        "confidence_raw": float(scores[predicted_id]),
        "probabilities": {label: float(scores[idx]) for idx, label in enumerate(LABELS)},
        "shap_top_features": shap_top_features or [],
    }


def _write_prediction_file(
    output_path: Path,
    frame: pd.DataFrame,
    scores: np.ndarray,
    tier: str,
    split: str,
    shap_rows: list[list[dict[str, Any]]] | None = None,
) -> str:
    records = frame.to_dict(orient="records")
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(records):
            shap_top_features = shap_rows[idx] if shap_rows is not None else None
            handle.write(
                json.dumps(
                    _prediction_record(row=row, scores=scores[idx], tier=tier, split=split, shap_top_features=shap_top_features),
                    ensure_ascii=False,
                )
                + "\n"
            )
    return str(output_path)


def train_tier0_structured(prepared_dir: Path, output_dir: Path, seed: int = 42) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _read_metadata(prepared_dir)
    preprocessor = joblib.load(metadata["structured_preprocessor_path"])
    train_df, val_df, test_df = _load_raw_splits(prepared_dir)

    X_train = _to_dense(preprocessor.transform(train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]))
    X_val = _to_dense(preprocessor.transform(val_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]))
    X_test = _to_dense(preprocessor.transform(test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]))

    y_train = train_df["label_id"].to_numpy(dtype=np.int64)
    y_val = val_df["label_id"].to_numpy(dtype=np.int64)
    y_test = test_df["label_id"].to_numpy(dtype=np.int64)
    train_weights = _class_sample_weights(y_train)

    device = _xgboost_device()
    candidate_params = [
        {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 900, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 1.0, "reg_lambda": 1.0},
        {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 700, "subsample": 0.90, "colsample_bytree": 0.80, "min_child_weight": 1.0, "reg_lambda": 1.5},
        {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 650, "subsample": 0.85, "colsample_bytree": 0.90, "min_child_weight": 2.0, "reg_lambda": 1.0},
        {"max_depth": 7, "learning_rate": 0.04, "n_estimators": 600, "subsample": 0.80, "colsample_bytree": 0.80, "min_child_weight": 3.0, "reg_lambda": 2.0},
    ]

    leaderboard: list[dict[str, Any]] = []
    best_model = None
    best_params = None
    best_score = -1.0
    for params in candidate_params:
        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=len(LABELS),
            random_state=seed,
            device=device,
            tree_method="hist",
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params["min_child_weight"],
            reg_lambda=params["reg_lambda"],
            eval_metric="mlogloss",
        )
        model.fit(
            X_train,
            y_train,
            sample_weight=train_weights,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        val_scores = model.predict_proba(X_val)
        val_pred = val_scores.argmax(axis=1)
        val_metrics = compute_classification_metrics(y_val, val_pred, val_scores)
        leaderboard.append({"params": params, "val_macro_f1": val_metrics["macro_f1"]})
        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            best_model = model
            best_params = params

    assert best_model is not None
    feature_names = preprocessor.get_feature_names_out().tolist()
    train_scores = best_model.predict_proba(X_train)
    val_scores = best_model.predict_proba(X_val)
    test_scores = best_model.predict_proba(X_test)
    test_pred = test_scores.argmax(axis=1)
    test_metrics = compute_classification_metrics(y_test, test_pred, test_scores)

    train_contribs = _extract_pred_contribs(best_model, X_train, feature_names)
    global_importance = np.abs(train_contribs[:, :, :-1]).mean(axis=(0, 1))
    top_idx = np.argsort(global_importance)[::-1][:12]
    selected_features = [_feature_manifest_entry(feature_names[idx], global_importance[idx]) for idx in top_idx]
    selected_feature_names = {item["feature_name"] for item in selected_features}

    prediction_paths: dict[str, str] = {}
    split_artifacts = [
        ("train", train_df, train_scores, _extract_pred_contribs(best_model, X_train, feature_names)),
        ("val", val_df, val_scores, _extract_pred_contribs(best_model, X_val, feature_names)),
        ("test", test_df, test_scores, _extract_pred_contribs(best_model, X_test, feature_names)),
    ]
    for split_name, frame, scores, contribs in split_artifacts:
        rows = frame.to_dict(orient="records")
        shap_rows: list[list[dict[str, Any]]] = []
        for row_idx, row in enumerate(rows):
            predicted_id = int(np.argmax(scores[row_idx]))
            pred_contribs = contribs[row_idx, predicted_id, :-1]
            shap_rows.append(
                _top_contributions_for_row(
                    row=row,
                    feature_names=feature_names,
                    pred_contribs=pred_contribs,
                    selected_feature_names=selected_feature_names,
                )
            )
        prediction_paths[split_name] = _write_prediction_file(
            output_path=output_dir / f"predictions_{split_name}.jsonl",
            frame=frame,
            scores=scores,
            tier="tier0_xgboost_structured",
            split=split_name,
            shap_rows=shap_rows,
        )

    selected_features_path = output_dir / "selected_structured_features.json"
    selected_features_path.write_text(json.dumps(selected_features, indent=2), encoding="utf-8")
    joblib.dump(best_model, output_dir / "tier0_xgboost.joblib")
    result = {
        "tier": "tier0_xgboost_structured",
        "best_params": best_params,
        "validation_macro_f1": best_score,
        "test_metrics": test_metrics,
        "selected_structured_features": selected_features,
        "selected_structured_features_path": str(selected_features_path),
        "prediction_paths": prediction_paths,
        "leaderboard": leaderboard,
    }
    (output_dir / "tier0_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


def _top_terms(model: Any, vectorizer: TfidfVectorizer, top_k: int = 20) -> dict[str, list[str]]:
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    coef = model.coef_
    top: dict[str, list[str]] = {}
    for class_idx, label in enumerate(LABELS):
        class_coef = coef[class_idx]
        top_idx = np.argsort(class_coef)[-top_k:][::-1]
        top[label] = feature_names[top_idx].tolist()
    return top


def _score_sparse_model(model: Any, X: Any, model_name: str) -> np.ndarray:
    if model_name == "logreg":
        return model.predict_proba(X)
    raw_scores = model.decision_function(X)
    if raw_scores.ndim == 1:
        raw_scores = np.stack([-raw_scores, raw_scores], axis=1)
    return _softmax(raw_scores)


def train_tier1_sparse_text(prepared_dir: Path, output_dir: Path, seed: int = 42) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = _load_raw_splits(prepared_dir)
    y_train = train_df["label_id"].to_numpy(dtype=np.int64)
    y_val = val_df["label_id"].to_numpy(dtype=np.int64)
    y_test = test_df["label_id"].to_numpy(dtype=np.int64)
    train_text = train_df["tagged_text"].fillna("").tolist()
    val_text = val_df["tagged_text"].fillna("").tolist()
    test_text = test_df["tagged_text"].fillna("").tolist()

    vectorizer_grid = list(
        ParameterGrid(
            {
                "max_features": [30000, 50000],
                "min_df": [2, 5],
                "ngram_range": [(1, 2)],
                "sublinear_tf": [True],
            }
        )
    )
    classifier_specs = [
        ("logreg", {"C": 0.5}),
        ("logreg", {"C": 1.0}),
        ("logreg", {"C": 3.0}),
        ("linearsvc", {"C": 0.5}),
        ("linearsvc", {"C": 1.0}),
        ("linearsvc", {"C": 2.0}),
    ]

    leaderboard: list[dict[str, Any]] = []
    best_bundle: dict[str, Any] | None = None
    best_score = -1.0
    for vec_params in vectorizer_grid:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            max_features=vec_params["max_features"],
            min_df=vec_params["min_df"],
            ngram_range=vec_params["ngram_range"],
            sublinear_tf=vec_params["sublinear_tf"],
        )
        X_train = vectorizer.fit_transform(train_text)
        X_val = vectorizer.transform(val_text)
        for model_name, model_params in classifier_specs:
            if model_name == "logreg":
                model = LogisticRegression(
                    C=model_params["C"],
                    class_weight="balanced",
                    max_iter=1500,
                    solver="saga",
                    random_state=seed,
                )
            else:
                model = LinearSVC(
                    C=model_params["C"],
                    class_weight="balanced",
                    random_state=seed,
                )
            model.fit(X_train, y_train)
            val_scores = _score_sparse_model(model, X_val, model_name=model_name)
            val_pred = val_scores.argmax(axis=1)
            val_metrics = compute_classification_metrics(y_val, val_pred, val_scores)
            leaderboard.append(
                {
                    "vectorizer": vec_params,
                    "classifier": {"name": model_name, **model_params},
                    "val_macro_f1": val_metrics["macro_f1"],
                }
            )
            if val_metrics["macro_f1"] > best_score:
                best_score = val_metrics["macro_f1"]
                best_bundle = {
                    "vectorizer": vectorizer,
                    "model": model,
                    "model_name": model_name,
                    "vectorizer_params": vec_params,
                    "model_params": model_params,
                }

    assert best_bundle is not None
    best_model = best_bundle["model"]
    vectorizer = best_bundle["vectorizer"]
    train_scores = _score_sparse_model(best_model, vectorizer.transform(train_text), model_name=best_bundle["model_name"])
    val_scores = _score_sparse_model(best_model, vectorizer.transform(val_text), model_name=best_bundle["model_name"])
    test_scores = _score_sparse_model(best_model, vectorizer.transform(test_text), model_name=best_bundle["model_name"])
    test_pred = test_scores.argmax(axis=1)
    test_metrics = compute_classification_metrics(y_test, test_pred, test_scores)
    top_terms = _top_terms(best_model, vectorizer)

    prediction_paths = {
        "train": _write_prediction_file(output_dir / "predictions_train.jsonl", train_df, train_scores, "tier1_sparse_text", "train"),
        "val": _write_prediction_file(output_dir / "predictions_val.jsonl", val_df, val_scores, "tier1_sparse_text", "val"),
        "test": _write_prediction_file(output_dir / "predictions_test.jsonl", test_df, test_scores, "tier1_sparse_text", "test"),
    }

    joblib.dump(vectorizer, output_dir / "tier1_vectorizer.joblib")
    joblib.dump(best_model, output_dir / "tier1_sparse_model.joblib")
    result = {
        "tier": "tier1_sparse_text",
        "best_vectorizer_params": best_bundle["vectorizer_params"],
        "best_model": {"name": best_bundle["model_name"], **best_bundle["model_params"]},
        "validation_macro_f1": best_score,
        "test_metrics": test_metrics,
        "prediction_paths": prediction_paths,
        "top_terms": top_terms,
        "leaderboard": leaderboard,
    }
    (output_dir / "tier1_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_tier2_handoff(
    prepared_dir: Path,
    tier0_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_metadata = _read_metadata(prepared_dir)
    prediction_paths = {
        split: tier0_dir / f"predictions_{split}.jsonl"
        for split in ("train", "val", "test")
    }
    for split, path in prediction_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Tier 0 prediction file not found for split '{split}': {path}")

    out_paths = {
        split: output_dir / f"{split}.jsonl"
        for split in ("train", "val", "test")
    }
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        prediction_map = {
            _record_key(record): record
            for record in iter_jsonl(prediction_paths[split])
        }
        count = 0
        with Path(base_metadata["prepared_split_paths"][split]).open(encoding="utf-8", errors="replace") as in_handle, out_paths[split].open("w", encoding="utf-8") as out_handle:
            for line in in_handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                prediction = prediction_map.get(_record_key(record))
                if prediction is None:
                    raise KeyError(f"Missing Tier 0 prediction artifact for record {record.get('hadm_id')}")
                structured_highlights = prediction.get("shap_top_features", [])
                record["tier0_prediction"] = prediction.get("predicted", "")
                record["tier0_confidence_raw"] = prediction.get("confidence_raw", 0.0)
                record["tier0_probabilities"] = prediction.get("probabilities", {})
                record["tier0_shap_top_features"] = structured_highlights
                record["tier2_narrative"] = tier2_handoff_narrative(record, structured_highlights=structured_highlights)
                out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        counts[split] = count

    metadata = {
        **base_metadata,
        "source_prepared_dir": str(prepared_dir.resolve()),
        "tier0_dir": str(tier0_dir.resolve()),
        "prepared_split_paths": {split: str(path.resolve()) for split, path in out_paths.items()},
        "handoff_split_counts": counts,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
