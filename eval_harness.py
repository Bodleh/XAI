import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from model.training import build_and_train_model
from xai.explainers import (
    prepare_xai_tools,
    explain_with_lime,
    explain_with_shap,
    compute_global_pfi,
)


def _format_probabilities(proba: np.ndarray, target_names: List[str]) -> Dict[str, float]:
    return {target_names[i]: float(proba[i]) for i in range(len(target_names))}


def run_batch_explanations(
    sample_count: int = 8,
    random_state: int = 42,
    top_k: int = 10,
    output_dir: str = "reports",
) -> Dict[str, Any]:
    """
    Run the classifier on a fixed set of samples and dump explanations for PPT use.

    Returns a dictionary that is also saved as reports/explanations.json
    and CSV sidecars for quick inspection.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data, X_train, X_test, y_train, y_test, model = build_and_train_model(random_state=random_state)
    clf_classes = model.named_steps["clf"].classes_
    class_names = [str(c) for c in clf_classes]
    lime_explainer, shap_explainer = prepare_xai_tools(
        data, X_train, y_train, model, class_names=class_names
    )

    rng = np.random.default_rng(random_state)
    sample_count = min(sample_count, len(X_test))
    chosen_indices = rng.choice(len(X_test), size=sample_count, replace=False)

    records = []
    explanations = []

    for idx in chosen_indices:
        x_input = X_test.iloc[[idx]]
        true_label = str(y_test.iloc[idx])
        proba = model.predict_proba(x_input)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = class_names[pred_idx]

        lime_list = explain_with_lime(lime_explainer, model, x_input, pred_idx, num_features=top_k)
        _, shap_contrib = explain_with_shap(
            shap_explainer, model, X_train, x_input, pred_idx=pred_idx, top_k=top_k
        )

        records.append({
            "sample_index": int(idx),
            "true_label": true_label,
            "predicted_label": pred_label,
            **{f"proba_{class_names[i]}": float(proba[i]) for i in range(len(proba))}
        })

        explanations.append({
            "sample_index": int(idx),
            "true_label": true_label,
            "predicted_label": pred_label,
            "probabilities": _format_probabilities(proba, class_names),
            "lime": [{"feature": feat, "weight": float(w)} for feat, w in lime_list],
            "shap": [{"feature": feat, "value": float(val)} for feat, val in shap_contrib],
        })

    pfi_top = compute_global_pfi(model, X_test, y_test, top_k=top_k)
    pfi_payload = [
        {"feature": feat, "importance_mean": imp_mean, "importance_std": imp_std}
        for feat, imp_mean, imp_std in pfi_top
    ]

    summary = {
        "meta": {
            "dataset": data.details.get("name", "openml_1461") if hasattr(data, "details") else "openml_1461",
            "model": "Preprocess (impute+onehot+scale) + RandomForestClassifier(300)",
            "sample_count": sample_count,
            "random_state": random_state,
            "top_k": top_k,
            "class_names": class_names,
        },
        "global_pfi": pfi_payload,
        "samples": explanations,
    }

    # Save structured outputs for slides/reporting.
    pd.DataFrame(records).to_csv(output_path / "samples_predictions.csv", index=False)
    pd.DataFrame(pfi_payload).to_csv(output_path / "global_pfi.csv", index=False)
    with open(output_path / "explanations.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Friendly console message.
    print(f"[done] Saved {sample_count} samples with LIME/SHAP explanations to {output_path}")
    print(f"       - samples_predictions.csv")
    print(f"       - explanations.json")
    print(f"       - global_pfi.csv")

    return summary


if __name__ == "__main__":
    run_batch_explanations()
