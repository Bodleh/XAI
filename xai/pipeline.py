import numpy as np
import pandas as pd

from xai.explainers import (
    explain_with_lime,
    explain_with_shap,
    compute_global_pfi,
)

def predict_and_explain(
    data, X_train: pd.DataFrame, X_test: pd.DataFrame, y_test,
    model, lime_explainer, shap_explainer,
    x_input: pd.DataFrame,
    show_top_k: int = 10
):
    proba = model.predict_proba(x_input)[0]
    pred = int(np.argmax(proba))
    label_name = data.target_names[pred]

    print("PREDICTION")
    print(f"Predicted class: {pred} ({label_name})")
    print(f"Probabilities: {dict(zip(data.target_names.tolist(), proba.round(4)))}")
    print()

    # LIME
    print("XAI #1: LIME (Local Explanation)")
    pred_class = int(model.predict(x_input)[0])
    lime_list = explain_with_lime(lime_explainer, model, x_input, pred_class, num_features=show_top_k)
    for rule, weight in lime_list:
        direction = "mendorong ke kelas prediksi" if weight > 0 else "menjauh dari kelas prediksi"
        print(f"- {rule:45s} | weight={weight:+.4f} ({direction})")
    print()

    # SHAP
    print("XAI #2: SHAP (Local Explanation)")
    pred_class, shap_contrib = explain_with_shap(shap_explainer, model, X_train, x_input, top_k=show_top_k)
    print(f"SHAP dihitung untuk kelas: {pred_class} ({data.target_names[pred_class]})")
    for feat, val in shap_contrib:
        direction = "naikkan skor kelas" if val > 0 else "turunkan skor kelas"
        print(f"- {feat:30s} | shap={val:+.5f} ({direction})")
    print()

    # PFI
    print("XAI #3: Permutation Feature Importance (Global)")
    pfi_top = compute_global_pfi(model, X_test, y_test, top_k=show_top_k)
    print("Fitur paling berpengaruh secara global (semakin besar semakin penting):")
    for feat, imp_mean, imp_std in pfi_top:
        print(f"- {feat:30s} | importance={imp_mean:.5f} ± {imp_std:.5f}")
    print()

    return pred, proba
