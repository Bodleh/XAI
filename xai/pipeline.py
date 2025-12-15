import numpy as np
from xai.explainers import explain_with_lime, explain_with_shap, compute_global_pfi


def predict_and_explain(
    X_train, X_test, y_test,
    model, lime_explainer, shap_explainer,
    x_input,
    class_names,
    show_top_k=10
):
    clf_classes = model.named_steps["clf"].classes_

    proba = model.predict_proba(x_input)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = clf_classes[pred_idx]

    print("PREDICTION")
    print(f"Predicted class: {pred_label} ({class_names[pred_idx]})")
    print(f"Probabilities: {dict(zip(class_names, proba.round(4)))}")
    print()

    # LIME (local)
    print("XAI #1: LIME (Local Explanation)")
    lime_list = explain_with_lime(lime_explainer, model, x_input, pred_idx, num_features=show_top_k)
    for rule, weight in lime_list:
        direction = "mendorong ke kelas prediksi" if weight > 0 else "menjauh dari kelas prediksi"
        print(f"- {rule:20s} | weight={weight:+.4f} ({direction})")
    print()

    # SHAP (local)
    print("XAI #2: SHAP (Local Explanation)")
    shap_idx, shap_contrib = explain_with_shap(
        shap_explainer, model, X_train, x_input, pred_idx=pred_idx, top_k=show_top_k
    )
    print(f"SHAP dihitung untuk kelas index: {shap_idx} ({class_names[shap_idx]})")
    for feat, val in shap_contrib:
        direction = "naikkan skor kelas" if val > 0 else "turunkan skor kelas"
        print(f"- {feat:20s} | shap={val:+.5f} ({direction})")
    print()

    # PFI (global)
    print("XAI #3: Permutation Feature Importance (Global)")
    pfi_top = compute_global_pfi(model, X_test, y_test, top_k=show_top_k)
    print("Fitur paling berpengaruh secara global (semakin besar semakin penting):")
    for feat, imp_mean, imp_std in pfi_top:
        print(f"- {feat:20s} | importance={imp_mean:.5f} +/- {imp_std:.5f}")
    print()

    return pred_label, proba
