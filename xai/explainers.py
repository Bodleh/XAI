import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import permutation_importance

def prepare_xai_tools(data, X_train: pd.DataFrame, model):
    lime_explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=data.target_names.tolist(),
        mode="classification",
        discretize_continuous=True
    )

    rf_model = model.named_steps["clf"]
    shap_explainer = shap.TreeExplainer(rf_model)

    return lime_explainer, shap_explainer


def explain_with_lime(lime_explainer, model, X_instance: pd.DataFrame, pred_class: int, num_features: int = 10):
    exp = lime_explainer.explain_instance(
        data_row=X_instance.values[0],
        predict_fn=model.predict_proba,
        num_features=num_features,
        labels=[pred_class],
    )
    return exp.as_list(label=pred_class)


def explain_with_shap(shap_explainer, model, X_train: pd.DataFrame, X_instance: pd.DataFrame, top_k: int = 10):
    scaler = model.named_steps["scaler"]

    x_scaled = scaler.transform(X_instance)
    pred_class = int(model.predict(X_instance)[0])

    try:
        shap_out = shap_explainer(x_scaled)
        values = shap_out.values
    except TypeError:
        values = shap_explainer.shap_values(x_scaled)

    if isinstance(values, list):
        sv = np.array(values[pred_class])[0]
    else:
        v = np.array(values)
        if v.ndim == 3:
            class_dim = v.shape[2]
            cls = pred_class if class_dim > 1 else 0
            sv = v[0, :, cls]
        elif v.ndim == 2:
            sv = v[0, :]
        elif v.ndim == 1:
            sv = v
        else:
            raise ValueError(f"Unexpected SHAP values shape: {v.shape}")

    feature_names = X_train.columns.tolist()
    contrib = sorted(
        [(feature_names[i], float(sv[i])) for i in range(len(feature_names))],
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_k]

    return pred_class, contrib


def compute_global_pfi(model, X_test: pd.DataFrame, y_test, top_k: int = 10, random_state: int = 42):
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=50, random_state=random_state, n_jobs=-1,
        scoring="f1"
    )

    importances = pd.Series(result.importances_mean, index=X_test.columns)
    importances_std = pd.Series(result.importances_std, index=X_test.columns)

    top = importances.sort_values(ascending=False).head(top_k)
    return [(feat, float(top[feat]), float(importances_std[feat])) for feat in top.index]
