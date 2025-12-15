import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import permutation_importance
def prepare_xai_tools(data, X_train, y_train, model, class_names=None):
    preprocess = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]

    X_train_enc = preprocess.transform(X_train)

    if hasattr(X_train_enc, "toarray"):
        X_train_lime = X_train_enc[:5000].toarray()
    else:
        X_train_lime = np.array(X_train_enc[:5000])

    if class_names is None:
        class_names = [str(c) for c in clf.classes_]

    lime_explainer = LimeTabularExplainer(
        training_data=X_train_lime,
        feature_names=preprocess.get_feature_names_out(),
        class_names=class_names,
        mode="classification",
        discretize_continuous=False
    )

    shap_explainer = shap.TreeExplainer(clf)

    return lime_explainer, shap_explainer

def explain_with_lime(lime_explainer, model, X_instance, pred_idx, num_features=10):
    preprocess = model.named_steps["preprocess"]

    x_enc = preprocess.transform(X_instance)
    if hasattr(x_enc, "toarray"):
        x_enc = x_enc.toarray()
    else:
        x_enc = np.array(x_enc)

    def predict_fn_encoded(X_encoded):
        return model.named_steps["clf"].predict_proba(X_encoded)

    exp = lime_explainer.explain_instance(
        data_row=x_enc[0],
        predict_fn=predict_fn_encoded,
        num_features=num_features,
        labels=[pred_idx],
    )
    return exp.as_list(label=pred_idx)

def explain_with_shap(shap_explainer, model, X_train, X_instance, pred_idx, top_k=10):
    preprocess = model.named_steps["preprocess"]
    x_enc = preprocess.transform(X_instance)

    try:
        values = shap_explainer(x_enc).values
    except TypeError:
        values = shap_explainer.shap_values(x_enc)

    if isinstance(values, list):
        sv = np.array(values[pred_idx])[0]
    else:
        v = np.array(values)
        sv = v[0, :, pred_idx] if v.ndim == 3 else v[0, :]

    feat_names = preprocess.get_feature_names_out()
    contrib = sorted(
        [(feat_names[i], float(sv[i])) for i in range(len(feat_names))],
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_k]

    return pred_idx, contrib

def compute_global_pfi(model, X_test: pd.DataFrame, y_test, top_k: int = 10, random_state: int = 42):
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=50, random_state=random_state, n_jobs=-1,
        scoring="f1_macro"
    )

    importances = pd.Series(result.importances_mean, index=X_test.columns)
    importances_std = pd.Series(result.importances_std, index=X_test.columns)

    top = importances.sort_values(ascending=False).head(top_k)
    return [(feat, float(top[feat]), float(importances_std[feat])) for feat in top.index]
