from model.training import build_and_train_model
from xai.explainers import prepare_xai_tools
from xai.pipeline import predict_and_explain

def main():
    data, X_train, X_test, y_train, y_test, model = build_and_train_model()
    lime_explainer, shap_explainer = prepare_xai_tools(data, X_train, y_train, model)

    x_input = X_test.iloc[[0]]

    clf_classes = model.named_steps["clf"].classes_
    class_names = [f"class_{c}" for c in clf_classes]

    predict_and_explain(
        X_train, X_test, y_test,
        model, lime_explainer, shap_explainer,
        x_input,
        class_names=class_names,
        show_top_k=10
    )

if __name__ == "__main__":
    main()
