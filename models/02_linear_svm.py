from sklearn.svm import LinearSVC

from common import (
    RANDOM_STATE,
    load_and_preprocess,
    split_70_15_15,
    print_split_info,
    evaluate_model,
    save_metrics,
    save_report,
    save_model,
)

print("Loading data...")
X, y, label_encoder, zero_var_cols = load_and_preprocess()
X_train, X_val, X_test, y_train, y_val, y_test = split_70_15_15(X, y)
print_split_info(X_train, X_val, X_test)

model = LinearSVC(
    C=1.0,
    max_iter=5000,
    random_state=RANDOM_STATE,
)

print("Training Linear SVM on 70% training set...")
model.fit(X_train, y_train)

val_metrics, val_pred = evaluate_model(model, X_val, y_val, "validation", label_encoder)
test_metrics, test_pred = evaluate_model(model, X_test, y_test, "final_test", label_encoder)

val_metrics["model"] = "SVM (Linear)"
test_metrics["model"] = "SVM (Linear)"

save_metrics([val_metrics, test_metrics], "linear_svm_metrics.csv")
save_report(y_test, test_pred, label_encoder, "linear_svm_test_report.txt")
save_model(model, label_encoder, X.columns, "linear_svm_70_15_15.pkl")
