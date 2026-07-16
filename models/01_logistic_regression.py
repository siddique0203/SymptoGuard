from sklearn.linear_model import LogisticRegression

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

model = LogisticRegression(
    solver="saga",
    max_iter=2000,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

print("Training Logistic Regression on 70% training set...")
model.fit(X_train, y_train)

val_metrics, val_pred = evaluate_model(model, X_val, y_val, "validation", label_encoder)
test_metrics, test_pred = evaluate_model(model, X_test, y_test, "final_test", label_encoder)

val_metrics["model"] = "Logistic Regression"
test_metrics["model"] = "Logistic Regression"

save_metrics([val_metrics, test_metrics], "logistic_regression_metrics.csv")
save_report(y_test, test_pred, label_encoder, "logistic_regression_test_report.txt")
save_model(model, label_encoder, X.columns, "logistic_regression_70_15_15.pkl")
