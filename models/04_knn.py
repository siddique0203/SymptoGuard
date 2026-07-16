from sklearn.neighbors import KNeighborsClassifier

from common import (
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

# Hamming distance is appropriate for binary symptom vectors.
model = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance",
    metric="hamming",
    n_jobs=-1,
)

print("Training KNN on 70% training set...")
model.fit(X_train, y_train)

val_metrics, val_pred = evaluate_model(model, X_val, y_val, "validation", label_encoder)
test_metrics, test_pred = evaluate_model(model, X_test, y_test, "final_test", label_encoder)

val_metrics["model"] = "KNN"
test_metrics["model"] = "KNN"

save_metrics([val_metrics, test_metrics], "knn_metrics.csv")
save_report(y_test, test_pred, label_encoder, "knn_test_report.txt")
save_model(model, label_encoder, X.columns, "knn_70_15_15.pkl")
