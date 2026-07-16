import optuna
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

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

N_TRIALS = 20

print("Loading data...")
X, y, label_encoder, zero_var_cols = load_and_preprocess()
X_train, X_val, X_test, y_train, y_val, y_test = split_70_15_15(X, y)
print_split_info(X_train, X_val, X_test)

num_classes = len(label_encoder.classes_)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30),
        "subsample": trial.suggest_float("subsample", 0.60, 1.00),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.00),
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    return accuracy_score(y_val, val_pred)

print(f"Tuning XGBoost using validation set ({N_TRIALS} trials)...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS)

print("Best XGBoost parameters:")
print(study.best_params)
print(f"Best validation accuracy during tuning: {study.best_value:.4f}")

best_params = {
    **study.best_params,
    "objective": "multi:softprob",
    "num_class": num_classes,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

model = XGBClassifier(**best_params)

print("Training tuned XGBoost on 70% training set...")
model.fit(X_train, y_train)

val_metrics, val_pred = evaluate_model(model, X_val, y_val, "validation", label_encoder)
test_metrics, test_pred = evaluate_model(model, X_test, y_test, "final_test", label_encoder)

val_metrics["model"] = "XGBoost (Tuned)"
test_metrics["model"] = "XGBoost (Tuned)"

save_metrics([val_metrics, test_metrics], "xgboost_tuned_metrics.csv")
save_report(y_test, test_pred, label_encoder, "xgboost_tuned_test_report.txt")
save_model(model, label_encoder, X.columns, "xgboost_tuned_70_15_15.pkl")
