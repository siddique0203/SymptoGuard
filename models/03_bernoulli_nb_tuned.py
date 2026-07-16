import optuna
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

from common import (
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

def objective(trial):
    alpha = trial.suggest_float("alpha", 0.01, 10.0, log=True)
    model = BernoulliNB(alpha=alpha)
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    return accuracy_score(y_val, val_pred)

print(f"Tuning Bernoulli Naive Bayes using validation set ({N_TRIALS} trials)...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS)

best_alpha = study.best_params["alpha"]
print(f"Best alpha: {best_alpha:.6f}")
print(f"Best validation accuracy during tuning: {study.best_value:.4f}")

model = BernoulliNB(alpha=best_alpha)

print("Training tuned Bernoulli Naive Bayes on 70% training set...")
model.fit(X_train, y_train)

val_metrics, val_pred = evaluate_model(model, X_val, y_val, "validation", label_encoder)
test_metrics, test_pred = evaluate_model(model, X_test, y_test, "final_test", label_encoder)

val_metrics["model"] = "Bernoulli Naive Bayes (Tuned)"
test_metrics["model"] = "Bernoulli Naive Bayes (Tuned)"
val_metrics["best_alpha"] = best_alpha
test_metrics["best_alpha"] = best_alpha

save_metrics([val_metrics, test_metrics], "bernoulli_nb_tuned_metrics.csv")
save_report(y_test, test_pred, label_encoder, "bernoulli_nb_tuned_test_report.txt")
save_model(model, label_encoder, X.columns, "best_nb_model_70_15_15.pkl")
