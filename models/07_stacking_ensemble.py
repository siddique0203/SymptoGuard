from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import StackingClassifier
from catboost import CatBoostClassifier
import joblib

from common import (
    RANDOM_STATE,
    MODELS_DIR,
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

# Same idea as previous stacking code: chi-square feature selection + stacking.
selector = SelectKBest(chi2, k=150)

print("Fitting chi-square feature selector on training set only...")
X_train_sel = selector.fit_transform(X_train, y_train)
X_val_sel = selector.transform(X_val)
X_test_sel = selector.transform(X_test)

base_models = [
    ("logistic", LogisticRegression(solver="saga", max_iter=2000, n_jobs=-1, random_state=RANDOM_STATE)),
    ("bernoulli_nb", BernoulliNB(alpha=0.1427)),
    ("catboost", CatBoostClassifier(
        iterations=300,
        depth=8,
        learning_rate=0.05,
        loss_function="MultiClass",
        random_seed=RANDOM_STATE,
        verbose=False,
    )),
]

model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(solver="saga", max_iter=2000, n_jobs=-1, random_state=RANDOM_STATE),
    cv=3,
    n_jobs=-1,
    passthrough=True,
)

print("Training Stacking Ensemble on 70% training set...")
model.fit(X_train_sel, y_train)

val_metrics, val_pred = evaluate_model(model, X_val_sel, y_val, "validation", label_encoder)
test_metrics, test_pred = evaluate_model(model, X_test_sel, y_test, "final_test", label_encoder)

val_metrics["model"] = "Stacking Ensemble"
test_metrics["model"] = "Stacking Ensemble"
val_metrics["selected_features"] = 150
test_metrics["selected_features"] = 150

save_metrics([val_metrics, test_metrics], "stacking_ensemble_metrics.csv")
save_report(y_test, test_pred, label_encoder, "stacking_ensemble_test_report.txt")

save_model(model, label_encoder, X.columns, "stacking_ensemble_70_15_15.pkl")
joblib.dump(selector, MODELS_DIR / "stacking_feature_selector.pkl")
print(f"Saved stacking feature selector: {MODELS_DIR / 'stacking_feature_selector.pkl'}")
