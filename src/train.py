"""Train and compare yield models, then score the held-out test set once.

Usage:
    python src/train.py

Tuning uses 5-fold stratified cross-validation inside the training set only. The test set is
scored a single time at the end, so its numbers are an honest estimate of performance on new
applicants. (The original notebook picked settings by their test-set score, which makes the
test score optimistic.)

Each model outputs a probability of enrolling. Two kinds of results are reported:
    ranking quality   ROC AUC: how well the model orders admits from least to most likely to enroll
    yes/no decisions  Cohen's kappa: agreement beyond chance, using a probability cutoff chosen
                      on training-set cross-validation (with a 22% enrollment rate, a 50% cutoff
                      is rarely the best one)
"""
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import CV_FOLDS, DEMOGRAPHIC_FEATURES, RANDOM_STATE, data_source
from prepare import CATEGORICAL, LEAKY, NUMERIC, Preprocessor, load_split

CV = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

MODELS = {
    "Logistic regression": (
        LogisticRegression(max_iter=5000),
        {"model__C": [0.01, 0.03, 0.1, 0.3, 1.0]},
    ),
    "Random forest": (
        RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1),
        {"model__max_features": [0.1, 0.2, 0.35], "model__min_samples_leaf": [3, 10, 25]},
    ),
    "Gradient boosting": (
        HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        {"model__learning_rate": [0.03, 0.1], "model__max_leaf_nodes": [7, 15, 31],
         "model__max_iter": [200, 400]},
    ),
}


def make_pipeline(model, drop=(), add=()):
    cats = [c for c in CATEGORICAL if c not in drop]
    nums = [c for c in NUMERIC if c not in drop] + list(add)
    encode = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
        ("num", StandardScaler(), nums),
    ])
    return Pipeline([("encode", encode), ("model", model)])


def best_kappa_cutoff(y, proba):
    cutoffs = np.arange(0.10, 0.71, 0.01)
    kappas = [cohen_kappa_score(y, proba >= c) for c in cutoffs]
    return float(cutoffs[int(np.argmax(kappas))])


def fit_and_score(name, model, grid, X_train, y_train, X_test, y_test, drop=(), add=()):
    search = GridSearchCV(make_pipeline(model, drop, add), grid, scoring="roc_auc", cv=CV, n_jobs=-1)
    search.fit(X_train, y_train)
    oof = cross_val_predict(clone(search.best_estimator_), X_train, y_train, cv=CV, method="predict_proba")[:, 1]
    cutoff = best_kappa_cutoff(y_train, oof)
    proba = search.best_estimator_.predict_proba(X_test)[:, 1]
    pred = proba >= cutoff
    row = {
        "model": name,
        "cv_auc": round(search.best_score_, 4),
        "test_auc": round(roc_auc_score(y_test, proba), 4),
        "cutoff": round(cutoff, 2),
        "test_kappa": round(cohen_kappa_score(y_test, pred), 4),
        "test_kappa_at_0.5": round(cohen_kappa_score(y_test, proba >= 0.5), 4),
        "test_accuracy": round(accuracy_score(y_test, pred), 4),
        "brier": round(brier_score_loss(y_test, proba), 4),
        "predicted_enrollees": round(float(proba.sum()), 1),
        "actual_enrollees": int(y_test.sum()),
        "best_params": json.dumps({k.replace("model__", ""): v for k, v in search.best_params_.items()}),
    }
    return row, search.best_estimator_, proba


def main():
    path, out, label = data_source()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Using {label} data: {path.name}")
    raw_train, raw_test, y_train, y_test = load_split(path)
    prep = Preprocessor().fit(raw_train)
    X_train, X_test = prep.transform(raw_train), prep.transform(raw_test)

    base_rate = y_test.mean()
    print(f"Test set: {len(y_test)} admits, {base_rate:.1%} enrolled. "
          f"Predicting 'won't enroll' for everyone is {1 - base_rate:.1%} accurate (kappa 0).")

    # ---- Model comparison, all features ----
    rows, fitted = [], {}
    for name, (model, grid) in MODELS.items():
        print(f"Tuning {name}...", flush=True)
        row, est, proba = fit_and_score(name, model, grid, X_train, y_train, X_test, y_test)
        rows.append(row)
        fitted[name] = (est, proba)
        print(f"  CV AUC {row['cv_auc']:.3f} | test AUC {row['test_auc']:.3f} | test kappa {row['test_kappa']:.3f}")
    results = pd.DataFrame(rows).sort_values("cv_auc", ascending=False)
    results.to_csv(out / "model_comparison.csv", index=False)
    best = results.iloc[0]["model"]  # chosen on training CV, not on test
    best_est, best_proba = fitted[best]
    print(f"Selected on CV AUC: {best}")

    # ---- Ablations on the selected model ----
    model, grid = MODELS[best]
    ablations = [dict(results.iloc[0], variant="All features")]
    row, _, _ = fit_and_score(best, model, grid, X_train, y_train, X_test, y_test, drop=DEMOGRAPHIC_FEATURES)
    ablations.append(dict(row, variant="Without sex, ethnicity, race, religion"))
    prep_v1 = Preprocessor(merit="v1").fit(raw_train)
    row, _, _ = fit_and_score(best, model, grid, prep_v1.transform(raw_train), y_train,
                              prep_v1.transform(raw_test), y_test, drop=["merit_amount_k"])
    ablations.append(dict(row, variant="Original merit grouping (71% of codes lost)"))
    row, _, _ = fit_and_score(best, model, grid, X_train, y_train, X_test, y_test, add=LEAKY)
    ablations.append(dict(row, variant="Adding the Opt Out flag (likely leakage)"))
    abl = pd.DataFrame(ablations)[["variant", "cv_auc", "test_auc", "test_kappa", "predicted_enrollees", "actual_enrollees"]]
    abl.to_csv(out / "ablations.csv", index=False)
    print(abl.to_string(index=False))

    # ---- Which features matter (permutation importance on test, in AUC points) ----
    X_imp = X_test.drop(columns=LEAKY)
    imp = permutation_importance(best_est, X_imp, y_test, scoring="roc_auc", n_repeats=10,
                                 random_state=RANDOM_STATE, n_jobs=-1)
    pd.DataFrame({"feature": X_imp.columns, "auc_drop": imp.importances_mean, "std": imp.importances_std}) \
        .sort_values("auc_drop", ascending=False).round(4).to_csv(out / "feature_importance.csv", index=False)

    # ---- Calibration and lift by predicted-probability decile (aggregates only) ----
    dec = pd.DataFrame({"proba": best_proba, "enrolled": y_test})
    dec["decile"] = pd.qcut(dec["proba"].rank(method="first"), 10, labels=range(10, 0, -1)).astype(int)
    table = dec.groupby("decile").agg(admits=("enrolled", "size"), predicted_rate=("proba", "mean"),
                                       actual_rate=("enrolled", "mean"), enrollees=("enrolled", "sum"))
    table["share_of_all_enrollees"] = table["enrollees"] / table["enrollees"].sum()
    table.round(4).to_csv(out / "deciles.csv")

    # ---- Enrollment rates by merit tier (training set, aggregates only) ----
    tiers = pd.DataFrame({"merit_tier": X_train["merit_tier"], "enrolled": y_train}).groupby("merit_tier") \
        .agg(admits=("enrolled", "size"), enroll_rate=("enrolled", "mean")).sort_values("admits", ascending=False)
    tiers.round(4).to_csv(out / "merit_tier_rates.csv")

    summary = {"data": label, "test_admits": int(len(y_test)), "test_enroll_rate": round(float(base_rate), 4),
               "selected_model": best, "top20pct_share_of_enrollees": round(float(table.loc[[1, 2], "share_of_all_enrollees"].sum()), 4)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
