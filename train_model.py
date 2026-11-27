from __future__ import annotations
from pathlib import Path
import json, warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, precision_score, recall_score, f1_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "data" / "indian_cyclone_cleaned_engineered.csv"
MODEL_DIR = BASE / "models"
REPORT_DIR = BASE / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42

NUMERIC_A = ["lat", "lon", "wind", "pressure", "pressure_drop", "month", "hour",
             "day_of_year_sin", "day_of_year_cos"]
NUMERIC_B = [c for c in NUMERIC_A if c != "wind"]
CATEGORICAL = ["basin"]
LABEL_ORDER = ["D", "DD", "CS", "SCS", "VSCS", "ESCS", "SuCS"]
LABEL_NAMES = {
    "D": "Depression", "DD": "Deep Depression", "CS": "Cyclonic Storm",
    "SCS": "Severe Cyclonic Storm", "VSCS": "Very Severe Cyclonic Storm",
    "ESCS": "Extremely Severe Cyclonic Storm", "SuCS": "Super Cyclonic Storm"
}

def load_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}\nPlace the CSV in the project's data folder.")
    df = pd.read_csv(DATA_FILE)
    required = {"storm_id", "time", "year", "basin", "lat", "lon", "wind",
                "pressure", "pressure_drop", "month", "hour",
                "day_of_year_sin", "day_of_year_cos", "grade"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for col in NUMERIC_A + ["year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["basin"] = df["basin"].astype("string").str.upper().str.strip()
    df["grade"] = df["grade"].astype("string").str.strip()
    df = df.drop_duplicates()
    df = df.dropna(subset=["storm_id", "grade", "basin", "lat", "lon"])
    df = df[df["grade"].isin(LABEL_ORDER)]
    df = df[df["basin"].isin(["BOB", "ARB"])]
    df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)]
    if df["storm_id"].nunique() < 5:
        raise ValueError("Not enough distinct storms for group-aware evaluation.")
    return df.reset_index(drop=True)

def make_pipeline(model, numeric_features):
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", categorical_pipe, CATEGORICAL)
    ])
    return Pipeline([("preprocess", prep), ("model", model)])

def candidates():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=250, min_samples_leaf=2, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=3, class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.06, max_depth=2,
            random_state=RANDOM_STATE
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=RANDOM_STATE
        )
    }

def weighted_fit(pipe, X, y):
    # For boosting models that do not expose class_weight, use inverse-frequency
    # sample weights computed exclusively from the current training fold.
    model = pipe.named_steps["model"]
    if isinstance(model, (GradientBoostingClassifier, HistGradientBoostingClassifier)):
        counts = y.value_counts()
        weights = y.map(lambda label: len(y) / (len(counts) * counts[label])).to_numpy()
        pipe.fit(X, y, model__sample_weight=weights)
    else:
        pipe.fit(X, y)
    return pipe

def grouped_cv_score(model, X, y, groups, numeric_features):
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr, va in splitter.split(X, y, groups):
        fold_pipe = make_pipeline(clone(model), numeric_features)
        weighted_fit(fold_pipe, X.iloc[tr], y.iloc[tr])
        pred = fold_pipe.predict(X.iloc[va])
        scores.append(f1_score(y.iloc[va], pred, average="macro", zero_division=0))
    return float(np.mean(scores)), [float(s) for s in scores]

def evaluate_experiment(df, name, numeric_features, train_idx, test_idx):
    features = numeric_features + CATEGORICAL
    X = df[features]
    y = df["grade"]
    groups = df["storm_id"]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    g_train = groups.iloc[train_idx]

    rows, cv_scores = [], {}
    for model_name, estimator in candidates().items():
        cv_mean, fold_scores = grouped_cv_score(estimator, X_train, y_train, g_train, numeric_features)
        cv_scores[model_name] = cv_mean
        rows.append({"Experiment": name, "Model": model_name, "CV Macro F1": cv_mean,
                     "CV Fold Scores": json.dumps(fold_scores)})
    comparison = pd.DataFrame(rows).sort_values("CV Macro F1", ascending=False).reset_index(drop=True)
    best_name = comparison.iloc[0]["Model"]
    best = make_pipeline(clone(candidates()[best_name]), numeric_features)
    weighted_fit(best, X_train, y_train)
    pred = best.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
    }
    report_dict = classification_report(y_test, pred, labels=LABEL_ORDER,
        target_names=[LABEL_NAMES[x] for x in LABEL_ORDER], output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose().reset_index(names="Class")
    cm = confusion_matrix(y_test, pred, labels=LABEL_ORDER)
    result = {
        "name": name, "features": features, "numeric_features": numeric_features,
        "best_name": best_name, "pipeline": best, "metrics": metrics,
        "comparison": comparison, "classification_report": report_df,
        "confusion_matrix": cm, "test_actual": y_test, "test_pred": pred,
        "test_indices": test_idx
    }
    return result

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    df = load_dataset()
    # A single storm-aware, approximately stratified holdout. All records from
    # any one storm remain in exactly one partition.
    X_for_split = df[NUMERIC_A + CATEGORICAL]
    y = df["grade"]
    groups = df["storm_id"]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X_for_split, y, groups))
    train_storm_ids = set(groups.iloc[train_idx])
    test_storm_ids = set(groups.iloc[test_idx])
    if train_storm_ids.intersection(test_storm_ids):
        raise RuntimeError("Data leakage detected: a storm appears in both train and test partitions.")

    exp_a = evaluate_experiment(df, "A - With wind", NUMERIC_A, train_idx, test_idx)
    exp_b = evaluate_experiment(df, "B - Without wind", NUMERIC_B, train_idx, test_idx)

    # Select deployable current-intensity classifier using Experiment A's
    # training-only grouped CV macro-F1. The test set is used only for reporting.
    selected = exp_a
    bundle = {
        "model": selected["pipeline"],
        "best_name": selected["best_name"],
        "features": selected["features"],
        "numeric_features": selected["numeric_features"],
        "categorical_features": CATEGORICAL,
        "labels": LABEL_ORDER,
        "label_names": LABEL_NAMES,
        "experiments": {
            "with_wind": {k: selected[k] for k in ["best_name", "metrics"]},
            "without_wind": {k: exp_b[k] for k in ["best_name", "metrics"]}
        },
        "summary": {
            "observations": int(len(df)),
            "storms": int(df["storm_id"].nunique()),
            "bob_records": int((df["basin"] == "BOB").sum()),
            "arb_records": int((df["basin"] == "ARB").sum()),
            "first_year": int(df["time"].dt.year.min()),
            "last_year": int(df["time"].dt.year.max()),
            "train_records": int(len(train_idx)),
            "test_records": int(len(test_idx)),
            "train_storms": int(df.iloc[train_idx]["storm_id"].nunique()),
            "test_storms": int(df.iloc[test_idx]["storm_id"].nunique()),
            "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
            "class_distribution": {k: int(v) for k, v in df["grade"].value_counts().items()},
        },
        "defaults": {c: float(df[c].median()) for c in NUMERIC_A},
        "basins": sorted(df["basin"].dropna().unique().tolist()),
        "trained_on": "Indian North Indian Ocean dataset (BOB and ARB)",
    }
    joblib.dump(bundle, MODEL_DIR / "indian_cyclone_model.joblib")

    # Save detailed machine-readable and human-readable reports.
    comparisons = pd.concat([exp_a["comparison"], exp_b["comparison"]], ignore_index=True)
    comparisons.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    for exp, slug in [(exp_a, "with_wind"), (exp_b, "without_wind")]:
        exp["classification_report"].to_csv(REPORT_DIR / f"classification_report_{slug}.csv", index=False)
        pd.DataFrame(exp["confusion_matrix"], index=LABEL_ORDER, columns=LABEL_ORDER).to_csv(
            REPORT_DIR / f"confusion_matrix_{slug}.csv")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(exp["confusion_matrix"], cmap="Blues")
        ax.set_xticks(range(len(LABEL_ORDER)), LABEL_ORDER, rotation=35, ha="right")
        ax.set_yticks(range(len(LABEL_ORDER)), LABEL_ORDER)
        ax.set_xlabel("Predicted grade"); ax.set_ylabel("Actual grade")
        ax.set_title(f"Confusion Matrix - {exp['name']}")
        for i in range(len(LABEL_ORDER)):
            for j in range(len(LABEL_ORDER)):
                ax.text(j, i, str(exp["confusion_matrix"][i, j]), ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax); fig.tight_layout()
        fig.savefig(REPORT_DIR / f"confusion_matrix_{slug}.png", dpi=160)
        plt.close(fig)

    summary = {
        "dataset": bundle["summary"],
        "experiment_a": {"selected_model": exp_a["best_name"], **exp_a["metrics"]},
        "experiment_b": {"selected_model": exp_b["best_name"], **exp_b["metrics"]},
        "holdout_note": "Storm-grouped stratified holdout; no storm_id appears in both train and test.",
        "warning": "Experiment A classifies the observed/current intensity because wind is an input; it is not future forecasting."
    }
    (REPORT_DIR / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "INDIAN CYCLONE INTENSITY CLASSIFICATION - EVALUATION",
        "=" * 58,
        f"Observations: {len(df)} | Unique storms: {df.storm_id.nunique()}",
        f"Train: {len(train_idx)} records / {df.iloc[train_idx].storm_id.nunique()} storms",
        f"Test: {len(test_idx)} records / {df.iloc[test_idx].storm_id.nunique()} storms",
        "",
        "Experiment A - with wind (current intensity classification)",
        f"Selected model: {exp_a['best_name']}",
        json.dumps(exp_a["metrics"], indent=2),
        "",
        "Experiment B - without wind (reduced-feature comparison)",
        f"Selected model: {exp_b['best_name']}",
        json.dumps(exp_b["metrics"], indent=2),
        "",
        "Model selection is based on grouped cross-validation macro F1 on training data.",
        "The holdout test set is not used to select the model.",
        "This is not a future-intensity forecasting model."
    ]
    (REPORT_DIR / "evaluation_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\nModel comparison:")
    print(comparisons[["Experiment", "Model", "CV Macro F1"]].to_string(index=False))
    print(f"\nSaved model: {MODEL_DIR / 'indian_cyclone_model.joblib'}")

if __name__ == "__main__":
    main()
