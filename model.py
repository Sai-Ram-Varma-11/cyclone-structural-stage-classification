
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from preprocess import DATASET_FILE, FEATURES, PROCESSED_DIR, TARGET

MODEL_FILE = PROCESSED_DIR / "models.joblib"
RESULTS_FILE = PROCESSED_DIR / "results.json"

GRADE_ORDER = ["D", "DD", "CS", "SCS", "VSCS", "ESCS", "SuCS"]


def load_data():
    if not DATASET_FILE.exists():
        raise SystemExit("Dataset not found. Run: python preprocess.py")

    data = pd.read_csv(DATASET_FILE)

    required = ["storm_id"] + FEATURES + [TARGET]
    missing = [column for column in required if column not in data.columns]

    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    return data.dropna(subset=["storm_id", TARGET]).copy()


def split_by_storm(data, labels):
    splitter = GroupShuffleSplit(
        n_splits=50,
        test_size=0.25,
        random_state=42
    )

    best_split = None
    best_score = -1

    for train_index, test_index in splitter.split(
        data, groups=data["storm_id"]
    ):
        train = data.iloc[train_index]
        test = data.iloc[test_index]

        if not set(labels).issubset(set(train[TARGET])):
            continue

        score = len(set(labels).intersection(set(test[TARGET])))

        if score > best_score:
            best_split = (train, test)
            best_score = score

    if best_split is None:
        raise SystemExit(
            "Could not create a training split containing every grade."
        )

    train, test = best_split

    if not set(train["storm_id"]).isdisjoint(set(test["storm_id"])):
        raise SystemExit("Storm leakage detected.")

    return train, test


def build_preprocessor():
    numeric_features = [
        feature for feature in FEATURES
        if feature != "basin"
    ]

    categorical_features = ["basin"]

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_features
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]),
                categorical_features
            )
        ]
    )


def build_models():
    models = {}

    for name, classifier in {
        "decision_tree": DecisionTreeClassifier(
            max_depth=5,
            class_weight="balanced",
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }.items():
        models[name] = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier)
        ])

    return models


def analyse_errors(actual, predicted, labels):
    grade_index = {
        grade: index for index, grade in enumerate(labels)
    }

    gaps = [
        abs(grade_index[a] - grade_index[p])
        for a, p in zip(actual, predicted)
    ]

    pairs = [
        f"{a} -> {p}"
        for a, p in zip(actual, predicted)
        if a != p
    ]

    return {
        "errors": int(sum(gap > 0 for gap in gaps)),
        "neighbouring_errors": int(sum(gap == 1 for gap in gaps)),
        "distant_errors": int(sum(gap > 1 for gap in gaps)),
        "top_confusions": pd.Series(
            pairs, dtype=str
        ).value_counts().head(3).to_dict()
    }


def evaluate(model, test, labels):
    actual = test[TARGET].to_numpy()
    predicted = model.predict(test[FEATURES])

    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        labels=labels,
        average="macro",
        zero_division=0
    )

    class_precision, class_recall, class_f1, support = (
        precision_recall_fscore_support(
            actual,
            predicted,
            labels=labels,
            zero_division=0
        )
    )

    per_class = pd.DataFrame({
        "grade": labels,
        "precision": class_precision,
        "recall": class_recall,
        "f1": class_f1,
        "support": support
    })

    return {
        "accuracy": float((actual == predicted).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "per_class": per_class.to_dict(orient="records"),
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=labels
        ).tolist(),
        "error_analysis": analyse_errors(
            actual, predicted, labels
        )
    }


def main():
    data = load_data()

    labels = [
        grade for grade in GRADE_ORDER
        if grade in set(data[TARGET])
    ]

    train, test = split_by_storm(data, labels)

    print("DATASET INFORMATION")
    print("Observations:", len(data))
    print("Unique cyclones:", data["storm_id"].nunique())
    print("Training observations:", len(train))
    print("Testing observations:", len(test))
    print("Training cyclones:", train["storm_id"].nunique())
    print("Testing cyclones:", test["storm_id"].nunique())
    print("No cyclone appears in both sets.")

    models = build_models()
    results = {}

    for name, model in models.items():
        model.fit(train[FEATURES], train[TARGET])
        results[name] = evaluate(model, test, labels)

        result = results[name]

        print(f"\n{name.upper()}")
        print(f"Accuracy: {result['accuracy']:.3f}")
        print(f"Precision: {result['precision']:.3f}")
        print(f"Recall: {result['recall']:.3f}")
        print(f"F1-score: {result['f1']:.3f}")

        print("\nPER-GRADE RESULTS")
        print(
            pd.DataFrame(result["per_class"])
            .round(3)
            .to_string(index=False)
        )

        print("\nERROR ANALYSIS")
        print(result["error_analysis"])

    forest = models["random_forest"]

    feature_names = (
        forest.named_steps["preprocessor"].get_feature_names_out()
    )

    importance = pd.Series(
        forest.named_steps["classifier"].feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    print("\nRANDOM FOREST FEATURE IMPORTANCE")
    print(importance.round(3).to_string())

    feature_ranges = {}

    for feature in FEATURES:
        if feature == "basin":
            feature_ranges[feature] = sorted(
                data[feature].dropna().unique().tolist()
            )
        else:
            feature_ranges[feature] = {
                "min": float(data[feature].min()),
                "median": float(data[feature].median()),
                "max": float(data[feature].max())
            }

    summary = {
        "n_observations": int(len(data)),
        "n_storms": int(data["storm_id"].nunique()),
        "n_train_observations": int(len(train)),
        "n_test_observations": int(len(test)),
        "grades": labels,
        "n_grades": len(labels),
        "features": FEATURES,
        "selected_model": "Random Forest",
        "feature_ranges": feature_ranges,
        "decision_tree": results["decision_tree"],
        "random_forest": results["random_forest"]
    }

    joblib.dump(models, MODEL_FILE)
    RESULTS_FILE.write_text(json.dumps(summary, indent=2))

    print("\nMODELS SAVED:", MODEL_FILE)
    print("RESULTS SAVED:", RESULTS_FILE)


if __name__ == "__main__":
    main()
