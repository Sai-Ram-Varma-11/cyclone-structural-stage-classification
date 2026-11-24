
import json

import joblib
import pandas as pd
import streamlit as st

from model import MODEL_FILE, RESULTS_FILE
from preprocess import DATASET_FILE, FEATURES

st.set_page_config(
    page_title="Cyclone Structural-Stage Classification",
    page_icon="🌪️",
    layout="wide"
)


@st.cache_data
def load_data():
    return pd.read_csv(DATASET_FILE)


@st.cache_data
def load_results():
    with open(RESULTS_FILE, "r") as file:
        return json.load(file)


@st.cache_resource
def load_models():
    return joblib.load(MODEL_FILE)


if not all(path.exists() for path in [DATASET_FILE, MODEL_FILE, RESULTS_FILE]):
    st.error("Required files are missing. Run preprocess.py and model.py first.")
    st.stop()

data = load_data()
results = load_results()
models = load_models()

forest = results["random_forest"]

st.title("Cyclone Structural-Stage Classification")
st.caption(
    "SIH26070 | Machine learning classification using historical "
    "meteorological cyclone observations"
)

overview, prediction, evaluation = st.tabs(
    ["Dashboard", "Cyclone Prediction", "Model Evaluation"]
)

# ---------------- DASHBOARD ----------------

with overview:
    st.subheader("Project Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Observations", f"{results['n_observations']:,}")
    c2.metric("Cyclones", f"{results['n_storms']:,}")
    c3.metric("Random Forest Accuracy", f"{forest['accuracy']:.1%}")
    c4.metric("Grades", len(results["grades"]))

    st.write(
        f"Training observations: {results['n_train_observations']:,} | "
        f"Testing observations: {results['n_test_observations']:,}"
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Cyclone Grade Distribution")
        grade_counts = data["grade"].value_counts().reindex(
            results["grades"], fill_value=0
        )
        st.bar_chart(grade_counts, x_label="Cyclone Grade",
                     y_label="Observations")

    with right:
        st.subheader("Dataset Information")
        st.write("**Geographical basins:**")
        st.write("Bay of Bengal (BOB)")
        st.write("Arabian Sea (ARB)")

        st.write("**Model input features:**")
        for feature in FEATURES:
            st.write(f"- {feature}")

    st.divider()
    st.subheader("Dataset Preview")
    st.dataframe(data.head(10), width="stretch")

# ---------------- PREDICTION ----------------

with prediction:
    st.subheader("Predict Cyclone Grade")
    st.write("Enter the meteorological observation details below.")

    ranges = results["feature_ranges"]

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            year = st.number_input(
                "Year", min_value=1900, max_value=2100,
                value=int(ranges["year"]["median"])
            )

            month = st.number_input(
                "Month", min_value=1, max_value=12,
                value=int(ranges["month"]["median"])
            )

            lat = st.number_input(
                "Latitude",
                min_value=-90.0, max_value=90.0,
                value=float(ranges["lat"]["median"]),
                format="%.2f"
            )

        with col2:
            lon = st.number_input(
                "Longitude",
                min_value=-180.0, max_value=180.0,
                value=float(ranges["lon"]["median"]),
                format="%.2f"
            )

            pressure = st.number_input(
                "Atmospheric Pressure",
                min_value=800.0, max_value=1100.0,
                value=float(ranges["pressure"]["median"]),
                format="%.1f"
            )

            basin = st.selectbox(
                "Cyclone Basin",
                ["BOB", "ARB"]
            )

        submitted = st.form_submit_button(
            "Predict Cyclone Grade",
            type="primary",
            width="stretch"
        )

    if submitted:
        input_data = pd.DataFrame([{
            "year": year,
            "month": month,
            "lat": lat,
            "lon": lon,
            "pressure": pressure,
            "basin": basin
        }])[FEATURES]

        model = models["random_forest"]
        predicted_grade = model.predict(input_data)[0]

        probabilities = pd.Series(
            model.predict_proba(input_data)[0],
            index=model.classes_
        ).sort_values(ascending=False)

        st.divider()
        st.subheader("Prediction Result")

        c1, c2 = st.columns(2)
        c1.metric("Predicted Cyclone Grade", predicted_grade)
        c2.metric(
            "Model Confidence",
            f"{probabilities[predicted_grade]:.1%}"
        )

        st.write("**Prediction probabilities**")
        st.bar_chart(probabilities, x_label="Grade",
                     y_label="Probability")

        st.caption(
            "This is a research prototype using historical observations. "
            "It is not an operational cyclone warning or forecasting system."
        )

# ---------------- MODEL EVALUATION ----------------

with evaluation:
    st.subheader("Model Performance Comparison")

    comparison = pd.DataFrame({
        "Decision Tree": [
            results["decision_tree"]["accuracy"],
            results["decision_tree"]["precision"],
            results["decision_tree"]["recall"],
            results["decision_tree"]["f1"]
        ],
        "Random Forest": [
            forest["accuracy"],
            forest["precision"],
            forest["recall"],
            forest["f1"]
        ]
    }, index=["Accuracy", "Precision", "Recall", "F1 Score"])

    st.dataframe(
        (comparison * 100).round(2).astype(str) + "%",
        width="stretch"
    )

    st.bar_chart(comparison, y_label="Score")

    st.divider()
    st.subheader("Random Forest Confusion Matrix")

    if "confusion_matrix" in forest:
        labels = results["grades"]
        matrix = pd.DataFrame(
            forest["confusion_matrix"],
            index=labels,
            columns=labels
        )
        st.dataframe(matrix, width="stretch")

    if "per_class" in forest:
        st.subheader("Grade-wise Performance")
        per_class = pd.DataFrame(forest["per_class"])

        if "stage" in per_class.columns:
            per_class = per_class.set_index("stage")

        st.dataframe(per_class.round(3), width="stretch")

    errors = forest.get("error_analysis", {})

    if errors:
        st.divider()
        st.subheader("Prediction Error Analysis")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Errors", errors.get("errors", 0))
        c2.metric(
            "Neighbouring Errors",
            errors.get("neighbouring_errors", 0)
        )
        c3.metric(
            "Distant Errors",
            errors.get("distant_errors", 0)
        )

st.divider()
st.caption(
    "Cyclone Structural-Stage Classification | Academic Research Project"
)
