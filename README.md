# Cyclone Structural-Stage Classification System

**SIH 2026 | Problem Statement: SIH26070**

A machine learning project that classifies historical cyclone observations into structural intensity grades using meteorological data.

## Project Overview

The system uses historical cyclone observations to train and evaluate two machine learning algorithms:

- Decision Tree Classifier
- Random Forest Classifier

The project also provides an interactive Streamlit dashboard for dataset exploration, cyclone grade prediction, and model evaluation.

**Note:** This implementation uses meteorological observations from the Stormtrack Python package. It does not currently use HURSAT-B1 satellite imagery or IBTrACS satellite-derived features.

## Dataset

The dataset is loaded using the Stormtrack Python package:

```python
import stormtrack as st

data = st.load()
df = data.observations
```

### Dataset information

| Description | Value |
|---|---:|
| Original observations | 7,585 |
| Original features | 19 |
| Geographical basins | BOB, ARB, LAND |
| Processed observations | 7,172 |
| Unique cyclones in processed data | 387 |
| Cyclone grades | 7 |

The processed dataset includes observations from:

- BOB – Bay of Bengal
- ARB – Arabian Sea

Land observations are excluded during preprocessing.

### Model features

The model uses the following six input features:

| Feature | Description |
|---|---|
| year | Observation year |
| month | Observation month |
| lat | Cyclone latitude |
| lon | Cyclone longitude |
| pressure | Atmospheric pressure |
| basin | Geographical cyclone basin |

The target variable is `grade`.

### Cyclone grades

The dataset contains these grade labels:

- D
- DD
- CS
- SCS
- VSCS
- ESCS
- SuCS

These labels are retained from the source dataset.

## Project Structure

```text
cyclone-structural-stage-classification/
│
├── app.py
├── model.py
├── preprocess.py
├── inspect_dataset.py
├── test_dataset.py
├── requirements.txt
├── README.md
│
└── data/
    ├── raw/
    │   └── cyclone_dataset.csv
    │
    └── processed/
        ├── cyclone_dataset.csv
        ├── models.joblib
        └── results.json
```

## Machine Learning Models

Two supervised classification algorithms are implemented:

1. **Decision Tree Classifier** – A basic classification model.
2. **Random Forest Classifier** – An ensemble model used as the primary prediction model.

The data is split by cyclone ID to prevent observations from the same cyclone appearing in both training and testing sets.

| Dataset | Observations |
|---|---:|
| Training | 5,322 |
| Testing | 1,850 |
| Training cyclones | 290 |
| Testing cyclones | 97 |

No cyclone appears in both sets.

## Model Evaluation

### Random Forest results

| Metric | Score |
|---|---:|
| Accuracy | 58.0% |
| Precision (macro) | 52.4% |
| Recall (macro) | 52.9% |
| F1-score (macro) | 51.6% |

The model also generates:

- Confusion matrix
- Per-grade precision, recall and F1-score
- Prediction error analysis
- Feature importance

The results indicate moderate classification performance. Some rare grades have limited observations, which affects their individual prediction performance.

## Streamlit Dashboard

The interactive application provides three main sections:

**1. Dashboard**
- Dataset overview
- Cyclone grade distribution
- Dataset preview
- Model information

**2. Cyclone Prediction**
- Enter meteorological observation details
- Select cyclone basin
- Predict cyclone grade
- View prediction probabilities

**3. Model Evaluation**
- Decision Tree vs Random Forest comparison
- Confusion matrix
- Grade-wise performance
- Prediction error analysis

## Installation

Python 3.10–3.13 compatibility should be checked against the installed package versions.

Clone the repository:

```bash
git clone https://github.com/Sai-Ram-Varma-11/cyclone-structural-stage-classification.git
cd cyclone-structural-stage-classification
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

Run the following commands in order:

```bash
python inspect_dataset.py
python preprocess.py
python model.py
streamlit run app.py
```

The Streamlit dashboard will open in your browser.

The processed dataset and trained models are saved in `data/processed/`.

## Limitations

- The current implementation uses meteorological observations, not satellite imagery.
- Model performance is moderate and should not be interpreted as operational cyclone forecasting accuracy.
- Rare cyclone grades have relatively few samples.
- The system is an academic research prototype, not an official cyclone warning system.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Altair

## Author

**Sai Ram Varma**

GitHub: https://github.com/Sai-Ram-Varma-11

---

Academic machine learning project – SIH 2026.
