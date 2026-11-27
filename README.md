# Indian Cyclone Intensity Prediction System

A Streamlit and machine-learning academic project for classifying the intensity grade of historical North Indian Ocean cyclone observations from the **Bay of Bengal (BOB)** and **Arabian Sea (ARB)**.

> **Important:** This project classifies the intensity associated with an observation. Experiment A uses the observed wind speed, which is closely related to the intensity grade. It is not a future cyclone-intensity forecasting model and is not an official IMD warning system.

## 1. Project structure

```text
cyclone-structural-stage-classification/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── indian_cyclone_cleaned_engineered.csv
├── models/
│   └── indian_cyclone_model.joblib     # created by training
└── reports/
    ├── evaluation_report.txt           # created by training
    ├── evaluation_summary.json         # created by training
    ├── model_comparison.csv             # created by training
    ├── classification_report_*.csv
    ├── confusion_matrix_*.csv
    └── confusion_matrix_*.png
```

The supplied CSV is already placed in the required `data/` folder. Do not rename it unless you also update `DATA_FILE` in `train_model.py` and `app.py`.

## 2. Dataset

File: `data/indian_cyclone_cleaned_engineered.csv`

The supplied dataset contains 6,764 observations and 14 columns before the training script's validation and duplicate checks.

| Column | Meaning / use |
|---|---|
| `storm_id` | Cyclone identifier; used only for group-aware splitting |
| `time` | Observation timestamp; used for audit and date coverage |
| `year` | Year; audit only |
| `basin` | BOB or ARB; categorical model feature |
| `lat`, `lon` | Geographic coordinates |
| `wind` | Observed wind speed |
| `pressure` | Atmospheric pressure |
| `pressure_drop` | Pressure-drop feature |
| `month`, `hour` | Temporal features |
| `day_of_year_sin`, `day_of_year_cos` | Cyclic day-of-year features |
| `grade` | Target class |

Target classes are retained as supplied: `D`, `DD`, `CS`, `SCS`, `VSCS`, `ESCS`, and `SuCS`.

The training script does not create synthetic rows, change labels, or use `storm_id`, `time`, or `year` as predictive features. Missing numeric inputs are imputed inside each training fold. The CSV itself is not overwritten.

## 3. Machine-learning methodology

Five classifiers are compared:

1. Random Forest
2. Extra Trees
3. Decision Tree
4. Gradient Boosting
5. HistGradientBoosting

A shared preprocessing pipeline imputes numeric missing values using the training fold's median and encodes `basin` using one-hot encoding. Tree models use balanced class weights where supported. The two boosting models use inverse-frequency sample weights calculated from the current training fold only.

### Leakage prevention

- The split is group-aware and stratified by `grade`.
- `storm_id` is used as the group identifier, not as an input feature.
- All observations belonging to a cyclone remain in one partition.
- Cross-validation also uses `StratifiedGroupKFold`.
- Preprocessing is fitted inside each pipeline and therefore only sees the corresponding training fold.
- The held-out test set is not used to select the model.

### Two feature experiments

**Experiment A – with wind:** Uses wind, pressure, pressure drop, location, basin, and engineered temporal features. This is the main current-observation intensity classifier.

**Experiment B – without wind:** Repeats model comparison without the `wind` input. This is a reduced-feature experiment to measure how much signal remains without the most directly intensity-related measurement.

The deployable model is selected from Experiment A using grouped cross-validation macro F1. Test-set results for both experiments are reported separately.

## 4. Evaluation metrics

The generated report includes:

- Accuracy
- Macro precision, recall, and F1
- Weighted F1
- Balanced accuracy
- Per-class precision, recall, F1, and support
- Confusion matrix
- Grouped cross-validation macro F1 for each candidate model

Macro F1 gives equal weight to each class, which is useful when categories are imbalanced. The rare `SuCS` category is represented by only a small number of distinct storms in this dataset, so its test metrics can be unstable. Interpret per-class results together with their support counts.

## 5. Install and run on Windows

Open Command Prompt or PowerShell in this project folder.

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

Run `python train_model.py` again whenever you replace or update the dataset. The app loads the saved model and does not retrain on every launch.

## 6. Application pages

- **Home Dashboard:** dataset counts, basin split, and intensity distribution.
- **Cyclone Prediction:** accepts basin, coordinates, wind, pressure, pressure drop, date, and hour.
- **Model Performance:** displays the two feature experiments and model comparison.
- **Dataset Analysis:** preview, missing values, basin cross-tab, wind/pressure distributions, and yearly counts.
- **Indian Cyclone Map:** plots actual recorded coordinates and differentiates BOB and ARB observations.

## 7. Limitations and responsible use

- The model is only as representative as the supplied historical dataset.
- Wind is strongly related to the target intensity category. With-wind performance should not be presented as evidence of forecasting future intensity.
- The small number of `SuCS` storms limits reliable generalization for that class.
- A storm-grouped test split measures generalization to storms not represented in training, but does not guarantee future-year forecasting performance.
- Prediction probabilities are intentionally not shown as confidence scores because probability calibration has not been validated.
- The map displays recorded observation points, not reconstructed tracks.
- This is an academic demonstration, not an operational meteorological product. Use official IMD alerts and advisories for safety decisions.

## 11. Viva-ready explanation

**What does the system predict?**  
It predicts the cyclone intensity grade (`D` through `SuCS`) for a supplied set of cyclone observation features.

**Why is `storm_id` not a feature?**  
It identifies the cyclone. If records from the same cyclone appeared in both training and testing, the evaluation could become overly optimistic. It is therefore used only to keep each cyclone together during splitting.

**Why compare with-wind and without-wind models?**  
Wind speed is closely connected to intensity grade. The with-wind experiment evaluates classification when wind is known. The without-wind experiment checks how well other features distinguish the grades.

**Does the project forecast a cyclone's future intensity?**  
No. It classifies the intensity associated with the entered observation. Future forecasting requires a separate time-sequenced dataset design, where earlier observations predict a later observation.

**Why use macro F1?**  
It gives each grade equal importance, rather than allowing frequent classes to dominate the score.

**Where are the results stored?**  
The trained pipeline is saved in `models/indian_cyclone_model.joblib`; evaluation files are saved in `reports/`.
