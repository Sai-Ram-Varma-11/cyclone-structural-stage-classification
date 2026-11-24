# Cyclone Structural-Stage Classification System

Supervised machine learning project for **SIH 2026 problem statement SIH26070**.

## Objective

Classify historical tropical cyclone observations into intensity stages using
infrared satellite features (HURSAT-B1) and meteorological records (IBTrACS),
then analyse how the wind speed of each cyclone changed over time.
This is a historical classification study. It is not a forecasting system.

## Datasets (NOAA NCEI)

| Data | Source |
|------|--------|
| HURSAT-B1 v06 (satellite, NetCDF) | https://www.ncei.noaa.gov/products/hurricane-satellite-data |
| IBTrACS v04r01 (track and intensity, CSV) | https://www.ncei.noaa.gov/products/international-best-track-archive |

HURSAT-B1 v06 covers 1978 to 2015, so only storms from that period can be used.

## Files required and where to put them

1. HURSAT-B1: on the HURSAT page open "Data Access > NetCDF"
   (https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/).
   Open a year folder and download the files of 20 to 30 storms. Each storm is a
   file or folder whose name starts with the 13-character IBTrACS id, for example `2005236N23285`.
   Put the downloaded files (`.tar.gz` or `.nc`) in `data/raw/hursat/`.
2. IBTrACS: open
   https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/
   and download one CSV, for example the North Atlantic file `ibtracs.NA.list.v04r01.csv`
   or `ibtracs.since1980.list.v04r01.csv`.
   Put it in `data/raw/ibtracs/`.

Choose storms from the basin that matches your IBTrACS file. Include a mix of weak and strong storms
so that several stages have samples.

Project folders:

```
data/raw/hursat/       downloaded HURSAT-B1 storm files
data/raw/ibtracs/      downloaded IBTrACS CSV
data/processed/        created by the scripts (dataset, models, results)
```

## How satellite features are extracted

Each HURSAT-B1 NetCDF file is a 301 x 301 grid (about 8 km pixels) centred on the storm.
`preprocess.py` reads the `IRWIN` variable (infrared window brightness temperature, Kelvin) and
computes six numbers per image:

| Feature | Meaning |
|---------|---------|
| ir_min | coldest cloud-top temperature |
| ir_mean | mean temperature of the whole image |
| ir_std | spread of temperatures in the whole image |
| ir_cold_percent | percent of pixels colder than 220 K (deep clouds) |
| ir_center_mean | mean temperature of the central 51 x 51 pixels (about 200 km radius) |
| ir_center_std | spread of temperatures in the central area |

Pixels outside 150 to 350 K are treated as missing. Images with fewer than half valid pixels are skipped.
The storm id and image time are read from the file name
(`<storm id>.<name>.<year>.<month>.<day>.<hhmm>...`).
Two files for the same storm and time are reduced to the one with the smaller satellite view angle.

## Meteorological features and joining

From IBTrACS the project uses the USA agency columns (`USA_WIND`, `USA_PRES`), latitude and storm
nature. Only rows with nature `TS` (tropical) and a valid wind are kept. Satellite rows are joined to
IBTrACS rows by storm id and the nearest time within 90 minutes. Missing pressure is kept as missing
and filled with the training-set median inside the model pipeline, so no test information is used.

Model inputs (8): the six satellite features, central pressure, absolute latitude.

Wind speed is **not** a model input. The stage label is defined from wind speed, so using wind as an
input would let the model read the answer from the label rule.

## How labels are prepared

Label = Saffir-Simpson stage from the USA agency 1-minute sustained wind (`USA_WIND`, knots),
the same agency definition for every row.

| Stage | Wind (kt) |
|-------|-----------|
| Tropical Depression | below 34 |
| Tropical Storm | 34 - 63 |
| Category 1 | 64 - 82 |
| Category 2 | 83 - 95 |
| Category 3 | 96 - 112 |
| Category 4 | 113 - 136 |
| Category 5 | 137 and above |

## Algorithms

- Decision Tree Classifier (basic model, depth limited to 5 so it can be explained).
- Random Forest Classifier (main model, 200 trees).

Both use class weights to balance rare stages. The data is split 75% / 25% **by storm id** so no storm
appears in both training and testing sets.

## Evaluation

Accuracy, precision, recall, F1-score (macro average, each stage counts equally), per-class precision,
recall and F1-score, and a confusion matrix. Error analysis counts how many mistakes are between
neighbouring stages (for example Category 2 predicted as Category 3) and lists the most common confusions.

## Intensity-change extension

For every observation the wind speed is compared with the wind speed of the same cyclone 6 hours earlier.

- change of +5 kt or more: Strengthening
- change of -5 kt or less: Weakening
- anything in between: Relatively stable

5 kt is the reporting step of best-track winds, and 6 hours is the standard best-track interval.
This describes past behaviour only. It is not a forecast.

## Install

Python 3.10 to 3.12 is recommended.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python inspect_hursat.py
python preprocess.py
python model.py
streamlit run app.py
```

`inspect_hursat.py` prints the variables of the first HURSAT file. Use it to confirm the data is read correctly.
`preprocess.py` creates `data/processed/cyclone_dataset.csv`.
`model.py` trains both models and creates `models.joblib` and `results.json` in `data/processed/`.

## Limitations

- IRWIN is used as decoded by the NetCDF library. The optional calibration attributes stored in the files are not applied.
- HURSAT v06 was generated with an older IBTrACS version. A storm whose id differs in IBTrACS v04 is reported by `preprocess.py` and dropped.
- Small storm subsets give small test sets. Rare stages (Category 4 and 5) may have few or no test samples, so their metrics are unreliable.
- Pressure is available only where the USA agency reported it.
