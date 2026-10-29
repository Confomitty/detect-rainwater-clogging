# Waterlogging Risk Prediction — Silk Board, Bengaluru

A machine-learning pipeline that predicts which road segments near Silk Board
Junction are at risk of waterlogging during a rainfall event.

> **Status:** Pipeline is built and tested. Waiting for real labelled data
> before genuine model training can begin. See [What You Need First](#what-you-need-first).

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Setup](#setup)
3. [What You Need First](#what-you-need-first)
4. [Expected Data Format](#expected-data-format)
5. [Running the Pipeline](#running-the-pipeline)
6. [Running Tests](#running-tests)
7. [Making Predictions](#making-predictions)
8. [Limitations](#limitations)
9. [Next Steps](#next-steps)

---

## Project Structure

```
waterlogging_ml/
├── data/
│   ├── raw/            ← Drop your teammate's CSV files here
│   └── processed/      ← Train/test splits written here (gitignored)
├── models/             ← Saved .pkl model files land here
├── notebooks/          ← For data exploration (Jupyter)
├── src/
│   ├── schema.py       ← Column definitions and allowed values
│   ├── validate.py     ← Checks data quality before training
│   ├── preprocess.py   ← Cleans, encodes, and scales features
│   ├── train.py        ← Logistic Regression + Random Forest
│   ├── evaluate.py     ← Precision, recall, F1, confusion matrix
│   └── predict.py      ← Load saved model → predict new roads
├── tests/
│   ├── conftest.py     ← Synthetic data fixtures (for testing only)
│   ├── test_validate.py
│   ├── test_preprocess.py
│   └── test_train.py
├── run_pipeline.py     ← Single entry point for the full pipeline
└── requirements.txt
```

---

## Setup

**Requirements:** Python 3.13, virtual environment already created.

```bash
# Activate the virtual environment (Windows)
.venv\Scripts\activate

# Verify packages are installed
python -c "import sklearn, pandas, numpy; print('All good!')"
```

No new packages need to be installed — everything is already in `.venv`.

---

## What You Need First

> **Supervised machine learning cannot begin without labelled data.**

Before running `run_pipeline.py` on real data, your team must provide:

| Item | Description |
|---|---|
| **Road segments** | A list of road segments near Silk Board with unique IDs |
| **Rainfall events** | Historical rainfall events with dates, intensity (mm/hr), and 24h totals |
| **Elevation & slope** | From SRTM DEM or Survey of India data (1-arc-second resolution works) |
| **Land cover** | From satellite imagery (LULC classification) or BBMP records |
| **Drainage density** | Length of drains per unit area — from BBMP stormwater drain GIS data |
| **Impervious surface %** | From satellite imagery or BBMP data |
| **Waterlogging labels** | `1` = waterlogged, `0` = not waterlogged — **this is the most critical item** |

### How to get waterlogging labels

Labels must come from one of:
- BBMP / BWSSB official flood complaint records (linked to road segments and dates)
- Field surveys during monsoon events
- Geotagged social media posts + human verification (e.g. Twitter/X flood reports)
- Traffic sensor data showing speed drops correlated with rainfall events

**Do not invent or guess labels.** A model trained on wrong labels will make wrong predictions.

---

## Expected Data Format

Your teammate should deliver a CSV file with these columns:

### Required columns

| Column | Type | Example | Description |
|---|---|---|---|
| `road_id` | string/int | `"SB_001"` | Unique road segment ID |
| `event_id` | string | `"EVT_2023_06_15"` | Unique rainfall event ID |
| `latitude` | float | `12.9174` | Segment centroid latitude (WGS-84) |
| `longitude` | float | `77.6219` | Segment centroid longitude (WGS-84) |
| `elevation_m` | float | `912.3` | Elevation in metres |
| `slope_pct` | float | `1.5` | Road slope in percent |
| `rainfall_mm_hr` | float | `35.2` | Rainfall intensity during event (mm/hr) |
| `rainfall_mm_24h` | float | `82.0` | Total rainfall in prior 24 h (mm) |
| `drain_density` | float | `0.45` | Drainage length / area (m/m²) |
| `imperv_pct` | float | `78.5` | Impervious surface fraction (0–100) |
| `land_cover` | string | `"commercial"` | One of: `residential`, `commercial`, `road`, `green`, `water`, `industrial`, `mixed` |
| `road_type` | string | `"arterial"` | One of: `arterial`, `collector`, `local`, `highway` |
| `waterlogged` | int | `1` | **Target** — `1` = flooded, `0` = not flooded |

### Optional columns (improve accuracy if provided)

| Column | Type | Description |
|---|---|---|
| `road_width_m` | float | Road width in metres |
| `speed_limit_kmh` | float | Posted speed limit |
| `last_cleaned_days` | int | Days since last drain cleaning |

### Example rows

```csv
road_id,event_id,latitude,longitude,elevation_m,slope_pct,rainfall_mm_hr,rainfall_mm_24h,drain_density,imperv_pct,land_cover,road_type,waterlogged
SB_001,EVT_2023_06_15,12.9174,77.6219,912.0,1.2,38.5,95.0,0.42,85.0,commercial,arterial,1
SB_002,EVT_2023_06_15,12.9201,77.6244,908.5,0.8,38.5,95.0,0.65,72.0,residential,collector,0
SB_001,EVT_2023_08_22,12.9174,77.6219,912.0,1.2,12.0,25.0,0.42,85.0,commercial,arterial,0
```

Notice that `SB_001` appears in two events — this is expected and handled correctly by the group-aware train/test split.

---

## Running the Pipeline

Once your teammate delivers the CSV:

```bash
# Activate virtual environment
.venv\Scripts\activate

# Run the full pipeline
python run_pipeline.py --data data/raw/silk_board_data.csv

# With options
python run_pipeline.py \
    --data data/raw/silk_board_data.csv \
    --test-size 0.2 \
    --models-dir models/ \
    --save-processed
```

The pipeline will:
1. ✅ Validate the CSV and report any problems
2. ✅ Split data by `event_id` (prevents data leakage between rainfall events)
3. ✅ Preprocess: impute missing values, scale numbers, encode categories
4. ✅ Train Logistic Regression and Random Forest
5. ✅ Print precision, recall, F1, and confusion matrix for each model
6. ✅ Save `.pkl` files to `models/`

### Validate data only (no training)

```bash
python -m src.validate data/raw/silk_board_data.csv
```

---

## Running Tests

The tests use **synthetic data only** to verify the pipeline code works.
They do not test model accuracy on real data.

```bash
.venv\Scripts\activate
python -m pytest tests/ -v
```

Expected output: all tests pass with green ✓ marks.

---

## Making Predictions

After training, predict on new road data (without `waterlogged` column):

```bash
# From the command line — coming in Streamlit version
```

From Python:

```python
from src.predict import predict_from_csv, predict_single_road

# Predict for many roads from a CSV
results = predict_from_csv(
    model_path="models/random_forest.pkl",
    data_path="data/raw/new_roads.csv",
)
print(results[["road_id", "predicted_waterlogged", "prob_waterlogged"]])

# Predict for a single road (useful for a Streamlit form)
result = predict_single_road(
    model_path="models/random_forest.pkl",
    road_data={
        "road_id": "SB_099",
        "event_id": "EVT_FORECAST",
        "latitude": 12.917,
        "longitude": 77.622,
        "elevation_m": 912.0,
        "slope_pct": 1.2,
        "rainfall_mm_hr": 35.0,
        "rainfall_mm_24h": 80.0,
        "drain_density": 0.45,
        "imperv_pct": 78.0,
        "land_cover": "commercial",
        "road_type": "arterial",
    }
)
print(result)
# → {"predicted_waterlogged": 1, "prob_waterlogged": 0.82, "risk_label": "HIGH RISK"}
```

---

## Limitations

| Limitation | Detail |
|---|---|
| **No real data yet** | All metrics shown in tests are from synthetic random data — meaningless for real predictions |
| **Binary classification only** | Predicts waterlogged / not waterlogged, not flood depth or duration |
| **Static snapshot** | Does not model time-series progression of a flood event |
| **No spatial autocorrelation** | Roads close to each other are treated independently |
| **Requires historical labels** | Cannot be used for real-time prediction without historical flood records for training |
| **No hyperparameter tuning** | Default parameters used; add GridSearchCV after getting ≥200 labelled rows |
| **Silk Board region only** | Coordinate bounds and elevation ranges are set for this area |

---

## Next Steps

1. **Get labelled data** — your teammate's CSV with the `waterlogged` column filled in
2. **Run validation** — `python -m src.validate data/raw/yourfile.csv`
3. **Run the pipeline** — `python run_pipeline.py --data data/raw/yourfile.csv`
4. **Interpret results** — focus on recall for class=1 (catching real floods)
5. **Tune models** — add GridSearchCV when you have ≥200 rows
6. **Build Streamlit app** — call `predict_single_road()` from a web form
7. **(Advanced)** Add SUMO traffic simulation integration for impact modelling

---

*Built for the Silk Board waterlogging study — Bengaluru, 2026.*

