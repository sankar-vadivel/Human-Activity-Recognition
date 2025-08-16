# Human Activity Recognition using Machine Learning

Classifies human activities — **Walking, Walking Upstairs, Walking Downstairs,
Sitting, Standing, Laying** — from smartphone accelerometer and gyroscope
sensor data, using time-domain feature engineering and classical ML models.

## Overview

Raw sensor readings from a phone's accelerometer and gyroscope are collected
in fixed-width sliding windows (2.56s, 128 samples @ 50Hz — the same windowing
scheme used in the UCI HAR benchmark). From each window, 46 time-domain
statistical features are extracted (mean, standard deviation, min/max, mean
absolute deviation, signal energy, interquartile range, and cross-axis
correlation), which are then fed into classical ML classifiers.

## Results

| Model | Test Accuracy |
|---|---|
| Logistic Regression | 90.6% |
| SVM (RBF kernel) | 90.6% |
| **Random Forest** | **93.2%** |

Evaluation uses a **subject-wise train/test split** (not a random row split),
so the reported accuracy reflects how well the model generalizes to people it
has never seen — the realistic evaluation setup for HAR systems. Static
activities (Sitting/Standing/Laying) are classified almost perfectly; most
confusion occurs between the three walking variants, which is expected since
their sensor signatures are naturally more similar.

## Project Structure

```
├── notebooks/
│   └── HAR_Model.ipynb      # Full pipeline: load → features → train → evaluate
├── src/
│   └── generate_data.py     # Sensor data generation / feature extraction
├── data/
│   └── sensor_data.csv      # Generated feature dataset
├── requirements.txt
└── README.md
```

## Tech Stack

Python, NumPy, Pandas, Scikit-learn, Matplotlib, Jupyter Notebook

## Data

This repo ships with a **realistic simulated sensor dataset** (see
`src/generate_data.py`) — signals modeled after real accelerometer/gyroscope
characteristics per activity, with per-subject variation and sensor noise —
so the project runs end-to-end without any external download.

The code also supports the **real UCI HAR dataset**
([link](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones))
via `load_real_uci_har()` in `src/generate_data.py` — point it at the
extracted `UCI HAR Dataset` folder to swap in real smartphone recordings from
30 volunteers with no other code changes.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/HAR_Model.ipynb
```

## Future Improvements

- Try deep learning (1D-CNN / LSTM) directly on raw signal windows instead of
  hand-crafted features
- Real-time inference on-device
- Expand to a larger, more diverse activity set (e.g. cycling, running)
