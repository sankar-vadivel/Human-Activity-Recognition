"""
Simulated smartphone accelerometer/gyroscope sensor data generator.

Mimics the structure of the UCI "Human Activity Recognition Using Smartphones"
dataset: 6 activities, 30 subjects, fixed-width sliding windows of 128 readings
(2.56s @ 50Hz), with time-domain statistical features extracted per window.

NOTE: This generates SIMULATED signals for local development/testing when the
real UCI HAR dataset isn't reachable. To use the real dataset instead, download it from:
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
and point load_real_uci_har() at the extracted folder (see bottom of file).
"""

import numpy as np
import pandas as pd

ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
              "SITTING", "STANDING", "LAYING"]
N_SUBJECTS = 30
WINDOWS_PER_SUBJECT_ACTIVITY = 12   # keeps dataset a manageable size
SAMPLES_PER_WINDOW = 128            # 2.56s @ 50Hz, matches real dataset
AXES = ["x", "y", "z"]

# Rough per-activity signal characteristics (mean gravity offset, motion amplitude, frequency)
ACTIVITY_PARAMS = {
    "WALKING":            dict(base=(0.0, 1.0, 0.1), amp=0.6, freq=1.8, noise=0.08),
    "WALKING_UPSTAIRS":   dict(base=(0.05, 1.05, 0.15), amp=0.75, freq=1.6, noise=0.10),
    "WALKING_DOWNSTAIRS": dict(base=(-0.05, 0.95, 0.1), amp=0.85, freq=2.0, noise=0.12),
    "SITTING":            dict(base=(0.0, 0.05, 1.0), amp=0.03, freq=0.3, noise=0.02),
    "STANDING":           dict(base=(0.0, 0.02, 1.0), amp=0.02, freq=0.2, noise=0.015),
    "LAYING":             dict(base=(0.98, 0.02, 0.02), amp=0.02, freq=0.15, noise=0.015),
}


def _simulate_window(activity, rng, subject_bias):
    params = ACTIVITY_PARAMS[activity]
    t = np.linspace(0, SAMPLES_PER_WINDOW / 50.0, SAMPLES_PER_WINDOW)
    # per-subject variation: everyone carries/walks slightly differently
    amp_mult = subject_bias["amp_mult"]
    freq_mult = subject_bias["freq_mult"]
    offset = subject_bias["offset"]
    extra_noise = subject_bias["noise_mult"]

    signals = {}
    for i, axis in enumerate(AXES):
        base = params["base"][i] + offset[i]
        phase = rng.uniform(0, 2 * np.pi)
        amp = params["amp"] * amp_mult
        freq = params["freq"] * freq_mult
        wobble = 0.35 * np.sin(2.3 * freq * t + phase * 1.7 + rng.normal(0, 0.5))
        signal = (base
                  + amp * np.sin(2 * np.pi * freq * t + phase)
                  + amp * wobble
                  + rng.normal(0, params["noise"] * 2.2 * extra_noise, SAMPLES_PER_WINDOW))
        signals[f"acc_{axis}"] = signal
    for i, axis in enumerate(AXES):
        gyro = (amp * 40 * np.cos(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
                + rng.normal(0, params["noise"] * 22 * extra_noise, SAMPLES_PER_WINDOW))
        signals[f"gyro_{axis}"] = gyro
    return signals


def _extract_features(signals):
    feats = {}
    for name, sig in signals.items():
        feats[f"{name}_mean"] = np.mean(sig)
        feats[f"{name}_std"] = np.std(sig)
        feats[f"{name}_min"] = np.min(sig)
        feats[f"{name}_max"] = np.max(sig)
        feats[f"{name}_mad"] = np.mean(np.abs(sig - np.mean(sig)))
        feats[f"{name}_energy"] = np.mean(sig ** 2)
        feats[f"{name}_iqr"] = np.percentile(sig, 75) - np.percentile(sig, 25)
    # cross-axis correlations (very informative for distinguishing activities)
    for a1, a2 in [("acc_x", "acc_y"), ("acc_x", "acc_z"), ("acc_y", "acc_z")]:
        feats[f"corr_{a1}_{a2}"] = np.corrcoef(signals[a1], signals[a2])[0, 1]
    return feats


def generate_dataset(seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for subject in range(1, N_SUBJECTS + 1):
        subject_bias = dict(
            amp_mult=rng.uniform(0.8, 1.25),
            freq_mult=rng.uniform(0.85, 1.15),
            offset=rng.normal(0, 0.04, 3),
            noise_mult=rng.uniform(0.85, 1.3),
        )
        for activity in ACTIVITIES:
            for _ in range(WINDOWS_PER_SUBJECT_ACTIVITY):
                signals = _simulate_window(activity, rng, subject_bias)
                feats = _extract_features(signals)
                feats["subject"] = subject
                feats["activity"] = activity
                rows.append(feats)
    df = pd.DataFrame(rows)
    cols = ["subject", "activity"] + [c for c in df.columns if c not in ("subject", "activity")]
    return df[cols]


def load_real_uci_har(dataset_dir):
    """
    Load the REAL UCI HAR dataset once downloaded and extracted.
    dataset_dir should point to the extracted 'UCI HAR Dataset' folder.
    Download: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
    """
    import os
    features = pd.read_csv(os.path.join(dataset_dir, "features.txt"),
                            sep=r"\s+", header=None, names=["idx", "feature"])
    activity_labels = pd.read_csv(os.path.join(dataset_dir, "activity_labels.txt"),
                                   sep=r"\s+", header=None, names=["id", "activity"])

    def _load_split(split):
        X = pd.read_csv(os.path.join(dataset_dir, split, f"X_{split}.txt"),
                         sep=r"\s+", header=None, names=features["feature"])
        y = pd.read_csv(os.path.join(dataset_dir, split, f"y_{split}.txt"),
                         sep=r"\s+", header=None, names=["activity_id"])
        subj = pd.read_csv(os.path.join(dataset_dir, split, f"subject_{split}.txt"),
                            sep=r"\s+", header=None, names=["subject"])
        df = pd.concat([subj, X], axis=1)
        df["activity"] = y["activity_id"].map(activity_labels.set_index("id")["activity"])
        return df

    train_df = _load_split("train")
    test_df = _load_split("test")
    return pd.concat([train_df, test_df], ignore_index=True)


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/sensor_data.csv", index=False)
    print(f"Generated {len(df)} rows, {df.shape[1]} columns")
    print(df["activity"].value_counts())
