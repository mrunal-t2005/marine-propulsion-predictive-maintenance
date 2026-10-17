# MarinePropulsionAI ⚓

**Machine Learning for Marine Engine Fault Classification and Anomaly Detection**

MarinePropulsionAI is a machine learning project that analyzes marine propulsion sensor data to identify known fault categories and flag potentially anomalous operating conditions. It combines supervised fault classification with unsupervised anomaly detection, model evaluation, explainability, and a reusable inference pipeline.

> **Project status:** Research and development prototype. The current results are based on the available dataset and have not established real-world vessel deployment or operational reliability.

## Overview

Marine propulsion systems generate sensor readings that can help identify abnormal operating conditions. This project explores how machine learning can use those readings to support condition monitoring and fault analysis.

The project includes two complementary approaches:

- **Fault classification:** Predicts a fault category using labeled sensor data.
- **Anomaly detection:** Assigns anomaly scores to sensor observations and flags them using a selected threshold.

These approaches solve different problems. Classification predicts among known labels, while anomaly detection identifies observations that differ from learned operating patterns.

## Key Features

- Sensor data inspection and data-quality analysis.
- Exploratory data analysis and feature-separation analysis.
- Baseline Random Forest and XGBoost fault-classification experiments.
- Class-weight tuning to investigate performance on less frequent fault categories.
- Isolation Forest-based anomaly detection.
- Chronological training, validation, and test workflows.
- Precision-recall analysis and threshold selection.
- SHAP-based feature-importance analysis for model interpretation.
- Saved anomaly-detection pipeline and metadata for repeatable inference.
- Automated tests for input validation, feature ordering, threshold logic, and model artifact loading.

## Technology Stack

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn, XGBoost
- **Model explainability:** SHAP
- **Visualization:** Matplotlib, Seaborn
- **Model persistence:** Joblib
- **Testing:** Pytest

## Project Structure

```text
MarinePropulsionAI/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── analyze_data.py
│   ├── analyze_feature_separation.py
│   ├── check_data_quality.py
│   ├── check_temporal_stability.py
│   ├── detect_anomalies.py
│   ├── evaluate_anomaly_test.py
│   ├── evaluate_anomaly_thresholds.py
│   ├── evaluate_final_model.py
│   ├── explain_model.py
│   ├── inspect_data.py
│   ├── investigate_labels.py
│   ├── predict_anomalies.py
│   ├── train_and_save_anomaly_model.py
│   ├── train_baseline.py
│   ├── train_xgboost.py
│   └── tune_fault_thresholds.py
├── tests/
│   └── test_anomaly_inference.py
├── requirements.txt
└── README.md
```

Generated reports and trained model artifacts are created locally. Some directories may be empty in a fresh checkout until the relevant scripts are run.

## Dataset

The development dataset is expected at:

`data/raw/marine_propulsion.csv`

The working dataset contains **10,000 rows**, a timestamp column, 18 numeric sensor features, and a `Fault_Label` column used in supervised experiments.

The sensor features represent measurements such as air pressure, oil pressure, oil temperature, fuel flow, cylinder measurements, and vibration along multiple axes.

The dataset is processed chronologically for model evaluation rather than randomly splitting individual rows.

**Data provenance:** The origin and representativeness of the dataset have not been independently verified. Results should therefore be interpreted as experimental findings on the available data, not as proof of performance on operational marine engines.

The raw dataset is excluded from Git. Obtain the authorized dataset separately and place it at the expected path before running the scripts.

## Methodology

### 1. Data analysis

The analysis scripts inspect data quality, label distributions, feature separation, and temporal stability. Visualizations and reports help investigate relationships between sensor readings and labeled operating conditions.

### 2. Fault classification

Random Forest and XGBoost models are evaluated for predicting known fault categories. Experiments include class weighting and threshold tuning to investigate imbalanced-class performance.

### 3. Anomaly detection

An Isolation Forest pipeline is used to calculate anomaly scores from sensor features. Preprocessing includes median imputation and feature scaling.

A threshold is selected using a chronological validation segment and precision-recall analysis. The saved inference pipeline reuses this threshold when flagging new observations.

### 4. Explainability

SHAP-based analysis is used to investigate which features influence the trained classifier's predictions. Feature importance describes model behavior and should not be interpreted as proof of physical causation.

### 5. Automated testing

Pytest tests verify inference behavior independently of the full project dataset by constructing a small synthetic dataset and fitted model.

## Experimental Results

### Fault classification

The final reported chronological test evaluation for the tuned XGBoost classifier produced:

| Metric | Result |
|---|---:|
| Accuracy | 81.07% |
| Balanced accuracy | 80.80% |
| Macro F1-score | 0.7899 |
| Weighted F1-score | 0.8268 |

Performance varies substantially by fault category. Some categories have high scores, while the fuel-injection fault category has a lower F1-score. Aggregate metrics should not hide these class-level differences.

### Anomaly detection

The saved anomaly-detection pipeline uses a validation-selected threshold of approximately `-0.063393` on its anomaly score.

The validation experiment reported:

| Metric | Result |
|---|---:|
| Average precision | 0.784 |
| Precision at selected threshold | 0.628 |
| Recall at selected threshold | 0.741 |
| F1-score at selected threshold | 0.680 |

A separate final-test experiment reported precision of approximately 0.624, recall of 0.752, and F1-score of 0.682. However, the threshold was selected using a model fitted on the earlier training segment and then applied to a separately refitted model. This mismatch is a methodological limitation, so the final-test figures should be treated as preliminary rather than a definitive estimate of generalization.

The inference smoke test scored all 10,000 dataset rows and flagged 4,019 rows (40.19%) with the saved pipeline. This is not an independent test-set result and does not mean that 40.19% of real marine-engine observations are faulty.

## Getting Started

### Prerequisites

- Python 3.13 or a compatible Python version supported by the project dependencies.
- Git.
- The authorized dataset file at the expected location.

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd MarinePropulsionAI
```

Replace the placeholder with your repository's actual GitHub URL.

### 2. Create and activate a virtual environment

On Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the dataset

Place the dataset at:

```text
data/raw/marine_propulsion.csv
```

### 5. Train and save the anomaly-detection pipeline

```cmd
python src\train_and_save_anomaly_model.py
```

This script creates the local model and metadata artifacts required for inference.

### 6. Run anomaly inference

```cmd
python src\predict_anomalies.py
```

The script loads the saved artifacts, calculates anomaly scores, applies the saved threshold, and writes:

```text
reports/anomaly_inference_demo.csv
```

The output includes the timestamp, anomaly score, and Boolean anomaly flag for each successfully processed row.

### 7. Run automated tests

```cmd
python -m pytest tests -v
```

The current test suite contains seven tests covering feature validation, anomaly prediction output, threshold behavior, and saved-artifact handling.

## Reproducibility and Limitations

- Dataset provenance and real-world representativeness remain unverified.
- The project has not established deployment readiness or validated its predictions on live vessel telemetry.
- Classification performance varies by fault category.
- The anomaly threshold is sensitive to the training data and model score distribution.
- The current anomaly inference script processes the supplied dataset; it is not a live monitoring service.
- The final-test threshold experiment has a model-refitting mismatch that should be corrected in a future evaluation.
- Trained model artifacts and raw data are excluded from Git, so they must be generated or obtained locally.
- Model predictions should be treated as research outputs, not as a replacement for engineering inspection or safety procedures.

## Future Improvements

- Correct the anomaly threshold evaluation so the validation-fitted model and final evaluation use a consistent scoring scale.
- Evaluate on an independently sourced dataset or verified marine-engine telemetry.
- Add temporal or early-warning models that predict faults before they occur.
- Investigate class-specific performance and improve detection of underperforming fault categories.
- Add drift monitoring, model versioning, and repeatable experiment tracking.
- Build an interactive dashboard for sensor trends, anomaly scores, fault predictions, and model explanations.
- Develop a carefully validated inference API for future integration.

## License

A license has not yet been specified. Add an appropriate license before distributing or accepting contributions to the project.
