# Explainable ML for 30-Day Hospital Readmission Prediction

This repository contains an explainable machine learning framework to predict 30-day hospital readmission risk using Electronic Health Records (EHR). The implementation is based on the **Diabetes 130-US Hospitals dataset** from the UCI Machine Learning Repository.

## Project Structure
- `download_data.py`: Downloads the raw dataset and extracts it to `data/`.
- `preprocess.py`: Performs data cleaning, duplicates removal, deceased patient exclusion, categorical mapping (such as ICD-9 categories), feature engineering, scaling, and train-test splitting.
- `train.py`: Trains and benchmarks six models (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, Neural Network), plots ROC and Calibration curves, and saves the best model.
- `explain.py`: Performs SHAP and LIME interpretability analysis, generating global summary plots and local patient-level explanation plots.
- `hospital_readmission_paper.md`: The complete, publication-ready research paper (approx. 8,000–10,000 words) detailing the study.
- `results/`: Directory containing performance comparison tables and visualization plots.
- `models/`: Directory containing saved trained models (ignored in git).

## Setup & Running the Pipeline

### 1. Prerequisites
Ensure Python 3.10+ and Git are installed.

### 2. Install Dependencies
Create a virtual environment and install the required libraries:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install pandas numpy scikit-learn xgboost lightgbm shap lime matplotlib seaborn scipy
```

### 3. Run the Pipeline
To execute the pipeline end-to-end:
```bash
# 1. Download and extract the dataset
python download_data.py

# 2. Clean and preprocess the data
python preprocess.py

# 3. Train models and generate performance benchmarks
python train.py

# 4. Run explainability analysis (SHAP & LIME)
python explain.py
```

## Results Overview
The models were benchmarked on a clean cohort of **69,970 unique patient encounters**. **LightGBM** achieved the highest discriminative performance:
- **ROC-AUC:** 0.6563
- **Sensitivity (Recall):** 53.55%
- **Specificity:** 69.03%
- **Brier Calibration Score:** 0.1989

All evaluation and explainability plots (SHAP global summary, SHAP waterfall, LIME bar chart) are saved in the `results/` folder.
