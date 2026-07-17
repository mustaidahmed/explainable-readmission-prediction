# Cover Letter for Manuscript Submission

**Date:** July 17, 2026  

**To:**  
The Editor-in-Chief  
*Artificial Intelligence in Medicine*  
Elsevier  

**Subject:** Submission of Manuscript for Peer Review  

Dear Editor-in-Chief,  

I am pleased to submit our manuscript titled **"An Explainable Machine Learning Framework for Predicting 30-Day Hospital Readmissions Using Electronic Health Records"** for consideration for publication as a Research Article in *Artificial Intelligence in Medicine*.  

This study addresses one of the most critical barriers to the clinical adoption of machine learning: the "black-box" nature of high-performing predictive models in healthcare. Rather than presenting a simple application study on disease prediction, this work proposes a **reproducible, explainable AI (XAI) validation framework** designed to resolve clinical trust and methodological data leakage issues in Electronic Health Record (EHR) modeling.

### Core Contributions & Methodological Highlights:
1. **Clinical Compliance and Data Hygiene:** Unlike previous studies that report inflated performance by retaining multiple encounters for the same patient or including expired patients, our framework introduces a strict cohort selection pipeline. By keeping only first encounters and excluding deceased/hospice discharges, we ensure the independence of samples and clinical validity, testing the framework on a clean cohort of **69,970 unique patient encounters**.
2. **Algorithmic Benchmarking and Calibration:** We benchmark six distinct machine learning algorithms (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, and an MLP Neural Network) under identical preprocessing splits. Critically, we evaluate both discriminative accuracy (ROC-AUC) and **probability calibration (Brier scores and calibration curves)** to ensure that the risk scores are reliable and safe for bedside clinical decision support.
3. **Dual-Interpretability Validation (SHAP + LIME):** We bridge the gap between predictive capability and bedside clinical interpretability. By integrating both **SHapley Additive Explanations (SHAP)** for global feature attribution and **Local Interpretable Model-agnostic Explanations (LIME)** for patient-specific explanations, we validate the consistency of explanations on individual patient case studies.
4. **Statistical Significance Testing:** We present bootstrap-based 95% Confidence Intervals for all models and conduct a non-parametric hypothesis test comparing the best-performing model (LightGBM) against the baseline (Logistic Regression), reporting an honest, scientifically grounded evaluation of performance gains.

We believe this work is highly aligned with the scope of *Artificial Intelligence in Medicine*, as it offers a practical, transparent, and methodologically sound blueprint for integrating explainable machine learning into contemporary clinical workflows. 

This manuscript has not been published elsewhere, nor is it under consideration by any other journal. All authors have approved the manuscript and agree with its submission. The full source code, data preprocessing pipeline, and model evaluation scripts are made publicly available to ensure reproducibility: [https://github.com/mustaidahmed/explainable-readmission-prediction](https://github.com/mustaidahmed/explainable-readmission-prediction).

Thank you for your time and consideration of our work. I look forward to hearing from you.

Sincerely,  

**Mustaid Ahmed**  
Saint Louis University  
St. Louis, MO, USA  
Email: [Your Email Address]  
