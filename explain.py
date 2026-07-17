import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from lime import lime_tabular

def generate_explanations():
    print("Loading data for XAI analysis...")
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").values.ravel()
    
    os.makedirs("results", exist_ok=True)
    
    # 1. Determine the best model from the results json
    with open("results/model_comparison.json", "r") as f:
        comparison = json.load(f)
        
    best_model_name = max(comparison, key=lambda k: comparison[k]["ROC-AUC"])
    print(f"Best model based on ROC-AUC: {best_model_name}")
    
    # Load the best model
    model_filename = f"models/{best_model_name.lower().replace(' ', '_')}_model.pkl"
    model = joblib.load(model_filename)
    
    # 2. Get predictions on the test set
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = model.decision_function(X_test)
        probs = (probs - probs.min()) / (probs.max() - probs.min())
    
    preds = model.predict(X_test)
    
    # Find a patient index with high predicted risk (actual = 1)
    high_risk_indices = np.where((y_test == 1) & (probs > 0.4))[0]
    if len(high_risk_indices) > 0:
        high_risk_idx = int(high_risk_indices[0])
    else:
        high_risk_idx = int(np.argmax(probs))
        
    # Find a patient index with low predicted risk (actual = 0)
    low_risk_indices = np.where((y_test == 0) & (probs < 0.1))[0]
    if len(low_risk_indices) > 0:
        low_risk_idx = int(low_risk_indices[0])
    else:
        low_risk_idx = int(np.argmin(probs))
        
    print(f"Selected patient for high-risk explanation: Index {high_risk_idx} (Prob: {probs[high_risk_idx]:.4f})")
    print(f"Selected patient for low-risk explanation: Index {low_risk_idx} (Prob: {probs[low_risk_idx]:.4f})")
    
    # 3. Global SHAP Analysis
    print("Calculating SHAP values (using 500 representative test samples)...")
    # Sample test data to speed up SHAP
    np.random.seed(42)
    sample_indices = np.random.choice(X_test.index, size=500, replace=False)
    X_test_sample = X_test.iloc[sample_indices]
    
    # Use SHAP Explainer
    # If Tree model (XGBoost, LightGBM, Random Forest, Decision Tree), use TreeExplainer
    # Otherwise, use KernelExplainer or general Explainer
    is_tree = any(term in best_model_name for term in ["XGBoost", "LightGBM", "Random Forest", "Decision Tree"])
    
    if is_tree:
        print("Using SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(model)
        # Note: some models return raw margins, some probabilities. Let's get the values.
        shap_values = explainer(X_test_sample)
    else:
        print("Using SHAP general Explainer (Kernel/Model-agnostic)...")
        # Use a background dataset for reference (kmeans to summarize train set to 20 samples to make it fast)
        background = shap.kmeans(X_train, 20)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test_sample)
        
    print("SHAP calculations complete. Plotting global summary...")
    
    # Plot SHAP summary (violin/dot plot)
    plt.figure(figsize=(12, 10))
    if hasattr(shap_values, "values"):
        # For Explanation object (TreeExplainer)
        # If binary classification, SHAP returns list of shapes or 3D array for multiclass.
        # TreeExplainer on LightGBM/XGBoost returns shape (N, D) or (N, D, 2).
        # We need the values for class 1 if it has 3 dimensions.
        vals = shap_values.values
        if len(vals.shape) == 3:
            # shape is (N, D, 2) - take class 1
            shap_values_class1 = shap_values[:, :, 1]
            shap.summary_plot(shap_values_class1, X_test_sample, show=False)
        else:
            shap.summary_plot(shap_values, X_test_sample, show=False)
    else:
        # For list returned by KernelExplainer
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X_test_sample, show=False)
        else:
            shap.summary_plot(shap_values, X_test_sample, show=False)
            
    plt.title(f"SHAP Global Feature Importance Summary ({best_model_name})", fontsize=14)
    plt.tight_layout()
    plt.savefig("results/shap_summary_plot.png", dpi=300)
    plt.close()
    
    # 4. Local SHAP Explanation (Waterfall plot)
    print("Generating local SHAP waterfall plots...")
    plt.figure(figsize=(10, 6))
    
    # Re-calculate explainer for the single patients if needed, or index from explanation
    single_explainer = shap.Explainer(model, X_train)
    single_shap_high = single_explainer(X_test.iloc[[high_risk_idx]])
    single_shap_low = single_explainer(X_test.iloc[[low_risk_idx]])
    
    # Plot high-risk waterfall
    plt.figure(figsize=(10, 6))
    if len(single_shap_high.values.shape) == 3:
        # Multi-class output (class 0 and class 1)
        shap.plots.waterfall(single_shap_high[0, :, 1], show=False)
    else:
        shap.plots.waterfall(single_shap_high[0], show=False)
    plt.title(f"Local SHAP Explanation for High-Risk Patient (Index {high_risk_idx})", fontsize=12)
    plt.tight_layout()
    plt.savefig("results/shap_local_high_risk.png", dpi=300)
    plt.close()
    
    # Plot low-risk waterfall
    plt.figure(figsize=(10, 6))
    if len(single_shap_low.values.shape) == 3:
        shap.plots.waterfall(single_shap_low[0, :, 1], show=False)
    else:
        shap.plots.waterfall(single_shap_low[0], show=False)
    plt.title(f"Local SHAP Explanation for Low-Risk Patient (Index {low_risk_idx})", fontsize=12)
    plt.tight_layout()
    plt.savefig("results/shap_local_low_risk.png", dpi=300)
    plt.close()
    
    # 5. Local LIME Explanation
    print("Generating LIME explanations...")
    lime_explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['Low Risk', 'High Risk'],
        mode='classification',
        random_state=42
    )
    
    # Explain high risk patient
    exp_high = lime_explainer.explain_instance(
        data_row=X_test.iloc[high_risk_idx].values,
        predict_fn=model.predict_proba,
        num_features=10
    )
    
    # Save LIME HTML
    exp_high.save_to_file("results/lime_local_high_risk.html")
    
    # Save LIME plot
    fig = exp_high.as_pyplot_figure()
    plt.title(f"Local LIME Explanation for High-Risk Patient (Index {high_risk_idx})", fontsize=12)
    plt.tight_layout()
    plt.savefig("results/lime_local_high_risk.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("Explainability analysis complete. Visualizations saved in 'results/' folder.")

if __name__ == "__main__":
    generate_explanations()
