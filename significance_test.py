import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats

def run_significance_testing():
    print("Loading test data...")
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").values.ravel()
    
    model_names = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "Neural Network"
    ]
    
    probs = {}
    
    print("Generating predictions from saved models...")
    for name in model_names:
        model_path = f"models/{name.lower().replace(' ', '_')}_model.pkl"
        if not os.path.exists(model_path):
            print(f"Model path {model_path} does not exist.")
            continue
        model = joblib.load(model_path)
        
        if hasattr(model, "predict_proba"):
            probs[name] = model.predict_proba(X_test)[:, 1]
        else:
            p = model.decision_function(X_test)
            probs[name] = (p - p.min()) / (p.max() - p.min())
            
    print("\nRunning bootstrapping for 95% Confidence Intervals and Hypothesis Testing (B = 1000)...")
    np.random.seed(42)
    n_samples = len(y_test)
    n_bootstraps = 1000
    
    # Store bootstrap statistics
    boot_aucs = {name: [] for name in probs}
    boot_diffs = [] # LightGBM - Logistic Regression
    
    for i in range(n_bootstraps):
        if (i + 1) % 200 == 0:
            print(f"Bootstrap iteration {i + 1}/{n_bootstraps}...")
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # Ensure bootstrap sample has both classes
        while len(np.unique(y_test[indices])) < 2:
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
        for name in probs:
            auc = roc_auc_score(y_test[indices], probs[name][indices])
            boot_aucs[name].append(auc)
            
        diff = roc_auc_score(y_test[indices], probs["LightGBM"][indices]) - roc_auc_score(y_test[indices], probs["Logistic Regression"][indices])
        boot_diffs.append(diff)
        
    print("\n--- Statistical Significance Results ---")
    for name in probs:
        aucs = np.array(boot_aucs[name])
        mean_auc = np.mean(aucs)
        ci_lower = np.percentile(aucs, 2.5)
        ci_upper = np.percentile(aucs, 97.5)
        print(f"{name}: Mean AUC = {mean_auc:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
        
    # Calculate two-sided p-value for LightGBM vs Logistic Regression
    boot_diffs = np.array(boot_diffs)
    p_val_two_sided = 2 * min(np.sum(boot_diffs <= 0) / n_bootstraps, np.sum(boot_diffs >= 0) / n_bootstraps)
    print(f"\nHypothesis Test: LightGBM vs. Logistic Regression")
    print(f"Mean Difference in AUC: {np.mean(boot_diffs):.4f}")
    print(f"Two-sided bootstrap p-value: {p_val_two_sided:.4f}")
    
    # Save results to a json file
    stats_results = {
        name: {
            "mean_auc": float(np.mean(boot_aucs[name])),
            "ci_lower": float(np.percentile(boot_aucs[name], 2.5)),
            "ci_upper": float(np.percentile(boot_aucs[name], 97.5))
        } for name in probs
    }
    stats_results["comparison"] = {
        "models": "LightGBM vs Logistic Regression",
        "mean_diff": float(np.mean(boot_diffs)),
        "p_value": float(p_val_two_sided)
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/statistical_significance.json", "w") as f:
        import json
        json.dump(stats_results, f, indent=4)
        
if __name__ == "__main__":
    run_significance_testing()
