import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, brier_score_loss, roc_curve
)
from sklearn.calibration import calibration_curve

# Import XGBoost and LightGBM
import xgboost as xgb
import lightgbm as lgb

def train_and_evaluate():
    print("Loading datasets...")
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/y_test.csv").values.ravel()
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Calculate scale_pos_weight for class imbalance
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count
    print(f"Class imbalance ratio (neg/pos): {scale_pos_weight:.2f}")
    
    # Define models and their hyperparameter grids
    models_config = {
        "Logistic Regression": {
            "model": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            "params": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "penalty": ["l2"]
            }
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(class_weight="balanced", random_state=42),
            "params": {
                "max_depth": [5, 10, 15, 20],
                "min_samples_split": [2, 10, 20]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [10, 15, 20],
                "min_samples_leaf": [1, 4]
            }
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric="logloss", n_jobs=-1),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.01, 0.05, 0.1]
            }
        },
        "LightGBM": {
            "model": lgb.LGBMClassifier(scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1, verbose=-1),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [15, 31, 63]
            }
        },
        "Neural Network": {
            "model": MLPClassifier(max_iter=300, early_stopping=True, random_state=42),
            "params": {
                "hidden_layer_sizes": [(64, 32), (100,)],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate_init": [0.001, 0.01]
            }
        }
    }
    
    results = {}
    curves_data = {}
    
    # Set up plotting
    plt.figure(figsize=(10, 8))
    
    for name, config in models_config.items():
        print(f"\n--- Training {name} ---")
        clf = RandomizedSearchCV(
            config["model"],
            config["params"],
            n_iter=5,
            cv=3,
            scoring="roc_auc",
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        
        best_model = clf.best_estimator_
        print(f"Best parameters: {clf.best_params_}")
        
        # Save model
        joblib.dump(best_model, f"models/{name.lower().replace(' ', '_')}_model.pkl")
        
        # Predict
        y_pred = best_model.predict(X_test)
        
        # Predict probability
        if hasattr(best_model, "predict_proba"):
            y_prob = best_model.predict_proba(X_test)[:, 1]
        else:
            y_prob = best_model.decision_function(X_test)
            # scale decision function to 0-1
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
            
        # Calculate standard metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        
        # Calculate clinical metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        print(f"ROC-AUC: {roc_auc:.4f} | F1: {f1:.4f} | Sensitivity: {sensitivity:.4f} | Specificity: {specificity:.4f}")
        
        results[name] = {
            "ROC-AUC": float(roc_auc),
            "Accuracy": float(acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1-score": float(f1),
            "Sensitivity": float(sensitivity),
            "Specificity": float(specificity),
            "PPV": float(ppv),
            "NPV": float(npv),
            "Brier score": float(brier),
            "Best params": clf.best_params_
        }
        
        # Save curves for plotting
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        curves_data[name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "prob": y_prob.tolist()
        }
        
        # Plot ROC
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
        
    # Finalize and save ROC plot
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier (AUC = 0.500)")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curves Comparison")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig("results/roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot Calibration Curves
    plt.figure(figsize=(10, 8))
    for name, cdata in curves_data.items():
        prob_true, prob_pred = calibration_curve(y_test, cdata["prob"], n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=f"{name}")
    plt.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curves Comparison")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.savefig("results/calibration_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot Feature Importance of best model
    # We find the best model based on ROC-AUC
    best_model_name = max(results, key=lambda k: results[k]["ROC-AUC"])
    print(f"\nBest Model is: {best_model_name}")
    
    # Save results to json
    with open("results/model_comparison.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Generate Feature Importance for Tree/Linear Models if applicable
    best_clf = joblib.load(f"models/{best_model_name.lower().replace(' ', '_')}_model.pkl")
    
    if hasattr(best_clf, "feature_importances_"):
        importances = best_clf.feature_importances_
        feature_names = X_train.columns
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        
        plt.figure(figsize=(12, 8))
        feat_imp.head(15).plot(kind='barh', color='teal')
        plt.gca().invert_yaxis()
        plt.title(f"Top 15 Feature Importances ({best_model_name})")
        plt.xlabel("Relative Importance Score")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig("results/feature_importance.png", dpi=300)
        plt.close()
        
        # Save feature importance to CSV
        feat_imp.to_csv("results/feature_importances.csv")
        
    print("Training and evaluation completed. Results saved in 'results/' and 'models/' folders.")

if __name__ == "__main__":
    train_and_evaluate()
