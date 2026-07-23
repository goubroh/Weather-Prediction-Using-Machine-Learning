import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from data_preprocessing import load_and_preprocess_data

def evaluate_all_models():
    # 1. Load data and saved models
    _, X_test, _, y_test = load_and_preprocess_data(save_scaler=False)
    models = joblib.load("models/trained_models.joblib")

    results = {}

    # 2. Evaluate each algorithm
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = None

        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob) if y_prob is not None else None,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'fpr_tpr': roc_curve(y_test, y_prob) if y_prob is not None else (None, None, None)
        }

    # 3. Create metrics DataFrame
    rows = []
    for model_name, metrics in results.items():
        rows.append({
            'Algorithm': model_name,
            'Accuracy': round(metrics['accuracy'], 4),
            'Precision': round(metrics['precision'], 4),
            'Recall': round(metrics['recall'], 4),
            'F1 Score': round(metrics['f1_score'], 4),
            'ROC AUC': round(metrics['roc_auc'], 4) if metrics['roc_auc'] is not None else None
        })

    summary_df = pd.DataFrame(rows).sort_values(by='F1 Score', ascending=False)
    
    print("\n================ Model Comparative Performance Summary ================")
    print(summary_df.to_string(index=False))

    # 4. Save CSV results
    log_dir = "outputs/logs"
    os.makedirs(log_dir, exist_ok=True)
    csv_out_path = os.path.join(log_dir, "model_comparison_results.csv")
    summary_df.to_csv(csv_out_path, index=False)
    print(f"\nSummary table saved to {csv_out_path}")

    # 5. Plot ROC Curves (Resized popup figure)
    plot_dir = "outputs/plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Compact figure size for comfortable desktop display window
    plt.figure(figsize=(7.5, 5))
    
    for model_name, metrics in results.items():
        fpr, tpr, _ = metrics['fpr_tpr']
        if fpr is not None and tpr is not None:
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {metrics["roc_auc"]:.4f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Baseline (AUC = 0.50)')
    plt.title('ROC Curves - Seattle Weather Prediction Models', fontsize=11, fontweight='bold')
    plt.xlabel('False Positive Rate', fontsize=9.5)
    plt.ylabel('True Positive Rate', fontsize=9.5)
    plt.legend(loc='lower right', fontsize=8.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    # Save high-res image to disk, keep screen popup compact
    plot_out_path = os.path.join(plot_dir, "roc_curves.png")
    plt.savefig(plot_out_path, dpi=300)
    print(f"ROC plot saved to {plot_out_path}")
    plt.show()

if __name__ == "__main__":
    evaluate_all_models()