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
    _, X_test, _, y_test = load_and_preprocess_data()
    models = joblib.load("models/trained_models.joblib")

    results = {}

    # 2. Evaluate each algorithm
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        # Check for probability prediction capability (used for ROC AUC)
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

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

    # 4. Save results table to outputs/logs/
    csv_out_path = "outputs/logs/model_comparison_results.csv"
    summary_df.to_csv(csv_out_path, index=False)
    print(f"\nSummary table saved to {csv_out_path}")

    # 5. Plot and save ROC Curves to outputs/plots/
    plt.figure(figsize=(9, 6))
    for model_name, metrics in results.items():
        fpr, tpr, _ = metrics['fpr_tpr']
        if fpr is not None and tpr is not None:
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {metrics["roc_auc"]:.4f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title('ROC Curves - Seattle Weather Prediction Models')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)

    plot_out_path = "outputs/plots/roc_curves.png"
    plt.savefig(plot_out_path)
    print(f"ROC plot saved to {plot_out_path}")
    plt.show()

if __name__ == "__main__":
    evaluate_all_models()