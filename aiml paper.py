import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv(r'C:\Users\0aaso\Desktop\research\1\seattle-weather.csv')

# Create binary target variable: 1 if rain, else 0
data['target'] = data['weather'].apply(lambda x: 1 if x == 'rain' else 0)

# Select features and target
features = ['precipitation', 'temp_max', 'temp_min', 'wind']
X = data[features]
y = data['target']

# Handle missing or zero values if any - here just fillna with median as an example
X = X.fillna(X.median())

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA (optional) retain 3 components for demonstration
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Split data 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, stratify=y, random_state=42)

# Helper function for model evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc_score(y_test, y_prob):.4f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

# List of classifiers
models = {
    "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(kernel='linear', probability=True, random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    print(f"{name} Performance:")
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test)



# Create binary target variable: 1 if rain, else 0
data['target'] = data['weather'].apply(lambda x: 1 if x == 'rain' else 0)

# Select features and target
features = ['precipitation', 'temp_max', 'temp_min', 'wind']
X = data[features]
y = data['target']

# Handle missing or zero values if any
X = X.fillna(X.median())

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Split data 70% train, 30% test (using PCA-transformed data)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, stratify=y, random_state=42)

# Define models
models = {
    'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='linear', probability=True, random_state=42)
}

results = {}

# Train models, evaluate and gather results
for name, model in models.items():
    # All models now use the PCA-transformed and scaled data (X_train, X_test)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'fpr_tpr': roc_curve(y_test, y_prob) if y_prob is not None else (None, None, None)
    }

# Create a summary DataFrame for the metrics
summary_df = pd.DataFrame({
    'Algorithm': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': [],
    'ROC AUC': []
})

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

summary_df = pd.DataFrame(rows)


print(summary_df.sort_values(by='F1 Score', ascending=False))

# Plot ROC curves for all models with ROC AUC
plt.figure(figsize=(8,6))
for model_name, metrics in results.items():
    fpr, tpr, _ = metrics['fpr_tpr']
    if fpr is not None and tpr is not None:
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {metrics["roc_auc"]:.4f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curves for Seattle Weather Prediction Models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()