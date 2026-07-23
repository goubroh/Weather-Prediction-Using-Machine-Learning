import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    HistGradientBoostingClassifier,
    VotingClassifier
)
from sklearn.svm import SVC
from data_preprocessing import load_and_preprocess_data

def train_all_models():
    # 1. Load Preprocessed Data
    X_train, _, y_train, _ = load_and_preprocess_data(save_scaler=True)

    # Base models for Voting Ensemble
    rf_base = RandomForestClassifier(
        n_estimators=500,
        max_depth=16,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced_subsample',
        criterion='entropy',
        random_state=42,
        n_jobs=-1
    )

    gb_base = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.85,
        max_features='sqrt',
        random_state=42
    )

    svm_base = SVC(
        kernel='rbf', C=2.0, probability=True, class_weight='balanced', random_state=42
    )

    # 2. High-Accuracy Models Configuration
    models = {
        'Logistic Regression': LogisticRegression(
            solver='liblinear', class_weight='balanced', C=1.0, random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=6, min_samples_split=10, min_samples_leaf=4, random_state=42
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=7, weights='distance'
        ),
        'Naive Bayes': GaussianNB(),
        'SVM': svm_base,
        'Random Forest': rf_base,
        'Gradient Boosting': gb_base,
        'Hist Gradient Boosting': HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.04, max_depth=6, l2_regularization=0.1, random_state=42
        ),
        # Ensemble Voting Classifier for Peak Accuracy
        'Voting Ensemble': VotingClassifier(
            estimators=[
                ('rf', rf_base),
                ('gb', gb_base),
                ('svm', svm_base)
            ],
            voting='soft'
        )
    }

    trained_models = {}

    # 3. Execute Training Loop
    print("Training models on feature-engineered dataset...")
    for name, model in models.items():
        print(f" -> Training: {name}")
        model.fit(X_train, y_train)
        trained_models[name] = model

    # 4. Save Models
    os.makedirs("models", exist_ok=True)
    model_save_path = "models/trained_models.joblib"
    joblib.dump(trained_models, model_save_path)
    print(f"\nAll models trained successfully and saved to {model_save_path}")

if __name__ == "__main__":
    train_all_models()