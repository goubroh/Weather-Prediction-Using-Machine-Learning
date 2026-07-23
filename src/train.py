import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from data_preprocessing import load_and_preprocess_data

def train_all_models():
    # 1. Load Enhanced Features
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # 2. Optimized Models Configuration
    models = {
        'Logistic Regression': LogisticRegression(
            solver='liblinear', class_weight='balanced', C=1.0, random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=6, min_samples_split=5, random_state=42
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=7, weights='distance'
        ),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight='balanced', random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42
        ),
        'SVM': SVC(
            kernel='rbf', C=1.5, probability=True, class_weight='balanced', random_state=42
        )
    }

    trained_models = {}

    # 3. Train Models
    print("Training models on feature-engineered dataset...")
    for name, model in models.items():
        print(f" -> Training: {name}")
        model.fit(X_train, y_train)
        trained_models[name] = model

    # 4. Save Trained Models
    model_save_path = "models/trained_models.joblib"
    joblib.dump(trained_models, model_save_path)
    print(f"\nAll models trained and saved to {model_save_path}")

if __name__ == "__main__":
    train_all_models()