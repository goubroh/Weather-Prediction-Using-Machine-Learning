# 🌧️ Weather Prediction Using Machine Learning

## A Comparative Analysis of Feature-Engineered Machine Learning Models for Weather Prediction

An end-to-end Machine Learning project for **binary rainfall classification using the Seattle Weather Dataset**. The project compares nine supervised learning algorithms and investigates how domain-specific, time-aware feature engineering can improve rainfall prediction performance.

The pipeline incorporates **cyclical seasonal encoding, multi-day lag features, rolling statistics, and thermal indices** to provide the models with temporal and meteorological context.

The best-performing model, the **Soft Voting Ensemble**, achieved an **accuracy of 96.23%** and a **ROC-AUC of 0.9824**.

---

## 📌 Project Overview

Weather prediction is an important application of Data Science and Machine Learning with practical relevance to agriculture, transportation, water-resource management, and disaster management.

This project formulates rainfall prediction as a **binary classification problem**:

* `1` → Rain
* `0` → No Rain

The project evaluates nine supervised Machine Learning algorithms on the Seattle Weather Dataset and compares their performance using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* ROC Curves

A major focus of the project is **domain-specific feature engineering** rather than relying only on generic dimensionality reduction techniques such as PCA.

---

## 🎯 Objectives

The main objectives of this project are:

1. Build a Machine Learning pipeline for rainfall classification.
2. Compare multiple supervised learning algorithms.
3. Engineer time-aware weather features.
4. Capture seasonal and short-term temporal patterns.
5. Prevent data leakage during feature construction and preprocessing.
6. Evaluate models using multiple classification metrics.
7. Identify the most effective algorithm for rainfall prediction.
8. Analyze the effectiveness of ensemble learning on engineered weather features.

---

## 📊 Dataset

### Seattle Weather Dataset

The project uses the **Seattle Weather Dataset**, containing daily weather observations.

The dataset includes:

* `date`
* `precipitation`
* `temp_max`
* `temp_min`
* `wind`
* `weather`

The `weather` attribute contains five atmospheric conditions:

* Sun
* Rain
* Snow
* Drizzle
* Fog

For this project, the original weather category is converted into a binary rainfall target:

```text
Rain     → 1
Other    → 0
```

This transforms the problem into a binary rainfall classification task.

---

## 🧠 Feature Engineering

Instead of relying solely on raw meteorological variables, the project uses domain-specific feature engineering designed to capture seasonal and temporal patterns.

### 1. Cyclical Seasonal Encoding

Month and day-of-year are transformed using sine and cosine functions.

```text
sin_month
cos_month

sin_day
cos_day
```

This allows the model to understand the cyclical nature of time.

For example, December and January are represented as adjacent seasonal periods rather than distant numerical values.

---

### 2. Thermal Indices

Two temperature-derived features are calculated:

```text
temp_range = temp_max - temp_min

temp_avg = (temp_max + temp_min) / 2
```

These features provide additional information about daily temperature behavior.

---

### 3. Multi-Day Lag Features

Historical weather observations are incorporated using lag features.

Lag features are generated for:

* Precipitation
* Maximum temperature
* Minimum temperature
* Wind speed

Across:

```text
1-day lag
2-day lag
3-day lag
```

Examples:

```text
precip_lag_1
precip_lag_2
precip_lag_3

temp_max_lag_1
temp_max_lag_2
temp_max_lag_3

temp_min_lag_1
temp_min_lag_2
temp_min_lag_3

wind_lag_1
wind_lag_2
wind_lag_3
```

These features allow the models to use recent weather history rather than relying only on the current day's measurements.

---

### 4. Three-Day Rolling Features

Three-day rolling averages are calculated for:

```text
precipitation
maximum temperature
wind speed
```

Examples:

```text
precip_roll_3
temp_max_roll_3
wind_roll_3
```

The rolling features are calculated using **previous observations only**, preventing current-day or future information from leaking into the prediction features.

---

### 5. Data Leakage Prevention

The preprocessing pipeline follows a leakage-aware approach.

Key measures include:

* Data is sorted chronologically before feature construction.
* Lag features use only historical observations.
* Rolling statistics are shifted before calculation.
* Missing rows introduced by lag/rolling operations are removed.
* Feature scaling is fitted only on the training partition.
* The test partition is not used during model fitting.

---

## 🔬 Machine Learning Models

Nine supervised Machine Learning classifiers are evaluated.

### 1. Logistic Regression

A linear classification model used as a baseline.

Configuration:

```text
Solver: liblinear
Class Weight: balanced
C: 1.0
```

---

### 2. Decision Tree

A tree-based classifier capable of learning nonlinear decision boundaries.

Configuration:

```text
Maximum Depth: 6
Minimum Samples Split: 10
Minimum Samples Leaf: 4
```

---

### 3. K-Nearest Neighbors

A distance-based classification algorithm.

Configuration:

```text
k = 7
Weights = distance
```

---

### 4. Gaussian Naive Bayes

A probabilistic classifier based on the conditional independence assumption.

---

### 5. Support Vector Machine

An RBF-kernel SVM is used to model nonlinear decision boundaries.

Configuration:

```text
Kernel: RBF
C: 2.0
Class Weight: balanced
Probability: enabled
```

---

### 6. Random Forest

A tree-based ensemble consisting of multiple decision trees.

Configuration:

```text
Number of Trees: 500
Maximum Depth: 16
Criterion: entropy
Class Weight: balanced subsample
Feature Sampling: sqrt
```

---

### 7. Gradient Boosting

A sequential ensemble of decision trees.

Configuration:

```text
Boosting Rounds: 300
Learning Rate: 0.05
Tree Depth: 5
Subsample: 0.85
Feature Sampling: sqrt
```

---

### 8. Histogram-Based Gradient Boosting

A computationally efficient gradient boosting approach using histogram-based feature binning.

Configuration:

```text
Boosting Iterations: 250
Learning Rate: 0.04
Maximum Depth: 6
L2 Regularization: 0.1
```

---

### 9. Soft Voting Ensemble 🏆

The final ensemble combines:

```text
Random Forest
Gradient Boosting
SVM
```

The ensemble uses the **average predicted probability** from the three classifiers.

This allows predictions from multiple strong models to be combined into a single classification decision.

---

# 📈 Model Performance

The models were evaluated on a **stratified 20% held-out test set**.

| Model                      |   Accuracy |  Precision |     Recall |   F1-Score |    ROC-AUC |
| -------------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| 🏆 **Voting Ensemble**     | **96.23%** | **98.35%** | **92.97%** | **95.58%** | **0.9824** |
| **Random Forest**          | **96.23%** | **98.35%** | **92.97%** | **95.58%** | **0.9800** |
| **Hist Gradient Boosting** | **96.23%** | **98.35%** | **92.97%** | **95.58%** | **0.9777** |
| **Gradient Boosting**      |     95.89% |     98.33% |     92.19% |     95.16% |     0.9770 |
| **Decision Tree**          |     95.89% |     97.54% |     92.97% |     95.20% |     0.9656 |
| **SVM**                    |     84.25% |     83.61% |     79.69% |     81.60% |     0.9130 |
| **Logistic Regression**    |     84.93% |     83.87% |     81.25% |     82.54% |     0.9110 |
| **Naive Bayes**            |     73.29% |     67.61% |     75.00% |     71.11% |     0.8350 |
| **KNN**                    |     72.60% |     69.67% |     66.41% |     68.00% |     0.8322 |

The performance values above are taken directly from the final experimental evaluation.

---

# 🏆 Key Results

### Best Accuracy

Three models achieved the highest accuracy:

```text
Voting Ensemble          → 96.23%
Random Forest            → 96.23%
Hist Gradient Boosting   → 96.23%
```

### Best ROC-AUC

The **Soft Voting Ensemble** achieved the highest ROC-AUC:

```text
ROC-AUC = 0.9824
```

followed by:

```text
Random Forest            → 0.9800
Hist Gradient Boosting   → 0.9777
Gradient Boosting        → 0.9770
Decision Tree            → 0.9656
SVM                      → 0.9130
Logistic Regression      → 0.9110
Naive Bayes              → 0.8350
KNN                      → 0.8322
```

The complete ROC-AUC ranking is reported in the project evaluation.

---

# 🔍 Key Findings

### 🌲 Tree-Based Models Performed Best

Tree-based ensemble models consistently achieved the strongest results.

Random Forest, Gradient Boosting, Histogram-based Gradient Boosting, and the Voting Ensemble all achieved ROC-AUC values above `0.97`.

This demonstrates their ability to capture nonlinear relationships in the engineered weather features.

### 🏆 Voting Ensemble Achieved the Best Discrimination

The Soft Voting Ensemble achieved:

```text
Accuracy  → 96.23%
ROC-AUC   → 0.9824
F1-Score  → 95.58%
```

It provided the highest ROC-AUC among all nine models.

### 🌳 Random Forest Offers a Strong Simpler Alternative

Random Forest achieved the same **96.23% accuracy** as the Voting Ensemble and a very close ROC-AUC of `0.9800`.

The difference between the Voting Ensemble and Random Forest was only `0.0024 ROC-AUC`, meaning the additional complexity of the ensemble provides only a modest improvement.

### 📅 Temporal Feature Engineering Matters

The project demonstrates that incorporating:

* Seasonal patterns
* Multi-day weather history
* Rolling statistics
* Temperature-derived features

can substantially improve the ability of classical Machine Learning models to classify rainfall.

The final evaluation reports improved performance after replacing PCA-based dimensionality reduction with time-sensitive feature engineering.

---

# 🏗️ Project Architecture

```text
Raw Weather Data
       │
       ▼
Data Preprocessing
       │
       ├── Chronological Sorting
       │
       ├── Cyclical Encoding
       │
       ├── Thermal Features
       │
       ├── Lag Features
       │
       └── Rolling Features
       │
       ▼
Leakage Prevention
       │
       ▼
80:20 Stratified Train-Test Split
       │
       ▼
Feature Standardization
       │
       ▼
Machine Learning Models
       │
       ├── Logistic Regression
       ├── Decision Tree
       ├── KNN
       ├── Naive Bayes
       ├── SVM
       ├── Random Forest
       ├── Gradient Boosting
       ├── Hist Gradient Boosting
       └── Soft Voting Ensemble
       │
       ▼
Performance Evaluation
       │
       ├── Accuracy
       ├── Precision
       ├── Recall
       ├── F1-Score
       ├── ROC-AUC
       ├── Confusion Matrix
       └── ROC Curves
```

---

# 📁 Project Structure

```text
Weather Prediction Using Machine Learning/
│
├── data/
│   ├── raw/
│   │   └── seattle-weather.csv
│   │
│   └── processed/
│       └── engineered_weather_data.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict_and_compare.py
│
├── models/
│   ├── scaler.joblib
│   └── trained_models.joblib
│
├── outputs/
│   ├── logs/
│   │   └── model_comparison_results.csv
│   │
│   └── plots/
│       └── roc_curves.png
│
├── requirements.txt
└── README.md
```

> **Note:** Update the directory structure above if your actual GitHub repository uses different filenames or folders.

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/goubroh/Weather-Prediction-Using-Machine-Learning.git
```

```bash
cd Weather-Prediction-Using-Machine-Learning
```

---

## 2. Create a Virtual Environment

### Windows PowerShell

```powershell
python -m venv venv
```

```powershell
.\venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Run the project scripts from the root directory.

### Step 1 — Train Models

```bash
python src/train.py
```

This step preprocesses the weather dataset, creates the engineered features, trains the Machine Learning models, and saves the trained model artifacts.

---

### Step 2 — Evaluate Models

```bash
python src/evaluate.py
```

This generates the model comparison results and ROC curve visualizations.

---

### Step 3 — Compare Predictions

```bash
python src/predict_and_compare.py
```

This script can be used to compare predictions generated by the trained models.

---

# 📊 Evaluation Metrics

The project evaluates models using multiple metrics.

### Accuracy

Measures the percentage of correctly classified observations.

### Precision

Measures how many observations predicted as rainfall were actually rainfall events.

### Recall

Measures how many actual rainfall events were correctly detected.

### F1-Score

Provides the harmonic mean of precision and recall.

### ROC-AUC

Measures the classifier's ability to discriminate between rainfall and non-rainfall observations across classification thresholds.

### Confusion Matrix

Provides:

```text
True Positives
True Negatives
False Positives
False Negatives
```

---

# 💻 Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Joblib**
* **Machine Learning**
* **Feature Engineering**
* **Ensemble Learning**
* **Time-Series Feature Engineering**

---

# ⚠️ Limitations

The current project has several limitations:

1. The model is trained and evaluated using the **Seattle Weather Dataset**, so performance may not directly generalize to other geographical locations.

2. The task is formulated as **binary rainfall classification** rather than predicting rainfall intensity, duration, or accumulated precipitation.

3. Although lag and rolling features provide temporal awareness, the models are not sequential deep-learning architectures and therefore cannot explicitly model long-range temporal dependencies.

4. ROC-AUC does not directly account for application-specific costs associated with false positives and false negatives.

5. Further validation using repeated cross-validation, confidence intervals, and statistical testing would provide stronger evidence for differences between the leading models.

---

# 🔮 Future Improvements

Potential future improvements include:

* Adding humidity and atmospheric pressure features.
* Testing the pipeline on multiple geographical locations.
* Performing repeated cross-validation.
* Applying statistical significance testing.
* Exploring LSTM and other temporal deep-learning architectures.
* Extending the binary rainfall classification task to rainfall intensity prediction.
* Incorporating additional meteorological variables.
* Evaluating the models under different geographical and seasonal conditions.

These directions are consistent with the future-work discussion in the project evaluation.

---

# 📌 Conclusion

This project demonstrates the effectiveness of **domain-specific feature engineering combined with classical Machine Learning algorithms** for rainfall classification.

The **Soft Voting Ensemble** achieved the highest ROC-AUC of **0.9824**, while the Voting Ensemble, Random Forest, and Histogram-based Gradient Boosting models all achieved the highest accuracy of **96.23%**.

The results show that tree-based ensemble models are particularly effective when combined with temporal weather features such as cyclical seasonal encoding, multi-day lag variables, and rolling statistics.

At the same time, the very small performance gap between the Voting Ensemble and Random Forest suggests that a simpler single-model solution such as Random Forest can provide nearly equivalent performance with lower model complexity.

---

# 👨‍💻 Author

**Gourab Barui**

M.Tech Data Science
Amity University, Noida

---

# 🤝 Acknowledgements

Special thanks to:

**Dr. Neha Tyagi**
Department of CSE
Amity University

**Arhina Ghosh**
Department of CS
Noida Institute of Engineering and Technology

for their valuable guidance and support throughout the project.

---

# 🔗 Repository

**GitHub:**
https://github.com/goubroh/Weather-Prediction-Using-Machine-Learning

---

## ⭐ If you found this project useful

Feel free to **star ⭐ the repository**, explore the implementation, and share your feedback.
