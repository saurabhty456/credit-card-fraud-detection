# FRAUDGUARD: Credit Card Fraud Detection Using Machine Learning & Flask

> **A Comprehensive College Mini-Project in Machine Learning and Full-Stack Web Development**

---

## 1. Project Title
**Credit Card Fraud Detection Using Machine Learning and Web Development**  
*Application Name:* **FRAUDGUARD**

---

## 2. Project Objective
The primary objective of this project is to build an end-to-end, production-ready machine learning system and web application capable of identifying potentially fraudulent credit card transactions in real time. It bridges advanced data science (handling extreme class imbalance, data leakage prevention, PCA features, and ensemble modeling) with an intuitive, responsive web application for real-time inference and evaluation visualization.

---

## 3. Problem Statement
Credit card fraud costs consumers and financial institutions billions of dollars worldwide annually. Detecting fraud poses several critical computational challenges:
1. **Extreme Class Imbalance:** Fraudulent transactions account for a minuscule fraction of all transactions (less than 0.2%).
2. **Asymmetric Error Costs:** Failing to detect a fraudulent transaction (**False Negative**) results in direct financial loss and compromised account security, whereas incorrectly flagging a legitimate purchase (**False Positive**) causes customer friction and alert fatigue.
3. **Data Confidentiality:** Cardholder information is highly sensitive and protected by strict privacy laws (GDPR, PCI-DSS), requiring dimensionality reduction and anonymization before analysis.

---

## 4. Key Features
- **Exploratory Data Analysis (EDA):** Automated analysis computing distribution statistics, class imbalance ratios, and financial amount distributions.
- **Leak-Free Machine Learning Pipeline:** Strict stratification and preprocessing order ensuring no statistical leakage from test data into training steps.
- **Imbalance Handling:** Balanced class weighting applied to both linear (Logistic Regression) and ensemble (Random Forest) models.
- **Empirical Model Comparison:** Real-time side-by-side comparison across Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- **Dual-Mode Web Interface:**
  - **Mode 1 (Demo Transaction):** Instant pre-fill with authentic legitimate and fraudulent records taken directly from the unseen test split.
  - **Mode 2 (Manual Parameter Tuning):** Manual adjustment of Amount, Time, and PCA features (V1–V28).
- **Interactive Analytics Dashboard:** Real evaluation graphs (Confusion Matrices, ROC curves, Class Distribution, Correlation bar charts).
- **Ethical & Secure Architecture:** No collection or storage of genuine card numbers, CVV, or PINs.

---

## 5. Technology Stack
- **Programming Language:** Python 3
- **Machine Learning & Data Processing:**
  - `pandas` — Tabular data manipulation and statistical summaries
  - `numpy` — Numerical array operations
  - `scikit-learn` — Preprocessing (StandardScaler), classifiers (LogisticRegression, RandomForestClassifier), and evaluation metrics
  - `joblib` — Model and pipeline serialization
- **Data Visualization:**
  - `matplotlib` — Base plotting engine
  - `seaborn` — Statistical visualizations and heatmaps
- **Backend Web Framework:**
  - `Flask` — Lightweight WSGI web framework, REST API, request validation, and routing
- **Frontend Architecture:**
  - `HTML5` — Semantic structure and accessible forms
  - `CSS3` — Modern financial dashboard styling, responsive layouts, badges, and cards
  - `JavaScript (ES6)` — Dynamic DOM manipulation, test sample auto-population, and client-side validation

---

## 6. Dataset Description
The model is trained on the real-world **Credit Card Fraud Detection Dataset** (European cardholders, September 2013).
- **Total Transactions:** `284,807`
- **Total Features:** `31` (30 input features + 1 target variable `Class`)
- **Missing Values:** `0` (Zero missing entries across the entire dataset)
- **Duplicate Rows:** `1,081` (0.380%)
- **Data Span:** 48 consecutive hours (172,792 seconds)

---

## 7. Dataset Features
1. **`Time`:** Number of seconds elapsed between each transaction and the first transaction in the dataset.
2. **`V1` to `V28`:** Principal components obtained via **Principal Component Analysis (PCA)**. Due to privacy and security reasons, the original card details (user identities, locations, merchant categories) are transformed into these orthogonal numerical dimensions.
3. **`Amount`:** Transaction value in Euros (€). Ranges from `€0.00` to `€25,691.16`.
4. **`Class` (Target):**
   - `0` = Legitimate transaction
   - `1` = Fraudulent transaction

---

## 8. Class Imbalance
The dataset exhibits extreme class imbalance:
- **Legitimate (Class 0):** `284,315` transactions (**99.8273%**)
- **Fraudulent (Class 1):** `492` transactions (**0.1727%**)
- **Imbalance Ratio:** ~`1 fraud per 577 legitimate transactions`

### Why Accuracy is Misleading:
If a trivial baseline model simply predicts `0` for every transaction without looking at the features, it would achieve an accuracy of **99.83%**, yet miss **100% of all fraud cases**. Therefore, **Recall**, **Precision**, and **F1-Score** are the critical metrics for evaluating fraud detection performance.

---

## 9. Preprocessing Pipeline & Zero Data Leakage
To guarantee realistic model evaluation, **data leakage** is strictly prevented through the following order of operations:
1. Features `X` (Time, V1–V28, Amount) and target `y` (Class) are separated.
2. A **Stratified Train-Test Split (80% Train, 20% Test)** is executed using `stratify=y, random_state=42`. Stratification ensures the exact same 0.17% fraud ratio is preserved in both subsets:
   - **Train Set:** 227,845 records (394 fraud cases)
   - **Test Set:** 56,962 records (98 fraud cases)
3. `StandardScaler` is **fitted solely on the training data** (`X_train`), learning its mean and variance.
4. The test data (`X_test`) is transformed using the parameters learned from `X_train`.

---

## 10. Machine Learning Algorithms
Two distinct supervised learning algorithms were implemented and trained:

### Model 1: Logistic Regression (Linear Baseline)
- **Configuration:** `class_weight='balanced'`, `max_iter=1000`, `solver='lbfgs'`, `random_state=42`.
- **Mechanism:** Models the log-odds of a transaction being fraud as a linear combination of scaled input features using the sigmoid function:
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
- **Observation:** Fast training (<1s), high recall (91.84%), but linear boundary generates excessive false positives (1,386 false alarms, 6.10% precision).

### Model 2: Random Forest Classifier (Ensemble Non-linear)
- **Configuration:** `n_estimators=100`, `max_depth=12`, `class_weight='balanced'`, `random_state=42`, `n_jobs=-1`.
- **Mechanism:** Builds an ensemble of 100 decorrelated decision trees, aggregating their voting probabilities to capture non-linear feature interactions and high-dimensional boundaries.
- **Observation:** Drastically reduces false alarms down to 32 while retaining high recall (82.65%), yielding an outstanding F1-score of 0.7678.

---

## 11. Evaluation Metrics
All metrics were computed on the **unseen 20% test partition (56,962 transactions)**:

- **Accuracy:** Total correct classifications divided by total transactions:
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
- **Precision:** Of all transactions predicted as fraud, how many actually were fraud:
  $$\text{Precision} = \frac{TP}{TP + FP}$$
- **Recall (Sensitivity):** Of all actual frauds in the dataset, how many did the model detect:
  $$\text{Recall} = \frac{TP}{TP + FN}$$
- **F1-Score:** The harmonic mean of precision and recall:
  $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **ROC-AUC:** Area Under the Receiver Operating Characteristic Curve measuring separation capability across decision thresholds.

---

## 12. Empirical Model Comparison

| Evaluation Metric | Logistic Regression | Random Forest (Selected Best) | Performance Winner |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 97.55% | **99.91%** | Random Forest |
| **Precision (Class 1)** | 6.10% | **71.68%** | **Random Forest (+65.58%)** |
| **Recall (Class 1)** | **91.84%** (90/98) | 82.65% (81/98) | Logistic Regression |
| **F1-Score** | 0.1144 | **0.7678** | **Random Forest (+0.6534)** |
| **ROC-AUC** | 0.9721 | **0.9790** | Random Forest |
| **False Positives (FP)** | 1,386 false alarms | **32 false alarms** | **Random Forest (43x fewer errors)** |
| **False Negatives (FN)** | 8 missed frauds | 17 missed frauds | Logistic Regression |

### Why Random Forest Was Selected:
Logistic Regression's 6.10% precision means that for every 1 true fraud detected, **over 15 legitimate cardholders are falsely blocked**. In contrast, Random Forest maintains strong fraud capture (82.65%) while generating only 32 false positives across 56,962 transactions, giving an **F1-Score of 0.7678** vs 0.1144.

---

## 13. Project Architecture

```
User (Browser)
    │
    ▼
Modern Web Interface (HTML5 / CSS3 / JavaScript)
    │  [POST /predict with 30 transaction parameters]
    ▼
Flask Backend (app.py)
    │  [Input validation, type checking, missing value defense]
    ▼
Data Preprocessing (StandardScaler)
    │  [Transforms Time, Amount, V1–V28 using training distribution]
    ▼
Machine Learning Model (Random Forest Classifier)
    │  [Computes binary classification & tree voting probabilities]
    ▼
Inference Result Page (result.html)
    │  [Status: Legitimate / Fraud, Risk Level: Low / High, Risk Score %]
    ▼
User Display
```

---

## 14. Folder Structure

```
credit-card-fraud-detection/
│
├── app.py                      # Flask web application and API server
├── train_model.py              # ML pipeline: train, evaluate, and save artifacts
├── eda.py                      # Exploratory Data Analysis & visual generation
├── test_app.py                 # Automated end-to-end test suite (10 tests)
├── requirements.txt            # Python dependencies
├── README.md                   # Complete project documentation
├── VIVA_GUIDE.md               # College viva preparation questions & answers
├── .gitignore                  # Git cache and temporary files ignore list
│
├── dataset/
│   └── creditcard.csv          # Historical credit card dataset (284,807 rows)
│
├── models/
│   ├── fraud_model.pkl         # Trained Random Forest classifier
│   ├── logistic_model.pkl      # Trained Logistic Regression classifier
│   ├── scaler.pkl              # Fitted StandardScaler object
│   ├── pipeline.pkl            # Complete Scikit-learn Pipeline
│   ├── model_metrics.json      # Actual empirical metrics from test split
│   └── sample_transactions.json# Real test samples for Mode 1 demo
│
├── templates/
│   ├── base.html               # Master layout template with navbar & footer
│   ├── index.html              # Home page with project statistics & architecture
│   ├── predict.html            # Transaction analysis input form (Mode 1 & Mode 2)
│   ├── result.html             # Classification outcome, risk level, & probabilities
│   └── dashboard.html          # Performance charts, comparison table, & viva notes
│
├── static/
│   ├── css/
│   │   └── style.css           # Modern financial cybersecurity CSS styles
│   ├── js/
│   │   └── script.js           # Client-side input sanity and alert helpers
│   └── img/
│       └── plots/              # Visual plots served to web dashboard
│
└── reports/
    ├── eda_summary.json        # Raw statistical summary numbers
    └── plots/                  # Visual plots generated during EDA & Training
```

---

## 15. Installation & Setup

### Prerequisites
- Python 3.9+ or Python 3.14
- Command-line terminal (PowerShell, Command Prompt, or Bash)

### Step 1: Clone or Navigate to the Project Directory
```powershell
cd d:\credit-card-fraud-detection
```

### Step 2: Install Required Dependencies
```powershell
python -m pip install -r requirements.txt
```

---

## 16. Running EDA & Model Training

### Run Exploratory Data Analysis (EDA)
Generates distribution metrics and statistical charts:
```powershell
python eda.py
```
*Outputs saved to `reports/plots/` and `static/img/plots/`.*

### Train and Evaluate Machine Learning Models
Trains Logistic Regression & Random Forest with balanced class weighting, evaluates on test data, and saves models:
```powershell
python train_model.py
```
*Outputs saved to `models/`.*

---

## 17. Running the Automated Test Suite
To verify dataset integrity, model loading, web routes, and input validation:
```powershell
python test_app.py
```
*Executes 10 automated test cases.*

---

## 18. Launching the Web Application
Start the Flask development server:
```powershell
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 19. How to Use the Web Application
1. **Home Page (`/`):** View overall dataset metrics, system architecture flow, and quick links.
2. **Analyze Transaction (`/predict`):**
   - **Mode 1 (Demo Samples):** Click on any sample button (e.g., *Legitimate Sample A* or *Fraud Sample A*). The form automatically fills all 30 features using genuine records from the test split.
   - **Mode 2 (Manual Input):** Adjust Amount, Time, or any of the 28 PCA components.
   - Click **ANALYZE TRANSACTION**.
3. **Prediction Result (`/predict` POST):**
   - Displays whether the transaction is **LEGITIMATE** or **POTENTIAL FRAUDULENT**.
   - Displays the **Risk Level (LOW / HIGH)** and the **Model Probability Risk Score**.
4. **Dashboard (`/dashboard`):** Inspect model comparison tables, confusion matrix heatmaps, ROC curves, and class imbalance charts.

---

## 20. Limitations
1. **Historical Static Dataset:** Trained on 2013 data; fraud patterns evolve over time as adversaries develop new attack vectors (concept drift).
2. **Anonymized Features:** Because PCA was applied to original attributes, domain-specific feature engineering (e.g., geographical velocity, merchant risk categories) cannot be inspected directly.
3. **Stand-Alone Model:** Operates as an independent point-in-time classifier without real-time stateful user session tracking.

---

## 21. Future Scope
1. **Real-time Velocity Features:** Incorporating sequential transaction velocity (e.g., number of purchases in the last 15 minutes).
2. **Graph Neural Networks (GNN):** Detecting coordinated fraud rings and mule account networks.
3. **Explainable AI (XAI):** Integrating SHAP (SHapley Additive exPlanations) or LIME to provide cardholders with transparent explanations of flagged transactions.
4. **Active Learning Feedback Loop:** Allowing fraud analysts to confirm or dismiss alerts, continuously retraining the model on newly labeled data.

---

## 22. Conclusion
The **FraudGuard** project successfully demonstrates that machine learning models—specifically **Random Forest Classifiers** with balanced class weighting—can effectively detect credit card fraud in highly imbalanced conditions. By coupling a rigorous, leakage-free data science pipeline with a modern, user-friendly Flask interface, the project provides a practical and understandable demonstration of AI applications in financial security.
