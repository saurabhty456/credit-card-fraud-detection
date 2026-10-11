# FRAUDGUARD: College Viva Voce Defense Guide

> **Cheat-Sheet & Comprehensive Q&A for Viva Examiners and Project Defense**

---

### Q1: What is Credit Card Fraud Detection?
**Answer:**  
Credit card fraud detection is the process of identifying unauthorized, fraudulent transactions made using payment cards so that charges can be blocked before completion or flagged for immediate verification. In modern banking, this is automated using machine learning models that evaluate behavioral anomalies and historical patterns.

---

### Q2: What is Supervised Learning and why is this problem Supervised?
**Answer:**  
Supervised learning is a branch of machine learning where the algorithm learns a mapping function from input features ($X$) to known ground-truth target labels ($y$).  
This project is **supervised** because each historical transaction in our dataset is already labeled with its true outcome:
- `0` for Legitimate
- `1` for Fraudulent

---

### Q3: Why is this a Classification problem rather than Regression?
**Answer:**  
- **Regression** predicts continuous numerical values (e.g., predicting the stock price or transaction amount).
- **Classification** predicts discrete categorical classes. Here, we are deciding between two distinct categories: **Legitimate vs. Fraudulent** (Binary Classification).

---

### Q4: What is Class Imbalance and why is it severe in this dataset?
**Answer:**  
Class imbalance occurs when one class significantly outnumbers another. In our dataset:
- Legitimate transactions: **284,315** (99.827%)
- Fraudulent transactions: **492** (0.173%)

That is an imbalance ratio of **1 fraud for every 577 legitimate transactions**.  
If an examiner asks: *"What happens if you use standard algorithms without addressing this?"*  
**Answer:** The algorithm will prioritize the majority class to minimize overall error, predicting every transaction as legitimate. It would achieve 99.83% accuracy while catching zero fraud cases!

---

### Q5: Why do we use a Train/Test split?
**Answer:**  
A train/test split evaluates how well a model generalizes to **unseen data**. If we evaluate the model on the exact same data it was trained on, it could simply memorize the inputs (overfitting). We split the dataset into:
- **80% Training Set:** Used by the model to learn parameters and patterns.
- **20% Test Set:** Kept isolated and used strictly for final, unbiased performance evaluation.

---

### Q6: Why is `stratify=y` critical during the train/test split?
**Answer:**  
Because fraud occurs in only 0.17% of transactions, a purely random split could accidentally place too few fraud cases in the test set or train set. Setting `stratify=y` ensures that both the training and test sets maintain the **exact same 0.17% proportion of fraud**, guaranteeing realistic evaluation.

---

### Q7: Why do we use Feature Scaling (`StandardScaler`) and how do we prevent Data Leakage?
**Answer:**  
- **Why scale:** Features like `Time` (up to 172,792) and `Amount` (up to €25,691) have much larger magnitudes than PCA features `V1`–`V28` (which have mean ~0 and variance ~1). Without scaling, algorithms like Logistic Regression would artificially assign oversized weights to features with larger numeric scales.
- **Preventing Data Leakage:** We fit `StandardScaler` **only on the training split** (`scaler.fit_transform(X_train)`). We then use those exact learned parameters ($\mu$ and $\sigma$) to transform the test set and live website inputs (`scaler.transform(X_test)`). If we scaled before splitting, information from the test set would "leak" into the training phase, giving falsely optimistic results.

---

### Q8: What is Logistic Regression and how does it work here?
**Answer:**  
Logistic Regression is a linear classification algorithm that computes a linear combination of input features and passes it through the **sigmoid (logistic) function**:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
This maps any real value into a probability between $0$ and $1$. If the probability exceeds $0.5$ (or a specified threshold), it predicts Class 1 (Fraud).  
In our project, we enabled `class_weight='balanced'`, which penalizes errors on the rare fraud class much more heavily.

---

### Q9: What is Random Forest and why did it perform better?
**Answer:**  
Random Forest is an **ensemble learning method** based on bagging (Bootstrap Aggregating). It constructs multiple decision trees (in our project, 100 trees) on random subsets of data and features. Each tree votes on the class, and the majority vote wins.  
**Why it outperformed Logistic Regression:**
- Credit card fraud involves complex, non-linear relationships (e.g., specific combinations of PCA dimensions and amounts).
- While Logistic Regression drew a rigid linear boundary (causing 1,386 false alarms), Random Forest captures non-linear decision boundaries, reducing false alarms to just 32 while catching 82.65% of frauds.

---

### Q10: Why are Precision and Recall more important than Accuracy?
**Answer:**  
- **Accuracy** is misleading on imbalanced datasets (a dummy model that always predicts legitimate gets 99.83% accuracy).
- **Recall (Sensitivity):** $\frac{TP}{TP + FN}$  
  Measures: *"Of all actual frauds, what percentage did we catch?"*  
  High recall is vital because missing a fraud (False Negative) costs money.
- **Precision:** $\frac{TP}{TP + FP}$  
  Measures: *"When the model raises an alarm, how often is it right?"*  
  High precision prevents blocking legitimate customers unnecessarily.
- **F1-Score:** The harmonic mean balancing both Precision and Recall.

---

### Q11: What is a Confusion Matrix?
**Answer:**  
A $2 \times 2$ matrix summarizing classification performance:
1. **True Positives (TP):** Actual fraud correctly flagged as fraud (81 for Random Forest).
2. **True Negatives (TN):** Legitimate transaction correctly identified as legitimate (56,832).
3. **False Positives (FP) [Type I Error]:** Legitimate transaction incorrectly flagged as fraud (32).
4. **False Negatives (FN) [Type II Error]:** Actual fraud that slipped through undetected (17).

---

### Q12: What is ROC-AUC?
**Answer:**  
- **ROC (Receiver Operating Characteristic):** A curve plotting True Positive Rate (Recall) vs False Positive Rate ($1 - \text{Specificity}$) across all classification thresholds from 0 to 1.
- **AUC (Area Under the Curve):** A single metric summarizing the ROC curve. A random guess has an AUC of $0.50$, while a perfect classifier has an AUC of $1.00$. Our Random Forest achieved **0.9790**, demonstrating high discriminative power.

---

### Q13: Why did you choose Flask for the Backend?
**Answer:**  
Flask is a lightweight, Python-native micro-framework. Because the machine learning models were developed in Python using Scikit-learn, Flask enables direct, seamless in-memory loading of the serialized `.pkl` models without requiring cross-language bridges or heavy server dependencies.

---

### Q14: How does the model connect to the web application?
**Answer:**  
1. In `train_model.py`, the trained `RandomForestClassifier` and `StandardScaler` are saved to disk using `joblib.dump()`.
2. In `app.py`, these files are loaded once at server startup using `joblib.load()`.
3. When a user submits transaction parameters on `/predict`:
   - Flask validates that all 30 fields are numeric.
   - It packages them into a Pandas DataFrame with exact column names.
   - The saved `scaler.transform()` normalizes the input.
   - `model.predict()` determines the class (0 or 1) and `model.predict_proba()` calculates the risk score.
   - Flask passes this result to `result.html` for rendering.

---

### Q15: What are the limitations of this project?
**Answer:**  
1. **PCA Anonymization:** The raw attributes were pre-transformed with PCA, preventing domain-specific manual feature engineering like merchant categorization or physical card proximity.
2. **Static Historical Data:** Fraud tactics change over time (concept drift). In production, models must be periodically retrained with active learning feedback loops.
3. **Point-in-Time Inference:** Real banking systems examine multi-transaction velocity (e.g., 5 transactions in 2 minutes across different countries).
