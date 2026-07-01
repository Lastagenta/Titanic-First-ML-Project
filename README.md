# 🚢 Titanic: Machine Learning from Disaster

A repository containing my first end-to-end Data Science and Machine Learning project based on the legendary Kaggle competition. The model predicts the survival of passengers on the Titanic using historical data.

## 📊 Leaderboard Result
* **Public Score:** `0.78947` (~79% accuracy)
* **Model:** Random Forest Classifier

---

## 🛠️ Project Pipeline (Data Processing)

The entire project is implemented in a single Jupyter Notebook and consists of the following stages:

1. **Exploratory Data Analysis (EDA):** Analyzing the dataset structure, finding correlations, and identifying missing values (`NaN`).
2. **Data Cleaning & Feature Engineering:**
   * **Age Imputation:** Instead of using a global mean, missing `Age` values were filled using the median age of the passenger's respective ticket class (`Pclass`), which significantly improved accuracy.
   * **Fare Correction:** A single missing `Fare` value in the test set was filled with the global median.
   * **Label Encoding:** The categorical `Sex` feature was mapped to binary numerical values (`{'male': 0, 'female': 1}`).
3. **Data Leakage Prevention:** Both the training (`train.csv`) and testing (`test.csv`) datasets were processed together in a loop to ensure identical mathematical transformations and prevent data leakage.

---

## 🧠 Model Configuration

**Random Forest** from `scikit-learn` was chosen as the baseline algorithm. The model was configured with the following hyperparameters:
```python
RandomForestClassifier(
    n_estimators=100,  # 100 independent decision trees
    max_depth=5,       # Depth limit to prevent overfitting
    random_state=42    # Fixed random seed for reproducibility
)
