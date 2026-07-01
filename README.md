# Diabetes Risk Classification Pipeline

A machine learning pipeline that predicts diabetes risk from health, lifestyle, and demographic data, with an interactive Gradio interface for real-time predictions.

## Overview

This project analyzes a synthetic health dataset of over 600,000 individuals to identify key risk factors for diabetes and build a classification model that predicts diagnosis likelihood. The pipeline covers data cleaning, exploratory analysis, statistical hypothesis testing, model comparison, hyperparameter tuning, and deployment via a Gradio web interface.

## Dataset

- **Size:** 608,567 records, 26 features
- **Target variable:** `diagnosed_diabetes` (binary classification)
- **Features:** Demographics (age, gender, ethnicity, education, income, employment), lifestyle factors (physical activity, sleep, alcohol consumption, screen time, diet score), and clinical measurements (BMI, blood pressure, cholesterol panel, triglycerides, heart rate, waist-to-hip ratio), plus medical history flags (family history, hypertension, cardiovascular history)

## Exploratory Analysis & Statistical Testing

Several hypothesis tests were run to validate relationships between risk factors and diabetes diagnosis:

| Factor | Test | Result |
|---|---|---|
| BMI | t-test | Significant difference (diabetic vs. non-diabetic), medium effect size |
| Family History | Chi-square | Diabetes rate rises from 58.1% (no history) to 86.8% (with history) |
| Physical Activity | t-test | Diabetic group averages 18 fewer minutes/week of activity |
| Hypertension History | Chi-square | Statistically significant association |
| Triglycerides & HDL Cholesterol | t-test (Bonferroni-corrected) | Both significant after correction |

All tests rejected the null hypothesis at p < 0.05, confirming these factors are meaningfully associated with diabetes diagnosis in this dataset.

## Model Development

Four classification models were trained and compared:

| Model | Accuracy | ROC-AUC | Recall (Diabetic) | Precision (Diabetic) | F1 (Diabetic) |
|---|---|---|---|---|---|
| Logistic Regression | 0.628 | 0.695 | 0.596 | 0.755 | 0.666 |
| Decision Tree | 0.592 | 0.565 | 0.675 | 0.672 | 0.673 |
| Random Forest | 0.664 | 0.694 | 0.861 | 0.683 | 0.761 |
| **Gradient Boosting** | **0.671** | **0.707** | **0.863** | **0.689** | **0.766** |

**Gradient Boosting** was selected as the best-performing model and tuned further using `RandomizedSearchCV` (5-fold cross-validation, 20 parameter combinations, scored on ROC-AUC).

**Best parameters found:**
```
subsample: 1.0
n_estimators: 200
min_samples_split: 5
max_depth: 5
learning_rate: 0.2
```
**Best cross-validated ROC-AUC:** 0.7255

## Interactive Prediction Interface

The final tuned model is served through a Gradio interface (`app.py`) that takes in a user's health and lifestyle information and returns a diabetes risk prediction with an associated probability.

## Repository Contents

```
├── DiabetesProjectFinal.ipynb   # Full analysis: EDA, hypothesis testing, model training & tuning
├── app.py                       # Gradio interface for interactive predictions
├── gb_model.pkl                 # Trained Gradient Boosting model
├── model_columns.pkl            # Encoded feature columns used for alignment at inference time
├── requirements.txt             # Python dependencies
```

## How to Run

1. Clone the repository
   ```
   git clone https://github.com/GowriSNair/healthcare-classification-pipeline.git
   cd healthcare-classification-pipeline
   ```
2. Install dependencies
   ```
   pip install -r requirements.txt
   ```
3. Launch the Gradio app
   ```
   python app.py
   ```
4. Open the local URL shown in the terminal (typically `http://127.0.0.1:7860`) to use the interface

## Tools & Techniques

- **Languages/Libraries:** Python, Pandas, NumPy, Scikit-learn, SciPy, Gradio, Matplotlib, Seaborn
- **Statistical Methods:** Independent t-tests, Chi-square tests of independence, Bonferroni correction, Cohen's d, Cramér's V
- **Modelling:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, RandomizedSearchCV hyperparameter tuning
- **Evaluation Metrics:** Accuracy, ROC-AUC, Precision, Recall, F1-score

## Notes

- The dataset used is synthetic and intended for educational/portfolio purposes.
- The app currently runs locally; a hosted live demo may be added in a future update.
