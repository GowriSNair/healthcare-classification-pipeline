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

**Why Gradient Boosting?** In a screening context, a **false negative** (telling someone at risk that they're fine) is the most costly error. Gradient Boosting had the **highest recall on the diabetic class (0.863)** — it misses the fewest true cases — while also leading on ROC-AUC (0.707) and F1 (0.766). Logistic Regression, the baseline, had the best precision (0.755) but missed about 40% of diabetic cases (recall 0.596), which is why it was not selected.

**Stability:** 5-fold cross-validation gave a mean ROC-AUC of 0.7074 (std 0.0008), so performance does not depend on one lucky train/test split.

**What drives the predictions:** family history of diabetes, weekly physical activity and age together account for ~86% of the model's feature importance, followed by triglycerides and BMI. Demographic fields (gender, ethnicity, income, education, employment, smoking) received zero importance.

**Hyperparameter tuning:** `RandomizedSearchCV` (5-fold CV, 20 parameter combinations, scored on ROC-AUC) raised the cross-validated ROC-AUC from 0.707 to **0.7255** with:
```
n_estimators: 200, max_depth: 5, learning_rate: 0.2, min_samples_split: 5, subsample: 1.0
```

## Interactive Prediction Interface

A Gradio interface (`app.py`) takes a person's health and lifestyle information and returns a diabetes risk prediction with an associated probability. The app serves the Gradient Boosting model from the model-comparison step (`gb_model.pkl`, test ROC-AUC 0.707).

## Repository Contents

```
├── DiabetesProjectFinal.ipynb   # Full analysis: EDA, hypothesis testing, model training & tuning
├── app.py                       # Gradio interface for interactive predictions
├── gb_model.pkl                 # Trained Gradient Boosting model (scikit-learn 1.6.1)
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

## Limitations & Next Steps

- The dataset is synthetic, so findings illustrate the method rather than real clinical relationships.
- A ROC-AUC of ~0.71 is moderate; the model is a screening aid, not a diagnostic tool.
- Next steps: retrain and save the tuned model and evaluate it on the held-out test set; tune the decision threshold to push recall higher; try XGBoost/LightGBM; host a live demo.

## Notes

- `gb_model.pkl` was saved with scikit-learn 1.6.1, which is pinned in `requirements.txt` — newer versions cannot load it.
- The app runs locally; a hosted live demo may be added in a future update.

## Author

**Gowri S Nair** — Data Analyst · [LinkedIn](https://www.linkedin.com/in/gowri-s-nair-9ba795354) · [GitHub](https://github.com/GowriSNair)
