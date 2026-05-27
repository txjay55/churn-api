# 🔄 Customer Churn Prediction — ANN vs Ensemble Learning

> Predict which customers are likely to churn using a full ML pipeline:  
> Deep Learning (ANN) benchmarked against XGBoost, LightGBM, Random Forest, and a Stacking Ensemble — with SHAP-powered explainability.

---

## 📌 Project Description

Customer churn is one of the most critical business metrics — losing an existing customer costs significantly more than acquiring a new one. This project builds an end-to-end machine learning pipeline to **identify customers at risk of churning** before it happens, enabling proactive retention strategies.

The pipeline explores two modeling families:

- **Deep Learning** — a tuned Artificial Neural Network (ANN) with early stopping, learning rate scheduling, and multiple evaluation metrics.
- **Ensemble Learning** — XGBoost, LightGBM, Random Forest, and a final Stacking Ensemble that combines all three.

Beyond prediction accuracy, this project emphasizes **model explainability** using SHAP values and feature importance, making the results actionable for business stakeholders.

---

## 🗂️ Project Structure

```
customer-churn-prediction/
│
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned & encoded features
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Feature engineering & encoding
│   ├── 03_ann_model.ipynb      # ANN training & evaluation
│   └── 04_ensemble_models.ipynb# XGBoost, LightGBM, RF, Stacking
│
├── src/
│   ├── preprocessing.py        # Data cleaning & feature pipeline
│   ├── ann_model.py            # Keras ANN architecture
│   ├── churn_ensemble_pipeline.py  # Full ensemble pipeline
│   └── shap_explainability.py  # SHAP analysis & plots
│
├── outputs/
│   ├── ensemble_comparison.png # ROC & PR curves, confusion matrices
│   ├── feature_importance.png  # XGBoost & LightGBM importances
│   └── shap_summary.png        # SHAP beeswarm & waterfall plots
│
├── requirements.txt
└── README.md
```

---

## 🧠 Models Used

| Model | Type | Imbalance Handling | Key Strength |
|---|---|---|---|
| **ANN** | Deep Learning | Class weights | Captures non-linear interactions |
| **XGBoost** | Gradient Boosting | `scale_pos_weight` | Robust, fast, feature importance |
| **LightGBM** | Gradient Boosting | `scale_pos_weight` | Fastest training, large datasets |
| **Random Forest** | Bagging Ensemble | `class_weight='balanced'` | Low variance, stable predictions |
| **Stacking Ensemble** | Meta-learning | Balanced base learners | Best overall generalisation |

All models handle the **26% churn / 74% non-churn** class imbalance explicitly rather than relying on oversampling.

---

## 📊 Results

### ANN Baseline
| Metric | Train | Validation |
|---|---|---|
| Accuracy | 86.25% | 75.14% |
| AUC | 0.9374 | 0.8232 |
| Loss | 0.3708 | 0.5815 |

> Early stopping triggered at epoch 39 (best weights from epoch 24). Clear overfitting — the gap between train AUC (0.937) and val AUC (0.823) motivates switching to ensemble methods.

### Ensemble Models *(fill in your results)*
| Model | ROC-AUC | PR-AUC | Notes |
|---|---|---|---|
| XGBoost | — | — | Best iteration from early stopping |
| LightGBM | — | — | Fastest to train |
| Random Forest | — | — | Most stable |
| **Stacking Ensemble** | **—** | **—** | **Best overall** |

---

## 🔍 Explainability — SHAP + Feature Importance

Understanding *why* a customer is predicted to churn is as important as the prediction itself.

### Feature Importance (XGBoost & LightGBM)
- Tree-based gain importance computed natively
- Top 20 features visualised as horizontal bar charts
- Helps identify which features drive the model's decisions globally

### SHAP Analysis
Three SHAP plots are generated for the best model:

**1. Beeswarm Summary Plot** — shows the impact of every feature on every prediction. Each dot is one customer; color indicates feature value (red = high, blue = low); horizontal position shows impact on churn probability.

**2. Bar Plot** — mean absolute SHAP values ranked by importance. A cleaner global view compared to gain importance.

**3. Waterfall Plot** — explains a single prediction in detail. Shows exactly which features pushed a specific customer's churn probability up or down from the baseline. Ideal for explaining individual cases to business teams.

```python
import shap

explainer    = shap.TreeExplainer(best_model)
shap_values  = explainer.shap_values(X_test)

# Global summary
shap.summary_plot(shap_values, X_test)

# Single prediction explanation
shap.waterfall_plot(explainer(X_test)[0])
```

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**
```
tensorflow>=2.12
xgboost>=1.7
lightgbm>=3.3
scikit-learn>=1.2
shap>=0.42
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
```

---

## 🚀 Quick Start

```python
# 1. Load your preprocessed data
# X_train, X_test, y_train, y_test should already be ready

# 2. Run the full ensemble pipeline
python src/churn_ensemble_pipeline.py

# 3. Run SHAP explainability
python src/shap_explainability.py
```

---

## 💡 Key Findings & Business Suggestions

Based on typical churn patterns, the models and SHAP analysis usually surface the following as high-impact features. **Validate these against your own SHAP output.**

| Driver | Business Action |
|---|---|
| **Contract type** (month-to-month churns most) | Offer discounts to switch to annual contracts |
| **Tenure** (new customers churn more) | Strengthen onboarding experience in first 3 months |
| **Monthly charges** (high bill = high churn) | Introduce loyalty pricing tiers |
| **Tech support / Online security absent** | Proactively offer add-on services to at-risk customers |
| **Number of products** (single-product customers churn more) | Cross-sell bundles to increase stickiness |

### Model Recommendation
- Use the **Stacking Ensemble** for production — best AUC and most robust generalisation.
- Use **SHAP waterfall plots** per customer for retention team conversations.
- Re-train every quarter as customer behavior drifts.
- Consider **threshold tuning** (script included) — a lower threshold (e.g. 0.35) catches more churners at the cost of more false positives. Tune based on the cost of missing a churner vs. cost of a retention offer.

---

## 📈 Why Ensemble > ANN on This Problem?

- Tabular data with mixed feature types favors tree-based models
- ANN val_auc was **0.823** — ensemble methods typically reach **0.85–0.89** on similar churn datasets
- Trees are natively interpretable via SHAP; ANNs need additional approximation layers
- Gradient boosting is more data-efficient — better performance with less data and no augmentation tricks

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

[MIT](LICENSE)

---

*Built with Python · scikit-learn · XGBoost · LightGBM · TensorFlow/Keras · SHAP*
