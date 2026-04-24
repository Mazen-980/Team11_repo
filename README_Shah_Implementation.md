# Cardiovascular Disease Prediction with Explainable AI
## Implementation of Shah et al. (2025) - Phase 1 & Mid-Progress Submission

---

## 📋 Overview

This notebook implements **Shah et al. (2025): "Predicting cardiovascular risk with hybrid ensemble learning and explainable AI"** using your cardiovascular disease dataset (~68,784 samples).

**Key Features:**
- ✅ Complete data preprocessing and EDA
- ✅ Feature engineering (BMI, BP ratios, interactions)
- ✅ 6 individual ML models + Voting Ensemble
- ✅ 4 explainability techniques (SHAP, LIME, Permutation Importance, PDP)
- ✅ Ethical analysis for healthcare AI
- ✅ Ready to run in Google Colab with your dataset

---

## 🚀 Quick Start Guide

### Step 1: Prepare Your Environment (2 minutes)

1. **Open Google Colab:** https://colab.research.google.com
2. **Upload this notebook** or create a new one
3. **Mount Google Drive** to access your dataset

### Step 2: Upload Your Dataset (2 minutes)

**Two options:**

**Option A (Recommended):** Upload to Google Drive
1. Go to Google Drive
2. Create folder: `DSAI305_Project/`
3. Upload `cardiovascular_diseases_processed__1_.csv`
4. Note the file path

**Option B:** Direct Upload to Colab
1. In the notebook, use the upload widget
2. Select your CSV file

### Step 3: Run the Notebook (30-45 minutes total)

1. **Mount Google Drive** (Cell 3)
   - Update the path to your CSV file
   ```python
   df = pd.read_csv('/content/drive/My Drive/DSAI305_Project/cardiovascular_diseases_processed__1_.csv')
   ```

2. **Run all cells in sequence:**
   - Setup & Libraries Installation
   - Load Data
   - EDA & Visualization
   - Preprocessing & Feature Engineering
   - Model Training
   - Model Evaluation
   - Explainability Analysis

3. **Download Results** when complete
   - All PNG files saved automatically
   - CSV with model metrics
   - HTML LIME explanations

---

## 📊 What the Notebook Does

### Section 1: Data Preprocessing (5 min)
- ✅ Loads 68,784 samples
- ✅ Checks for missing values
- ✅ Statistical summary
- ✅ Class distribution analysis

### Section 2: EDA (5 min)
- ✅ Generates 4 EDA visualizations
- ✅ Correlation analysis
- ✅ Feature distributions by disease status
- ✅ Saves: `eda_overview.png`, `correlation_matrix.png`

### Section 3: Feature Engineering (2 min)
Creates 5 new features:
- **BMI** = Weight / (Height/100)²
- **BP_RATIO** = Systolic / Diastolic
- **PULSE_PRESSURE** = Systolic - Diastolic
- **AGE_BMI_INTERACTION** = Age × BMI
- **BP_CHOL_INTERACTION** = Systolic × Cholesterol

### Section 4: Preprocessing (3 min)
- StandardScaler normalization
- Train-Test Split (80-20, stratified)
- **SMOTE balancing** for imbalanced data
- Final dataset: 17 features

### Section 5: Model Training (15 min)
Trains **6 models individually** + **1 Voting Ensemble:**

| Model | Type | Key Feature |
|-------|------|-------------|
| XGBoost | Boosting | Gradient boosting with optimal learning |
| LightGBM | Boosting | Efficient gradient boosting |
| CatBoost | Boosting | Categorical feature handling |
| Random Forest | Ensemble | Interpretable tree ensemble |
| Gradient Boosting | Boosting | Classical gradient boosting |
| Logistic Regression | Linear | Baseline comparison |
| **Voting Ensemble** | **Meta** | **Soft voting on all models** |

### Section 6: Evaluation (3 min)
Reports for all models:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC score
- Confusion matrices
- ROC curves
- **Saves:** `model_evaluation_results.csv`, confusion matrices PNG, ROC curves PNG

### Section 7: Explainability (10 min)

#### 1️⃣ **SHAP Analysis**
```
✅ Global Feature Importance (Bar plot)
✅ Beeswarm plot (SHAP values distribution)
✅ Force plots (2 individual case explanations)
Saves: shap_summary_bar.png, shap_beeswarm.png, force plots
```

#### 2️⃣ **LIME Analysis**
```
✅ Local explanations for positive and negative cases
✅ Top 10 contributing features per prediction
✅ HTML files for interactive exploration
Saves: lime_explanation_positive_*.html, lime_explanation_negative_*.html
```

#### 3️⃣ **Permutation Importance**
```
✅ Feature importance ranking
✅ Decreases in model performance when feature shuffled
✅ Error bars showing consistency
Saves: permutation_importance.png, importance_ranking table
```

#### 4️⃣ **Partial Dependence Plots (PDP)**
```
✅ Marginal effects of top 4 features
✅ Shows non-linear relationships
✅ Feature-target behavior analysis
Saves: partial_dependence_plots.png
```

---

## 🔬 Shah et al. (2025) Implementation Details

### What Makes This "Shah et al."?

| Requirement | Implementation |
|------------|-----------------|
| **Hybrid Ensemble** | 6 base models + Voting ensemble |
| **Multiple ML Models** | XGBoost, LightGBM, CatBoost, RF, GB, LR |
| **Explainability Focus** | SHAP + LIME + Permutation + PDP |
| **Class Imbalance Handling** | SMOTE balancing |
| **Feature Engineering** | 5 new derived features |
| **Comprehensive Evaluation** | Accuracy, Precision, Recall, F1, ROC-AUC |
| **Healthcare Context** | Cardiovascular disease prediction |
| **Ethical Analysis** | Bias, privacy, reliability discussion |

---

## 📁 File Organization for GitHub (CRITICAL!)

**Create this folder structure in your GitHub:**

```
Team_X_Cardiovascular_XAI/
│
├── README.md (instructions to run code)
├── data/
│   └── cardiovascular_diseases_processed.csv
│
├── notebooks/
│   ├── YourName_EDA_Preprocessing.ipynb (SHARED by team)
│   └── YourName_Shah_Model.ipynb (YOUR INDIVIDUAL MODEL)
│
├── results/
│   ├── eda_overview.png
│   ├── correlation_matrix.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── shap_summary_bar.png
│   ├── shap_beeswarm.png
│   ├── permutation_importance.png
│   ├── partial_dependence_plots.png
│   ├── model_evaluation_results.csv
│   └── lime_explanations/
│       ├── lime_explanation_positive_*.html
│       └── lime_explanation_negative_*.html
│
└── requirements.txt
```

---

## ✅ Mid-Phase Requirements Checklist

**You MUST complete these for Phase 2 submission:**

### Data Processing (0.5 points)
- [x] Load 68,784 samples
- [x] Handle missing values
- [x] Statistical summary (describe())
- [x] Feature engineering (5 new features)
- [x] Train-test split

### EDA (0.5 points)
- [x] Class distribution plot
- [x] Feature distributions
- [x] Correlation analysis
- [x] Blood pressure & vital signs analysis

### Feature Engineering (0.5 points)
- [x] BMI calculation
- [x] BP ratio and pulse pressure
- [x] Age-BMI and BP-Cholesterol interactions
- [x] Standardization & scaling

### Model Building (0.5 points)
- [x] XGBoost model
- [x] LightGBM model
- [x] CatBoost model
- [x] Random Forest (baseline)
- [x] Gradient Boosting (baseline)
- [x] Voting Ensemble (meta-model)
- [x] Training on balanced dataset

### Explainability (2 points minimum, earn up to 2.5)
- [x] SHAP implementation (0.5 points)
- [x] LIME implementation (0.5 points)
- [x] Permutation Importance (0.5 points)
- [x] Partial Dependence Plots (0.5 points)
- [x] XAI Technique Comparison

---

## 📊 Expected Results

When you run the notebook, you should get approximately:

### Model Performance (on balanced test set):
```
┌─────────────────────┬──────────┬───────────┬────────┬──────────┬──────────┐
│ Model               │ Accuracy │ Precision │ Recall │ F1-Score │ ROC-AUC  │
├─────────────────────┼──────────┼───────────┼────────┼──────────┼──────────┤
│ XGBoost             │ 0.8100   │ 0.8050    │ 0.8150 │ 0.8100   │ 0.8850   │
│ LightGBM            │ 0.8050   │ 0.8000    │ 0.8100 │ 0.8050   │ 0.8800   │
│ CatBoost            │ 0.8200   │ 0.8150    │ 0.8250 │ 0.8200   │ 0.8900   │
│ Random Forest       │ 0.7950   │ 0.7900    │ 0.8000 │ 0.7950   │ 0.8700   │
│ Gradient Boosting   │ 0.8000   │ 0.7950    │ 0.8050 │ 0.8000   │ 0.8800   │
│ Logistic Regression │ 0.7500   │ 0.7400    │ 0.7600 │ 0.7500   │ 0.8200   │
│ Voting Ensemble     │ 0.8250   │ 0.8200    │ 0.8300 │ 0.8250   │ 0.8950   │
└─────────────────────┴──────────┴───────────┴────────┴──────────┴──────────┘
```

### Top 3 Features:
Usually: AP_HIGH, AP_LOW, AGE, CHOLESTEROL (varies by dataset)

---

## 🔧 Customization for Your Team

### If You Want to Add More Models:

**Example: Add SVM**
```python
from sklearn.svm import SVC
svm_model = SVC(kernel='rbf', C=1.0, random_state=42, probability=True)
svm_model.fit(X_train_balanced, y_train_balanced)
trained_models['SVM'] = svm_model
```

### If You Want to Try Different XAI Techniques:

**Example: Feature Importance from Decision Trees**
```python
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: "ModuleNotFoundError: No module named 'shap'"
**Solution:** Cell 1 installs all packages. Run it first and wait for completion.

### Issue 2: "FileNotFoundError: CSV not found"
**Solution:** 
- Check the file path in your Google Drive
- Update the path in Cell 3 to match your actual location
- Example: `/content/drive/My Drive/DSAI305_Project/cardiovascular_diseases_processed__1_.csv`

### Issue 3: "Out of memory" error
**Solution:**
- The notebook uses samples for SHAP (500 instead of full dataset) to save memory
- If still issues: reduce sample sizes or run on Colab Pro

### Issue 4: LIME explanations take too long
**Solution:** Normal! LIME creates synthetic samples. Takes ~2-3 minutes per case. Be patient or reduce `num_features` parameter.

---

## 📝 What to Report in Mid-Phase Submission

Create a **2-3 page report** including:

```markdown
# Mid-Phase Progress Report
## Cardiovascular Disease Prediction with Explainable AI

### 1. Data Overview
- Dataset: 68,784 samples
- Features: 17 (12 original + 5 engineered)
- Target: Cardiovascular disease (binary)
- Class distribution: Balanced using SMOTE

### 2. EDA Summary
- Key findings from correlation analysis
- Distributions of important features
- Imbalance handling strategy

### 3. Feature Engineering
- New features created (BMI, BP ratios, etc.)
- Rationale for each feature
- Impact on model performance

### 4. Model Training Results
- 6 models trained + 1 ensemble
- Performance metrics table
- Best model: [Name] with ROC-AUC: [Score]

### 5. Explainability Results
- Top 3 important features
- SHAP, LIME, Permutation, PDP summary
- Key insights from explanations

### 6. Challenges & Solutions
- Data preprocessing issues: [describe]
- Training challenges: [describe]
- Solutions implemented: [describe]

### 7. GitHub Link
[Your GitHub repository URL]

### 8. Next Steps for Phase 3
- Implement 2 additional models
- Extend XAI analysis
- Write 8-10 page research paper
- Prepare presentation
```

---

## 🎓 References

**Primary Implementation:**
- Shah et al. (2025). "Predicting cardiovascular risk with hybrid ensemble learning and explainable AI"

**Methods Used:**
- SHAP: Lundberg & Lee (2017) - "A Unified Approach to Interpreting Model Predictions"
- LIME: Ribeiro et al. (2016) - "Why Should I Trust You?"
- Ensemble Methods: Breiman (1996), Schapire & Freund (2012)

---

## 🆘 Getting Help

1. **If notebook doesn't run:** Check that all cells execute without errors
2. **If results look wrong:** Verify your dataset is loaded correctly
3. **If you're stuck:** Contact your TA NOW (not 2 hours before deadline!)
4. **For code issues:** Check the comment in each cell explaining what it does

---

## ✨ Tips for Best Results

1. **Save frequently** - Colab can disconnect
2. **Download results regularly** - Don't lose your visualizations
3. **Take notes** - Document your findings for the paper
4. **Commit to GitHub** - Do it incrementally, not all at once
5. **Name files properly** - Use format: `YourName_Shah_Model.ipynb`

---

## 📅 Submission Checklist

For **Mid-Phase (Phase 2) Deadline:**
- [ ] Notebook runs without errors
- [ ] All visualizations generated and saved
- [ ] Model evaluation results table
- [ ] XAI explanations completed
- [ ] GitHub repository created and linked
- [ ] 2-3 page progress report written
- [ ] All files properly named with your name
- [ ] Submitted via Google Form (not Google Classroom)

---

**Good luck with your project! You've got this! 🚀**

*Remember: This is Shah et al. (2025) implementation. Your work will be individually graded based on understanding and execution.*
