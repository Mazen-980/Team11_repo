# QUICK REFERENCE CARD
## Shah et al. (2025) Implementation - DSAI 305 Project

---

## 🎯 YOUR ASSIGNMENT

**IMPLEMENT:** Shah et al. (2025) - Hybrid Ensemble Learning for Cardiovascular Disease Prediction

**DATASET:** cardiovascular_diseases_processed__1__.csv (68,784 samples)

**DEADLINE (Mid-Phase):** April 17, 2026

**DELIVERABLES:**
- ✅ Colab notebook (ready to run)
- ✅ All 4 XAI techniques implemented
- ✅ Model evaluation results
- ✅ GitHub repository link
- ✅ 2-3 page progress report

---

## 📊 MODELS TO IMPLEMENT (Choose 1 main model for mid-phase)

| Model | Implementation | Status |
|-------|-----------------|--------|
| **XGBoost** | `xgb.XGBClassifier(n_estimators=200)` | ✅ READY |
| **LightGBM** | `lgb.LGBMClassifier(n_estimators=200)` | ✅ READY |
| **CatBoost** | `CatBoostClassifier(iterations=200)` | ✅ READY |
| Random Forest | `RandomForestClassifier(n_estimators=200)` | ✅ READY |
| Gradient Boosting | `GradientBoostingClassifier(n_estimators=200)` | ✅ READY |
| Logistic Regression | `LogisticRegression(max_iter=1000)` | ✅ READY |
| **Voting Ensemble** | `VotingClassifier(..., voting='soft')` | ✅ READY |

---

## 🔍 XAI TECHNIQUES REQUIRED (Implement all 4)

### 1️⃣ SHAP
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)  # Bar plot
shap.summary_plot(shap_values, X_test)  # Beeswarm plot
shap.force_plot(...)                    # Individual case
```
**Output:** 3 PNG files + insights on feature importance

### 2️⃣ LIME
```python
import lime.lime_tabular
explainer = lime.lime_tabular.LimeTabularExplainer(X_train, ...)
exp = explainer.explain_instance(X_test[i], model.predict_proba)
exp.save_to_file('explanation.html')
```
**Output:** 2 HTML files + local explanations

### 3️⃣ Permutation Importance
```python
from sklearn.inspection import permutation_importance
perm = permutation_importance(model, X_test, y_test, n_repeats=10)
# Create DataFrame and visualize as bar plot
```
**Output:** 1 PNG file + ranking table

### 4️⃣ Partial Dependence Plots (PDP)
```python
from sklearn.inspection import partial_dependence, plot_partial_dependence
plot_partial_dependence(model, X_test, [0, 1, 2, 3])
```
**Output:** 1 PNG file with 4 subplots

---

## 🚀 STEP-BY-STEP EXECUTION

### Step 1: Open Google Colab (2 min)
```
https://colab.research.google.com
```

### Step 2: Upload Notebook
- Open the Colab notebook provided
- OR paste code into new Colab cell

### Step 3: Mount Google Drive (1 min)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 4: Load Your Data (1 min)
```python
df = pd.read_csv('/content/drive/My Drive/DSAI305_Project/cardiovascular_diseases_processed__1_.csv')
print(f"Shape: {df.shape}")  # Should be (68784, 12)
```

### Step 5: Run EDA (5 min)
```
Cell 4: Load Data
Cell 5: Statistical Summary
Cell 6: Visualization
Cell 7: Correlation Analysis
```

### Step 6: Preprocessing (3 min)
```
Cell 8: Separate X and y
Cell 9: Feature Engineering (add 5 new features)
Cell 10: Train-Test Split
Cell 11: Scaling (StandardScaler)
Cell 12: SMOTE Balancing
```

### Step 7: Train Models (15 min)
```
Cell 13: Define Individual Models (6 models)
Cell 14: Train Models
Cell 15: Create Voting Ensemble
```

### Step 8: Evaluate (3 min)
```
Cell 16: Calculate Metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
Cell 17: Confusion Matrices
Cell 18: ROC Curves
```

### Step 9: SHAP (3 min)
```
Cell 19: SHAP Summary Bar Plot
Cell 20: SHAP Beeswarm Plot
Cell 21: SHAP Force Plots (2 cases)
```

### Step 10: LIME (5 min)
```
Cell 22: Initialize LIME Explainer
Cell 23: Explain Positive Case
Cell 24: Explain Negative Case
```

### Step 11: Permutation Importance (2 min)
```
Cell 25: Calculate & Visualize
Cell 26: Create Ranking Table
```

### Step 12: Partial Dependence (2 min)
```
Cell 27: Compute and Visualize Top 4 Features
```

### Step 13: Download Results (2 min)
```
Download all PNG files
Download CSV with metrics
Download HTML files
```

**TOTAL TIME: ~45 minutes**

---

## 📁 FILES YOU'LL GENERATE

### Visualization Files (Save these!)
```
eda_overview.png                    # EDA plots (4 subplots)
correlation_matrix.png              # Heatmap
confusion_matrices.png              # 6 confusion matrices
roc_curves.png                      # ROC curves for all models
shap_summary_bar.png                # Feature importance (bar)
shap_beeswarm.png                   # Feature importance (beeswarm)
permutation_importance.png          # Feature ranking
partial_dependence_plots.png        # 4 PDP subplots
```

### Data Files
```
model_evaluation_results.csv        # Metrics table
lime_explanation_positive_*.html    # Individual explanation
lime_explanation_negative_*.html    # Individual explanation
```

---

## ✅ QUALITY CHECKLIST

### Before Submitting:

- [ ] All 7 sections completed (EDA, Preprocessing, Feature Eng, Models, Eval, XAI, Ethics)
- [ ] 6 models trained individually
- [ ] Voting Ensemble created
- [ ] All 4 XAI techniques implemented
- [ ] Model metrics in results table
- [ ] At least 8 visualizations generated
- [ ] Notebook runs without errors
- [ ] GitHub repository created
- [ ] Files properly named with your name
- [ ] Progress report written (2-3 pages)

### Performance Targets (Rough Expectations):

| Metric | Target | Acceptable |
|--------|--------|------------|
| Ensemble Accuracy | 80%+ | 75%+ |
| Ensemble Recall | 80%+ | 75%+ |
| ROC-AUC | 85%+ | 80%+ |
| Code runs without error | Yes | Essential |
| All 4 XAI techniques | Yes | Essential |

---

## 🆘 TROUBLESHOOTING QUICK FIXES

| Problem | Solution |
|---------|----------|
| "Module not found" error | Run cell 1 (pip install -q ...) first |
| CSV not found | Check path: `/content/drive/My Drive/...` |
| SHAP takes too long | Use smaller sample (already set to 500) |
| LIME very slow | Normal! Wait 2-3 minutes |
| Out of memory | Use Colab Pro or reduce sample sizes |
| Plots don't show | Add `plt.show()` after each plot |
| SMOTE error | Ensure y_train has both classes (0 and 1) |

---

## 📝 WHAT TO WRITE IN PROGRESS REPORT

### Structure (2-3 pages):

**Page 1: Executive Summary**
- Problem: CVD prediction
- Dataset: 68,784 samples
- Approach: Ensemble + XAI
- Results: Best model accuracy

**Page 2: Technical Summary**
- EDA findings (class distribution, feature distributions)
- Feature engineering (5 new features created)
- Preprocessing (SMOTE balancing)
- Model performance table

**Page 3: XAI & Findings**
- Top 3 important features
- SHAP vs LIME vs Permutation vs PDP comparison
- Key clinical insights
- Challenges encountered

**Appendix:**
- GitHub link
- All visualization file names
- Next steps for Phase 3

---

## 🔗 IMPORTANT LINKS

**Colab:** https://colab.research.google.com

**Your Data:** [Your Google Drive Path]

**GitHub:** [Your Repository Link]

**TA Email:** [From your course]

**Submission Form:** [From course Google Classroom]

---

## ⏰ TIMELINE

| Phase | Deadline | Task |
|-------|----------|------|
| Phase 1 | 3/20/2026 | ✅ Literature Review (DONE) |
| **Phase 2** | **4/17/2026** | 🔴 Mid-Progress (THIS ONE!) |
| Phase 3 | 5/2/2026 | Final submission & presentation |

**DO NOT WAIT:** Start now, commit to GitHub daily!

---

## 💡 PRO TIPS FOR SUCCESS

1. **Commit to GitHub frequently**
   ```bash
   git add .
   git commit -m "Added SHAP analysis"
   git push origin main
   ```

2. **Save your work constantly** - Colab can disconnect
   
3. **Test one cell at a time** - Don't run everything at once
   
4. **Document your findings** - Take notes for the paper
   
5. **Keep model results CSV** - You'll need it for the paper
   
6. **Name files properly** - Critical for grading:
   ```
   YourName_Shah_Model.ipynb
   NOT: my_model.ipynb or Shah_Model.ipynb
   ```

7. **Get feedback early** - Don't wait until deadline
   
8. **Implement incrementally:**
   - Week 1: EDA + Preprocessing
   - Week 2: Model training
   - Week 3: XAI + Results

---

## 🎓 LEARNING OUTCOMES

By completing this project, you'll understand:

✅ **Machine Learning:**
- Ensemble methods (voting, stacking)
- Hyperparameter tuning
- Class imbalance handling (SMOTE)
- Model evaluation metrics

✅ **Explainable AI:**
- SHAP values and Shapley additive explanations
- LIME local approximations
- Feature importance interpretation
- Model transparency vs accuracy trade-offs

✅ **Healthcare AI:**
- Clinical data preprocessing
- Ethical considerations
- Regulatory compliance (HIPAA-like)
- Interpretability as safety requirement

✅ **Software Engineering:**
- Git version control
- Code documentation
- Reproducible research
- Professional development practices

---

## 🎁 BONUS OPPORTUNITY (3% Extra)

**Advanced Research Excellence Bonus:**

Implement EITHER:
1. **Multi-modal ensemble** (combine tabular + imaging if available)
2. **Novel feature engineering** (better than standard features)
3. **Comparative XAI analysis** (show strengths/weaknesses of each technique)
4. **Clinical validation** (compare predictions to clinician benchmarks)

If you do this, mention in your progress report!

---

## 🚀 YOU'RE READY!

**You have everything you need:**
✅ Complete Colab notebook (ready to run)
✅ README with instructions
✅ GitHub setup guide
✅ Quick reference card (this document)

**Next Actions:**
1. Open the Colab notebook
2. Mount Google Drive
3. Load your dataset
4. Run the notebook section by section
5. Download results
6. Create GitHub repository
7. Write progress report
8. Submit via Google Form

---

**REMEMBER:**
- Start NOW, don't wait
- Commit to GitHub daily
- Ask for help early
- Follow the naming conventions
- Submit on time!

**You've got this! 🚀 Good luck!**

---

*DSAI 305 - Solving Research Problems in Healthcare Through Explainable AI*
*Spring 2026*
