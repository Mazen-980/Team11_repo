# Explainable AI for Cardiovascular Disease Prediction

This repository contains the final implementation phase for a DSAI 305 healthcare AI project. The project investigates cardiovascular disease prediction using multiple machine learning, ensemble learning, and reinforcement learning approaches, with a strong emphasis on explainability and interpretability.

The objective is not only to train accurate predictive models, but also to make their predictions understandable through global, local, model-specific, and model-agnostic XAI techniques.

---

## Project Overview

Cardiovascular disease is one of the most important healthcare prediction problems because early risk identification can support prevention, screening, and clinical decision-making. However, in healthcare, model accuracy alone is not enough. A model must also provide transparent reasoning so that clinicians and researchers can understand why a prediction was produced.

This project implements four research-inspired notebooks on the same cardiovascular disease dataset:

1. **Shah et al. (2025)** — Hybrid stacked ensemble learning with XAI.
2. **Talukder et al. (2025)** — Hybrid XAI-ML framework using balancing strategies and Extra Trees.
3. **Rezk et al. (2024)** — XAI-augmented voting ensemble using XGBoost and LightGBM.
4. **Glanois et al. (2024)** — Interpretable reinforcement learning implementation with advanced XAI analysis.

Each notebook follows a consistent professional workflow:

- Dataset loading and validation
- Exploratory data analysis
- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
- Explainability and interpretability analysis
- Artifact saving for reporting and reproducibility

---

## Repository Structure

```text
.
├── shah-et-al-2025.ipynb
├── Shah et al. (2025)_outputs.zip
│
├── talukder-et-al-2025.ipynb
├── talukder_2025_outputs.zip
│
├── rezk-et-al-2024.ipynb
├── rezk_2024_outputs.zip
│
├── glanois-et-al-2024-bouns.ipynb
├── glanois_2024_dqn_policy_state_dict.pt
├── glanois_2024_outputs.zip
│
└── README.md
```

> Note: The file `glanois-et-al-2024-bouns.ipynb` contains the advanced reinforcement learning implementation and extended interpretability analysis.

---

## Dataset

The notebooks use a processed cardiovascular disease dataset available inside Kaggle input storage.

### Dataset details

| Item | Description |
|---|---|
| Problem type | Binary classification |
| Target column | `CARDIO_DISEASE` |
| Number of samples | 68,783 records |
| Original number of columns | 12 columns |
| Processed feature count | 17 features after feature engineering |
| Data type | Clinical/tabular patient records |

### Main original features

- `AGE`
- `GENDER`
- `HEIGHT`
- `WEIGHT`
- `AP_HIGH`
- `AP_LOW`
- `CHOLESTEROL`
- `GLUCOSE`
- `SMOKE`
- `ALCOHOL`
- `PHYSICAL_ACTIVITY`
- `CARDIO_DISEASE`

### Engineered features

Several clinically meaningful engineered features are added, including:

- `AGE_YEARS`
- `BMI`
- `PULSE_PRESSURE`
- `BP_RATIO`
- `AGE_BMI_INTERACTION`
- `BP_CHOLESTEROL_INTERACTION`

---

## Implemented Notebooks

### 1. Shah et al. (2025): Hybrid Ensemble XAI Model

**Notebook:** `shah-et-al-2025.ipynb`  
**Artifacts:** `Shah et al. (2025)_outputs.zip`

This notebook implements a hybrid ensemble pipeline inspired by Shah et al. The model combines multiple strong learners and uses a stacking strategy where base models generate predictive representations and a meta-model learns the final decision boundary.

#### Main models

- XGBoost
- LightGBM
- CatBoost
- Random Forest
- Gradient Boosting
- Logistic Regression
- Voting / stacking-style ensemble

#### Explainability techniques

- SHAP global feature importance
- SHAP beeswarm plots
- SHAP local explanations
- LIME local explanations
- Permutation importance
- PDP / ICE plots
- PCA / t-SNE projection analysis
- Surrogate decision tree rules

#### Saved outputs

The output package includes trained model files, evaluation tables, plots, XAI figures, local explanation files, and experiment summaries.

---

### 2. Talukder et al. (2025): HXAI-ML Framework

**Notebook:** `talukder-et-al-2025.ipynb`  
**Artifacts:** `talukder_2025_outputs.zip`

This notebook implements a hybrid explainable ML approach focused on class balancing and tree-based prediction. The main model is based on an Extra Trees Classifier combined with resampling strategies.

#### Main methods

- Random Oversampling
- SMOTE
- Tomek Links
- RO + Tomek Links
- Extra Trees Classifier
- Comparison with several baseline ML models

#### Main final model

- **RO + Tomek Links + Extra Trees Classifier**

#### Representative executed metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.7239 |
| Precision | 0.7660 |
| Recall | 0.6365 |
| F1-score | 0.6952 |
| ROC-AUC | 0.7897 |
| PR-AUC | 0.7725 |
| Specificity | 0.8095 |

#### Explainability techniques

- Permutation importance
- SHAP global explanation
- SHAP local waterfall explanation
- LIME local explanation
- PDP / ICE plots
- Calibration curve
- Threshold sensitivity analysis

---

### 3. Rezk et al. (2024): XAI-Augmented Voting Ensemble

**Notebook:** `rezk-et-al-2024.ipynb`  
**Artifacts:** `rezk_2024_outputs.zip`

This notebook implements an XAI-augmented voting ensemble inspired by Rezk et al. The main idea is to combine strong gradient boosting learners and then use explainability results to support feature selection and model interpretation.

#### Main models

- XGBoost
- LightGBM
- Soft Voting Ensemble: XGBoost + LightGBM
- XAI-selected feature retraining

#### Representative executed metrics

| Model | Accuracy | F1-weighted | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|
| XGBoost | 0.7321 | 0.7314 | 0.7992 | 0.7813 |
| LightGBM | 0.7300 | 0.7295 | 0.7975 | 0.7773 |
| Soft Voting XGBoost + LightGBM | 0.7303 | 0.7297 | 0.7989 | 0.7805 |

#### Explainability techniques

- SHAP feature importance
- SHAP local explanation
- LIME local explanation
- Permutation importance
- PDP / ICE plots
- XAI-guided feature selection
- Calibration analysis
- XAI method comparison

---

### 4. Glanois et al. (2024): Interpretable Reinforcement Learning

**Notebook:** `glanois-et-al-2024-bouns.ipynb`  
**Model weights:** `glanois_2024_dqn_policy_state_dict.pt`  
**Artifacts:** `glanois_2024_outputs.zip`

This notebook is the advanced implementation of the project. It translates interpretable reinforcement learning concepts into a cardiovascular disease prediction task.

Instead of directly predicting the target from all features, the RL agent learns a **feature acquisition policy**:

1. The agent starts with no revealed clinical features.
2. It decides which feature to acquire next.
3. Each acquired feature has a small cost.
4. The agent eventually makes a diagnostic action.
5. The reward balances prediction correctness and feature efficiency.

This makes the model more interpretable because the final decision is supported by a visible sequence of feature requests and decision steps.

#### Main RL method

- Deep Q-Network policy
- Feature acquisition environment
- Reward-based diagnostic decision-making
- Patient-level trajectory analysis

#### Advanced interpretability and XAI techniques

- Feature acquisition frequency
- Patient-level decision trajectories
- Reward decomposition
- Feature acquisition position heatmap
- Policy distillation into a decision tree
- SHAP explanations for policy behavior
- LIME local explanations
- Integrated Gradients
- Gradient × Input saliency
- Occlusion sensitivity
- Counterfactual feature interventions
- Policy sensitivity curves
- Cross-XAI agreement heatmap

This notebook provides the richest interpretability section in the repository and is designed to support the advanced research excellence component of the project.

---

## How to Run the Project on Kaggle

### 1. Open the notebook on Kaggle

Upload or open the required notebook in Kaggle.

### 2. Attach the dataset

The notebooks are configured to detect the dataset from one of the following Kaggle paths:

```python
/kaggle/input/datasets/mohameddarweish/cardiovascular-diseases
/kaggle/input/datasets/mohameddarweish/cardiovascular-diseases-processed
```

The expected CSV file name is:

```text
cardiovascular_diseases_processed.csv
```

### 3. Install missing dependencies if needed

Most packages are already available on Kaggle. If a package is missing, run:

```python
!pip install shap lime imbalanced-learn xgboost lightgbm catboost joblib
```

For the reinforcement learning notebook, PyTorch is required:

```python
!pip install torch
```

### 4. Run all cells from top to bottom

Each notebook is designed to run independently. The general execution order is:

1. Configuration
2. Imports
3. Dataset loading
4. EDA
5. Feature engineering
6. Preprocessing
7. Model training
8. Evaluation
9. XAI analysis
10. Artifact saving

### 5. Download generated artifacts

After execution, each notebook creates an output directory under Kaggle working storage, such as:

```text
/kaggle/working/talukder_2025_outputs
/kaggle/working/rezk_2024_outputs
/kaggle/working/glanois_2024_interpretable_rl_outputs
/kaggle/working/shah_2025_artifacts
```

The generated outputs can be zipped and uploaded to GitHub as the corresponding `*_outputs.zip` files.

---

## Main Evaluation Metrics

The notebooks evaluate models using several classification metrics:

- Accuracy
- Precision
- Recall / Sensitivity
- Specificity
- F1-score
- ROC-AUC
- PR-AUC / Average Precision
- Confusion matrix
- Classification report
- Calibration and threshold analysis where applicable

These metrics are saved as CSV, TXT, JSON, and figure files inside each output package.

---

## Explainability Summary

The project uses a broad range of XAI methods across the four implemented notebooks.

| XAI Technique | Purpose |
|---|---|
| SHAP | Global and local feature contribution analysis |
| LIME | Local model-agnostic explanation for individual predictions |
| Permutation Importance | Measures feature impact by performance degradation |
| PDP / ICE | Visualizes marginal effect of features on predictions |
| Surrogate Tree | Converts complex model behavior into interpretable rules |
| PCA / t-SNE | Visualizes learned representation and class separability |
| Calibration Analysis | Evaluates reliability of predicted probabilities |
| Threshold Analysis | Studies sensitivity-specificity trade-offs |
| Integrated Gradients | Neural/RL attribution analysis |
| Gradient × Input | Saliency-based neural explanation |
| Occlusion Sensitivity | Measures prediction sensitivity to removed features |
| Counterfactual Interventions | Studies how feature changes affect decisions |
| Reward Decomposition | Explains RL behavior through reward components |
| Policy Distillation | Converts RL policy behavior into an interpretable decision tree |

---

## Reproducibility

The notebooks use fixed random seeds where possible to improve reproducibility. However, results may vary slightly because of:

- Kaggle hardware differences
- GPU/CPU execution differences
- Randomized hyperparameter search
- Stochastic training in neural and RL models
- Library version differences

For best reproducibility, run each notebook from a clean Kaggle session and keep the generated output folders unchanged.

---

## Ethical and Clinical Considerations

This repository is intended for academic and research purposes only. The implemented models are not clinical decision systems and should not be used as a substitute for medical judgment.

Important considerations include:

- **Patient privacy:** The dataset must be handled without exposing personally identifiable information.
- **Bias and fairness:** Model behavior may differ across patient subgroups, so subgroup-level validation is important before real-world use.
- **Clinical overreliance:** Predictions should support, not replace, professional clinical reasoning.
- **Transparency:** XAI techniques are used to make model behavior more understandable, but explanations are approximations and should be interpreted carefully.
- **Generalization:** Models trained on one dataset may not generalize to other populations without external validation.

---

## Final Notes

This project provides a complete experimental framework for cardiovascular disease prediction using explainable AI. The implementation covers traditional ML, ensemble learning, hybrid XAI workflows, and interpretable reinforcement learning.

The strongest contribution of the repository is the comparison between multiple explainability methods across different model families, especially the final reinforcement learning notebook, which provides decision trajectories, reward analysis, policy explanations, and cross-XAI agreement.
