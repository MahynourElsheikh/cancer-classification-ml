# Breast Cancer Classification with Machine Learning

A binary classification project that predicts whether a breast mass is benign or malignant from features computed on digitized images of a fine needle aspirate (FNA). The notebook walks through exploratory analysis, feature selection, model training, validation, and hyperparameter tuning.

## Dataset

- **Source:** Wisconsin Diagnostic Breast Cancer (WDBC) dataset, originally from the UCI Machine Learning Repository (also bundled with scikit-learn). The copy used here was downloaded from Kaggle.
- **Samples:** 569
- **Features:** 30 numeric features describing cell nuclei (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), each given as a mean, standard error, and "worst" value
- **Target:** `benign_0__mal_1` (binary)
- **Missing values:** none

## Project Structure

```
.
├── cancer_classification.ipynb   # Full analysis and modeling
├── cancer_classification.csv     # Dataset
├── requirements.txt              # Python dependencies
└── README.md
```

## Approach

1. **Exploration:** inspected data types, summary statistics, and a correlation heatmap. Feature scales differ widely between columns.
2. **Feature reduction:** dropped five weakly informative or redundant columns based on correlation with the target and with other features (`fractal dimension error`, `mean fractal dimension`, `smoothness error`, `symmetry error`, `texture error`).
3. **Modeling:** trained a Logistic Regression model (75/25 train/test split) and compared it to a Random Forest.
4. **Validation:** used 5-fold cross-validation to check for overfitting.
5. **Regularization experiments:** used `LassoCV` and `ElasticNetCV` to see which features could be removed. Removing the zero-coefficient features slightly lowered recall, so the full reduced feature set was kept.
6. **Hyperparameter tuning:** `GridSearchCV` over `max_iter` and `penalty`; best result was `penalty='l2'`.

## Results

| Model | Test Accuracy | Notes |
|---|---|---|
| Logistic Regression | **0.98** | Precision/recall about 0.98 for both classes |
| Random Forest (10 trees) | 0.94 | Lower than Logistic Regression |

**Logistic Regression confusion matrix (test set):**

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 52 | 2 |
| **Actual 1** | 1 | 88 |

**5-fold cross-validation accuracy:** mean ≈ 0.951 (range 0.906 to 0.976), so the model is stable and not heavily overfit.

## Requirements

- Python 3.9+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter


## Disclaimer

This project is for educational purposes only and is not a medical diagnostic tool.
