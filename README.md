# ModelValidation Utility Class

## Overview
The `ModelValidation` class is a utility designed to validate and evaluate machine learning models. It provides functionalities to assess model performance through various metrics, visualizations, and insights such as feature importance, ROC/AUC curves, SHAP explanations, and confusion matrices.

This class is particularly helpful for analyzing classification models that predict probabilities for binary classification problems.


### Parameters:
- **model**: Trained machine learning model. Should have `predict_proba` and `feature_importances_` attributes.
- **X_valid**: A `pandas.DataFrame` containing the features for validation.
- **y_valid**: A `pandas.Series` containing the binary ground truth labels.
- **scaler**: (Optional) A scaler object (e.g., `StandardScaler`) used to transform the feature set.

---

## Methods

### 1. **get_model_information()**
Prints hyperparameter details of the model.

### 2. **plot_feature_importance_of_model()**
Plots the feature importance as a bar chart.

### 3. **get_confusion_matrix_dataframes_with_threshold(threshold: float)**
Generates predicted outcomes and separates them into four categories:
- True Positives (TP)
- True Negatives (TN)
- False Negatives (FN)
- False Positives (FP)

**Returns**: A tuple of DataFrames: `(df_predicted, df_true_positive, df_true_negative, df_false_negative, df_false_positive)`

### 4. **get_roc_auc_score()**
Computes the ROC AUC score.

### 5. **get_pr_auc_score()**
Computes the Precision-Recall AUC score.

### 6. **get_roc_curve()**
Plots the ROC curve with the AUC value.

### 7. **get_precision_recall_curve()**
Plots the Precision-Recall curve with the AUC value.

### 8. **get_evaluation_plots_for_classification()**
Generates evaluation plots:
- ROC Curve
- Precision-Recall Curve
- Precision-Recall vs. Threshold
- Probability Distributions (positive and negative class)

# Generate Insights
validator.get_model_information()
validator.plot_feature_importance_of_model()
validator.get_roc_curve()
validator.get_precision_recall_curve()
validator.get_confusion_matrix_and_classification_report(threshold=0.5)
validator.get_shap_plot()
validator.create_boxplot_grid(n_cols=3)


---

## Visualization Outputs
The class generates the following visualizations:
1. **Feature Importance Barplot**
2. **ROC Curve**
3. **Precision-Recall Curve**
4. **Precision-Recall vs Threshold Plot**
5. **Probability Distribution Plots** (Positive and Negative Class)
6. **Boxplot Grid**: Feature distribution by labels.
7. **SHAP Summary Plot**: Feature contributions.
8. **Confusion Matrix Heatmap**

---

## License
This project is open-source and available under the MIT License.

---

## Contact
For questions or suggestions, please feel free to contact the author or contribute to this repository.