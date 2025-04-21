import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
from tqdm import tqdm
from typing import Optional, Tuple, List, Any

# Setting Configuration
sns.set_style('whitegrid')
# Pandas Progress bar
tqdm.pandas()

class ModelValidation:
    def __init__(self, 
                 model: Any, 
                 X_valid: pd.DataFrame, 
                 y_valid: pd.Series, 
                 scaler: Optional[Any] = None):
        """
        A utility class for validating machine learning models using various evaluation metrics and visualizations.

        Parameters:
        model (Any): Trained machine learning model with required attributes like predict_proba and feature_importances_.
        X_valid (pd.DataFrame): Validation feature dataset. Index should not be a feature.
        y_valid (pd.Series): Ground truth labels. Should be integers (0 & 1) without NaN values.
        scaler (Optional[Any]): Scaler object for transforming features. Default is None.
        """

        self.model = model
        self.scaler = scaler
        self.y_actual = list(y_valid.astype(int))

        # Store feature names and importances
        self.feature_columns = self.model.feature_names_in_
        self.df_feature_importances = pd.Series(self.model.feature_importances_, 
                                                index=self.feature_columns).sort_values(ascending=False)
        
        # Create DataFrame for validation
        self.df = X_valid
        self.df_valid = X_valid[self.feature_columns].reset_index(drop=True).copy()
        self.df_valid['y_actual'] = self.y_actual

        # Transform features if scaler is provided
        self.X_valid = X_valid[self.feature_columns].reset_index(drop=True)
        if self.scaler is not None:
            self.X_valid = pd.DataFrame(self.scaler.transform(self.X_valid),
                                        columns=self.feature_columns).reset_index(drop=True)
            print("Features transformed & df_valid_transformed created!")

            self.df_valid_transformed = self.X_valid.copy()
            self.df_valid_transformed['y_actual'] = self.y_actual
        else:
            self.df_valid_transformed = self.X_valid.copy()
            self.df_valid_transformed['y_actual'] = self.y_actual

        # Predict probabilities
        self.y_negative_probs = list(self.model.predict_proba(self.X_valid)[:, 0])
        self.y_positive_probs = list(self.model.predict_proba(self.X_valid)[:, 1])

    def get_model_information(self) -> None:
        """
        Displays the model's hyperparameters and settings.
        """
        print('ccp_alpha\t:', self.model.ccp_alpha)
        print('criterion\t:', self.model.criterion)
        print('max_depth\t:', self.model.max_depth)
        print('max_leaf_nodes\t:', self.model.max_leaf_nodes)
        print('min_samples_leaf\t:', self.model.min_samples_leaf)
        print('min_samples_split\t:', self.model.min_samples_split)
        print('n_estimators\t:', self.model.n_estimators)
        print('bootstrap\t:', self.model.bootstrap)
        print('max_features\t:', self.model.max_features)
        print('class_weight\t:', self.model.class_weight)

    def plot_feature_importance_of_model(self) -> None:
        """
        Plots the feature importance of the trained model.
        """
        plt.figure(figsize=(20, 10))
        sns.barplot(y=self.df_feature_importances.index, x=self.df_feature_importances)
        plt.title('Feature Importance')
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.show()

    def get_confusion_matrix_dataframes_with_threshold(self, threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Creates dataframes for predicted results based on the given threshold.

        Parameters:
        threshold (float): Threshold for classifying positive probabilities.

        Returns:
        Tuple of DataFrames: df_predicted, df_true_positive, df_true_negative, df_false_negative, df_false_positive
        """
        df_predicted = self.df.copy()
        df_predicted['y_actual_label'] = self.y_actual
        df_predicted['y_negative_probs'] = self.y_negative_probs
        df_predicted['y_positive_probs'] = self.y_positive_probs
        df_predicted['predicted_labels'] = (np.array(self.y_positive_probs) >= threshold).astype(int)

        # Subdataframes
        df_true_positive = df_predicted[(df_predicted.predicted_labels == 1) & (df_predicted.y_actual_label == 1)]
        df_true_negative = df_predicted[(df_predicted.predicted_labels == 0) & (df_predicted.y_actual_label == 0)]
        df_false_negative = df_predicted[(df_predicted.predicted_labels == 0) & (df_predicted.y_actual_label == 1)]
        df_false_positive = df_predicted[(df_predicted.predicted_labels == 1) & (df_predicted.y_actual_label == 0)]

        print('df_predicted:', df_predicted.shape)
        print('df_true_positive:', df_true_positive.shape)
        print('df_true_negative:', df_true_negative.shape)
        print('df_false_negative:', df_false_negative.shape)
        print('df_false_positive:', df_false_positive.shape)

        return df_predicted, df_true_positive, df_true_negative, df_false_negative, df_false_positive

    def get_roc_auc_score(self):
        fpr, tpr, thresholds_roc = roc_curve(self.y_actual, self.y_positive_probs, pos_label=1)
        roc_auc = auc(fpr, tpr)
        return roc_auc 

    def get_pr_auc_score(self):
        fpr, tpr, thresholds_roc = roc_curve(self.y_actual, self.y_positive_probs, pos_label=1)
        roc_auc = auc(fpr, tpr)
        return roc_auc 
        
    def get_roc_curve(self):
        fpr, tpr, thresholds_roc = roc_curve(self.y_actual, self.y_positive_probs, pos_label=1)
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve on a single axis
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

    def get_precision_recall_curve(self):
        precision, recall, thresholds_pr = precision_recall_curve(self.y_actual, self.y_positive_probs, pos_label=1)
        pr_auc = auc(recall, precision)

        # Plot the Precision-Recall curve on a single axis
        plt.plot(recall, precision, label=f'AUC = {pr_auc:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()
        


    def get_all_evaluation_plots_for_classification(self) -> None:
        """
        Generates evaluation plots including ROC curve, Precision-Recall curve, and probability distributions.
        """
        fpr, tpr, thresholds_roc = roc_curve(self.y_actual, self.y_positive_probs, pos_label=1)
        roc_auc = auc(fpr, tpr)

        precision, recall, thresholds_pr = precision_recall_curve(self.y_actual, self.y_positive_probs, pos_label=1)
        pr_auc = auc(recall, precision)

        fig, axs = plt.subplots(4, 2, figsize=(12, 20))
        axs = axs.flatten()

        axs[0].plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        axs[0].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
        axs[0].set_xlabel('False Positive Rate')
        axs[0].set_ylabel('True Positive Rate')
        axs[0].set_title('ROC Curve')
        axs[0].legend()

        axs[1].plot(recall, precision, label=f'AUC = {pr_auc:.2f}')
        axs[1].set_xlabel('Recall')
        axs[1].set_ylabel('Precision')
        axs[1].set_title('Precision-Recall Curve')
        axs[1].legend()

        axs[2].plot(thresholds_pr, precision[:-1], label='Precision', color='blue')
        axs[2].plot(thresholds_pr, recall[:-1], label='Recall', color='orange')
        axs[2].set_xlabel('Threshold')
        axs[2].set_ylabel('Score')
        axs[2].set_title('Precision & Recall vs. Threshold')
        axs[2].legend()

        sns.kdeplot(x=self.y_positive_probs, fill=True, color='green', ax=axs[4])
        axs[4].set_title('Positive Class Probability Distribution')
        axs[4].set_xlabel('Positive Class Probability')
        axs[4].set_ylabel('Density')

        sns.kdeplot(x=self.y_negative_probs, fill=True, color='red', ax=axs[5])
        axs[5].set_title('Negative Class Probability Distribution')
        axs[5].set_xlabel('Negative Class Probability')
        axs[5].set_ylabel('Density')

        sns.kdeplot(x=self.y_positive_probs, hue=self.y_actual, fill=True, common_norm=False, alpha=0.5, ax=axs[6])
        axs[6].set_title('Positive Prob-Score Distribution by Category')

        sns.kdeplot(x=self.y_negative_probs, hue=self.y_actual, fill=True, common_norm=False, alpha=0.5, ax=axs[7])
        axs[7].set_title('Negative Prob-Score Distribution by Category')

        plt.tight_layout()
        plt.show()

    def get_confusion_matrix_and_classification_report(self, threshold: float) -> None:
        """
        Displays the confusion matrix and classification report for a given threshold.

        Parameters:
        threshold (float): Threshold for classifying positive probabilities.
        """
        conf_matrix = confusion_matrix(self.y_actual, 
                                       np.array(self.y_positive_probs) >= threshold)
        plt.figure(figsize=(4, 3))
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', annot_kws={'size': 10})
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('Actual Labels')
        plt.show()

        print("Classification Report")
        print(classification_report(self.y_actual, 
                                    np.array(self.y_positive_probs) >= threshold))

    def get_shap_plot(self,sample_size=5000) -> None:
        """
        Generates SHAP plots for understanding feature importance.
        """
        # Calculate the original positive-to-negative ratio
        pos_count = len(self.df_valid_transformed[self.df_valid_transformed['y_actual'] == 1])
        neg_count = len(self.df_valid_transformed[self.df_valid_transformed['y_actual'] == 0])
        total_count = len(self.df_valid_transformed)
        
        pos_ratio = pos_count / total_count
        neg_ratio = neg_count / total_count
    
        # Sample based on the original ratio
        pos_sample_size = int(sample_size * pos_ratio)
        neg_sample_size = sample_size - pos_sample_size  # Remaining samples for negatives
    
        df_test_pos = self.df_valid_transformed[self.df_valid_transformed['y_actual'] == 1].sample(pos_sample_size)
        df_test_neg = self.df_valid_transformed[self.df_valid_transformed['y_actual'] == 0].sample(neg_sample_size)
        df_test_sampled = pd.concat([df_test_pos, df_test_neg], ignore_index=True)
    
        X_test = df_test_sampled.iloc[:, :-1]
    
        # Calculate SHAP values
        explainer = shap.Explainer(self.model)
        chunk_size = 10
        shap_values = []
    
        with tqdm(total=len(X_test), desc="Calculating SHAP values") as pbar:
            for i in range(0, len(X_test), chunk_size):
                chunk_shap_values = explainer.shap_values(X_test.iloc[i:i + chunk_size])
                shap_values.append(chunk_shap_values)
                pbar.update(min(chunk_size, len(X_test) - i))
    
        shap_values = [np.concatenate([chunk[i] for chunk in shap_values]) for i in range(len(shap_values[0]))]
        print(f"Plotting SHAP Plot for sample size {sample_size}")
        shap.summary_plot(shap_values[1], X_test, plot_type="dot", feature_names=X_test.columns)
        plt.show()


    def create_boxplot_grid(self, n_cols: int = 3) -> None:
        """
        Creates a grid of boxplots for feature distributions by actual labels.

        Parameters:
        n_cols (int): Number of columns in the grid. Default is 3.
        """
        y_columns = self.feature_columns
        x_column = 'y_actual'
        df_feature_util = self.df_valid

        n_rows = len(y_columns) // n_cols + (1 if len(y_columns) % n_cols > 0 else 0)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten()

        # Plot each boxplot
        for i, col in tqdm(enumerate(y_columns)):
            ax = axes[i]
            sns.boxplot(
                data=df_feature_util,
                x=x_column,
                y=col,
                ax=ax,
                showfliers=False,
                palette='Set2'  # Use a predefined color palette
            )
            
            # Set transparency by updating patch properties
            for patch in ax.patches:
                patch.set_alpha(0.7)
            
            # Darken the median line
            for line in ax.lines:
                # The median line is the 5th line object, so we filter it by index
                if line.get_label() == '_median':
                    line.set_color('black')  # Darker color
                    line.set_linewidth(20)  # Increase the line width
            
            ax.set_title(f'{col}')
        
        # Hide any unused subplots
        for j in range(len(y_columns), len(axes)):
            axes[j].axis('off')
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()
