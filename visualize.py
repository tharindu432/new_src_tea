import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import logging
from config import Config


def visualize_performance(model_names, accuracies, reports, y_test, y_pred_dict, visualization_dir):
    """
    Visualize model performance with accuracy bar plots and confusion matrices.
    """
    try:
        # Accuracy bar plot
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, accuracies)
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        output_path = os.path.join(visualization_dir, 'model_accuracies.png')
        plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved accuracy plot to {output_path}")

        # Confusion matrices
        for model_name, y_pred in y_pred_dict.items():
            cm = np.zeros((len(np.unique(y_test)), len(np.unique(y_test))))
            for i, true_label in enumerate(np.unique(y_test)):
                for j, pred_label in enumerate(np.unique(y_pred)):
                    cm[i, j] = np.sum((y_test == true_label) & (y_pred == pred_label))
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues')
            plt.title(f'Confusion Matrix: {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            output_path = os.path.join(visualization_dir, f"{model_name}_confusion_matrix.png")
            plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
            plt.close()
            logging.info(f"Saved confusion matrix for {model_name} to {output_path}")
    except Exception as e:
        logging.error(f"Error visualizing performance: {str(e)}")
        raise