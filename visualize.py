import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import logging
from sklearn.metrics import confusion_matrix
from config import Config


def visualize_performance(model_names, accuracies, reports, y_test, y_pred_dict, visualization_dir):
    """
    Visualize model performance with accuracy bar plots and confusion matrices.
    """
    try:
        # Ensure visualization directory exists
        os.makedirs(visualization_dir, exist_ok=True)

        # Set style for better plots
        plt.style.use('default')
        sns.set_palette("husl")

        # Accuracy bar plot
        plt.figure(figsize=(12, 8))
        bars = plt.bar(model_names, accuracies,
                       color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'][:len(model_names)])
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)

        # Add value labels on bars
        for bar, accuracy in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        output_path = os.path.join(visualization_dir, 'model_accuracies.png')
        plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved accuracy plot to {output_path}")

        # Confusion matrices
        unique_labels = np.unique(y_test)
        n_classes = len(unique_labels)

        for model_name, y_pred in y_pred_dict.items():
            if len(y_pred) == 0:  # Skip if no predictions
                continue

            try:
                cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

                plt.figure(figsize=(max(8, n_classes), max(6, n_classes)))

                # Create heatmap
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=unique_labels, yticklabels=unique_labels,
                            cbar_kws={'label': 'Count'})

                plt.title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold')
                plt.xlabel('Predicted Label', fontsize=12)
                plt.ylabel('True Label', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)

                plt.tight_layout()
                output_path = os.path.join(visualization_dir, f"{model_name}_confusion_matrix.png")
                plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
                plt.close()
                logging.info(f"Saved confusion matrix for {model_name} to {output_path}")

            except Exception as e:
                logging.error(f"Error creating confusion matrix for {model_name}: {str(e)}")
                continue

        # Create performance summary plot
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Model Performance Summary', fontsize=16, fontweight='bold')

            # Accuracy comparison
            axes[0, 0].bar(model_names, accuracies, color='skyblue')
            axes[0, 0].set_title('Accuracy Comparison')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)

            # Precision comparison
            precisions = []
            for name in model_names:
                if name in reports and 'weighted avg' in reports[name]:
                    precisions.append(reports[name]['weighted avg']['precision'])
                else:
                    precisions.append(0)

            axes[0, 1].bar(model_names, precisions, color='lightgreen')
            axes[0, 1].set_title('Precision Comparison')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].tick_params(axis='x', rotation=45)

            # Recall comparison
            recalls = []
            for name in model_names:
                if name in reports and 'weighted avg' in reports[name]:
                    recalls.append(reports[name]['weighted avg']['recall'])
                else:
                    recalls.append(0)

            axes[1, 0].bar(model_names, recalls, color='lightcoral')
            axes[1, 0].set_title('Recall Comparison')
            axes[1, 0].set_ylabel('Recall')
            axes[1, 0].tick_params(axis='x', rotation=45)

            # F1-Score comparison
            f1_scores = []
            for name in model_names:
                if name in reports and 'weighted avg' in reports[name]:
                    f1_scores.append(reports[name]['weighted avg']['f1-score'])
                else:
                    f1_scores.append(0)

            axes[1, 1].bar(model_names, f1_scores, color='gold')
            axes[1, 1].set_title('F1-Score Comparison')
            axes[1, 1].set_ylabel('F1-Score')
            axes[1, 1].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            output_path = os.path.join(visualization_dir, 'performance_summary.png')
            plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
            plt.close()
            logging.info(f"Saved performance summary to {output_path}")

        except Exception as e:
            logging.error(f"Error creating performance summary: {str(e)}")

    except Exception as e:
        logging.error(f"Error visualizing performance: {str(e)}")
        raise