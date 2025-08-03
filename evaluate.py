from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os


def calculate_detailed_metrics(y_true, y_pred, model_name):
    """
    Calculate comprehensive evaluation metrics.
    """
    try:
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)

        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Per-class precision, recall, f1-score
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # Macro and weighted averages
        precision_macro = np.mean(precision)
        recall_macro = np.mean(recall)
        f1_macro = np.mean(f1)

        precision_weighted = np.average(precision, weights=support)
        recall_weighted = np.average(recall, weights=support)
        f1_weighted = np.average(f1, weights=support)

        detailed_metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'per_class_precision': precision,
            'per_class_recall': recall,
            'per_class_f1': f1,
            'support': support,
            'confusion_matrix': cm
        }

        return detailed_metrics

    except Exception as e:
        logging.error(f"Error calculating detailed metrics for {model_name}: {str(e)}")
        return None


def evaluate_model(model, X_test, y_test, model_name, visualization_dir):
    """
    Comprehensive model evaluation with detailed metrics and visualization.
    """
    try:
        logging.info(f"Evaluating model: {model_name}")

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Calculate detailed metrics
        detailed_metrics = calculate_detailed_metrics(y_test, y_pred, model_name)

        # Log results
        logging.info(f"{model_name} Results:")
        logging.info(f"  Accuracy: {accuracy:.4f}")

        if detailed_metrics:
            logging.info(f"  Precision (weighted): {detailed_metrics['precision_weighted']:.4f}")
            logging.info(f"  Recall (weighted): {detailed_metrics['recall_weighted']:.4f}")
            logging.info(f"  F1-Score (weighted): {detailed_metrics['f1_weighted']:.4f}")

        # Create per-class performance visualization
        if visualization_dir and detailed_metrics:
            create_per_class_visualization(detailed_metrics, model_name, visualization_dir)

        return accuracy, report, y_pred

    except Exception as e:
        logging.error(f"Error evaluating {model_name}: {str(e)}")
        # Return default values on error
        return 0.0, {}, np.zeros(len(y_test))


def create_per_class_visualization(metrics, model_name, visualization_dir):
    """
    Create per-class performance visualization.
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name} - Per-Class Performance Analysis', fontsize=16, fontweight='bold')

        n_classes = len(metrics['per_class_precision'])
        class_names = [f'Class_{i}' for i in range(n_classes)]

        # 1. Per-class precision
        axes[0, 0].bar(range(n_classes), metrics['per_class_precision'],
                       color='skyblue', alpha=0.8)
        axes[0, 0].set_title('Per-Class Precision')
        axes[0, 0].set_xlabel('Class')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].set_xticks(range(n_classes))
        axes[0, 0].set_xticklabels(class_names, rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # Add value labels
        for i, v in enumerate(metrics['per_class_precision']):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

        # 2. Per-class recall
        axes[0, 1].bar(range(n_classes), metrics['per_class_recall'],
                       color='lightcoral', alpha=0.8)
        axes[0, 1].set_title('Per-Class Recall')
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_xticks(range(n_classes))
        axes[0, 1].set_xticklabels(class_names, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # Add value labels
        for i, v in enumerate(metrics['per_class_recall']):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

        # 3. Per-class F1-score
        axes[1, 0].bar(range(n_classes), metrics['per_class_f1'],
                       color='lightgreen', alpha=0.8)
        axes[1, 0].set_title('Per-Class F1-Score')
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_xticks(range(n_classes))
        axes[1, 0].set_xticklabels(class_names, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # Add value labels
        for i, v in enumerate(metrics['per_class_f1']):
            axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

        # 4. Support (number of samples per class)
        axes[1, 1].bar(range(n_classes), metrics['support'],
                       color='gold', alpha=0.8)
        axes[1, 1].set_title('Support (Sample Count per Class)')
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].set_xticks(range(n_classes))
        axes[1, 1].set_xticklabels(class_names, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        # Add value labels
        for i, v in enumerate(metrics['support']):
            axes[1, 1].text(i, v + 0.5, f'{int(v)}', ha='center', va='bottom')

        plt.tight_layout()

        # Save plot
        output_path = os.path.join(visualization_dir, f'{model_name}_per_class_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Saved per-class analysis for {model_name}: {output_path}")

    except Exception as e:
        logging.error(f"Error creating per-class visualization for {model_name}: {str(e)}")


def compare_models_performance(model_results, visualization_dir):
    """
    Create comparative analysis of all models.
    """
    try:
        if not model_results:
            logging.warning("No model results available for comparison")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison Analysis', fontsize=16, fontweight='bold')

        model_names = list(model_results.keys())
        accuracies = [results['accuracy'] for results in model_results.values()]

        # Sort by accuracy for better visualization
        sorted_indices = np.argsort(accuracies)[::-1]
        sorted_names = [model_names[i] for i in sorted_indices]
        sorted_accuracies = [accuracies[i] for i in sorted_indices]

        # 1. Accuracy comparison
        bars = axes[0, 0].bar(range(len(sorted_names)), sorted_accuracies,
                              color=plt.cm.Set3(np.linspace(0, 1, len(sorted_names))))
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticks(range(len(sorted_names)))
        axes[0, 0].set_xticklabels(sorted_names, rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # Add accuracy values on bars
        for bar, acc in zip(bars, sorted_accuracies):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                            f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

        # 2. Precision-Recall scatter plot
        precisions = []
        recalls = []
        for name in model_names:
            if 'precision_weighted' in model_results[name]:
                precisions.append(model_results[name]['precision_weighted'])
                recalls.append(model_results[name]['recall_weighted'])
            else:
                precisions.append(model_results[name]['accuracy'])
                recalls.append(model_results[name]['accuracy'])

        scatter = axes[0, 1].scatter(precisions, recalls,
                                     c=range(len(model_names)),
                                     cmap='viridis', s=100, alpha=0.7)
        axes[0, 1].set_title('Precision vs Recall')
        axes[0, 1].set_xlabel('Precision')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].grid(True, alpha=0.3)

        # Add model name annotations
        for i, name in enumerate(model_names):
            axes[0, 1].annotate(name, (precisions[i], recalls[i]),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=8, alpha=0.8)

        # 3. Performance distribution
        all_metrics = []
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        for name in model_names:
            metrics = [
                model_results[name]['accuracy'],
                model_results[name].get('precision_weighted', model_results[name]['accuracy']),
                model_results[name].get('recall_weighted', model_results[name]['accuracy']),
                model_results[name].get('f1_weighted', model_results[name]['accuracy'])
            ]
            all_metrics.append(metrics)

        all_metrics = np.array(all_metrics)

        # Box plot of all metrics
        box_data = []
        box_labels = []
        for i, metric_name in enumerate(metric_names):
            box_data.extend(all_metrics[:, i])
            box_labels.extend([metric_name] * len(model_names))

        import pandas as pd
        df_box = pd.DataFrame({'Metric': box_labels, 'Score': box_data})

        # Use seaborn for better box plot
        sns.boxplot(data=df_box, x='Metric', y='Score', ax=axes[1, 0])
        axes[1, 0].set_title('Metric Distribution Across Models')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Model ranking
        rankings = []
        for i, name in enumerate(model_names):
            rank = sorted_names.index(name) + 1
            rankings.append(rank)

        axes[1, 1].barh(range(len(model_names)), rankings,
                        color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(model_names))))
        axes[1, 1].set_title('Model Ranking (1 = Best)')
        axes[1, 1].set_xlabel('Rank')
        axes[1, 1].set_yticks(range(len(model_names)))
        axes[1, 1].set_yticklabels(model_names)
        axes[1, 1].grid(True, alpha=0.3)

        # Add rank values
        for i, rank in enumerate(rankings):
            axes[1, 1].text(rank + 0.1, i, f'{rank}',
                            ha='left', va='center', fontweight='bold')

        plt.tight_layout()

        # Save comparison plot
        output_path = os.path.join(visualization_dir, 'model_performance_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Saved model comparison analysis: {output_path}")

    except Exception as e:
        logging.error(f"Error creating model comparison: {str(e)}")


def save_evaluation_report(model_results, output_dir):
    """
    Save detailed evaluation report to file.
    """
    try:
        report_path = os.path.join(output_dir, 'evaluation_report.txt')

        with open(report_path, 'w') as f:
            f.write("DETAILED MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Summary statistics
            accuracies = [results['accuracy'] for results in model_results.values()]
            f.write(f"Number of models evaluated: {len(model_results)}\n")
            f.write(f"Best accuracy: {max(accuracies):.4f}\n")
            f.write(f"Worst accuracy: {min(accuracies):.4f}\n")
            f.write(f"Mean accuracy: {np.mean(accuracies):.4f}\n")
            f.write(f"Std accuracy: {np.std(accuracies):.4f}\n\n")

            # Individual model results
            sorted_models = sorted(model_results.items(),
                                   key=lambda x: x[1]['accuracy'], reverse=True)

            for i, (model_name, results) in enumerate(sorted_models, 1):
                f.write(f"{i}. {model_name}\n")
                f.write(f"   Accuracy: {results['accuracy']:.4f}\n")

                if 'precision_weighted' in results:
                    f.write(f"   Precision: {results['precision_weighted']:.4f}\n")
                    f.write(f"   Recall: {results['recall_weighted']:.4f}\n")
                    f.write(f"   F1-Score: {results['f1_weighted']:.4f}\n")

                f.write("\n")

        logging.info(f"Saved evaluation report: {report_path}")

    except Exception as e:
        logging.error(f"Error saving evaluation report: {str(e)}")