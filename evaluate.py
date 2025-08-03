from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd


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
            create_confusion_matrix_visualization(detailed_metrics['confusion_matrix'],
                                                  model_name, visualization_dir)

        return accuracy, report, y_pred

    except Exception as e:
        logging.error(f"Error evaluating {model_name}: {str(e)}")
        # Return default values on error
        return 0.0, {}, np.zeros(len(y_test))


def create_confusion_matrix_visualization(cm, model_name, visualization_dir):
    """
    Create confusion matrix heatmap visualization.
    """
    try:
        plt.figure(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, cbar_kws={'shrink': 0.8})
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)

        # Add percentage annotations
        total = np.sum(cm)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                percentage = cm[i, j] / total * 100
                plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                         ha='center', va='center', fontsize=10, color='gray')

        plt.tight_layout()

        # Save confusion matrix
        output_path = os.path.join(visualization_dir, f'{model_name}_confusion_matrix.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Saved confusion matrix for {model_name}: {output_path}")

    except Exception as e:
        logging.error(f"Error creating confusion matrix for {model_name}: {str(e)}")


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
        bars1 = axes[0, 0].bar(range(n_classes), metrics['per_class_precision'],
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
        bars2 = axes[0, 1].bar(range(n_classes), metrics['per_class_recall'],
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
        bars3 = axes[1, 0].bar(range(n_classes), metrics['per_class_f1'],
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
        bars4 = axes[1, 1].bar(range(n_classes), metrics['support'],
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
    Create comprehensive comparative analysis of all models.
    """
    try:
        if not model_results:
            logging.warning("No model results available for comparison")
            return

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')

        model_names = list(model_results.keys())
        accuracies = [results['accuracy'] for results in model_results.values()]

        # Sort by accuracy for better visualization
        sorted_indices = np.argsort(accuracies)[::-1]
        sorted_names = [model_names[i] for i in sorted_indices]
        sorted_accuracies = [accuracies[i] for i in sorted_indices]

        # 1. Accuracy comparison with enhanced visualization
        colors = plt.cm.Set3(np.linspace(0, 1, len(sorted_names)))
        bars = axes[0, 0].bar(range(len(sorted_names)), sorted_accuracies, color=colors, alpha=0.8)
        axes[0, 0].set_title('Model Accuracy Comparison', fontweight='bold')
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticks(range(len(sorted_names)))
        axes[0, 0].set_xticklabels(sorted_names, rotation=45, ha='right')
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
        axes[0, 1].set_title('Precision vs Recall', fontweight='bold')
        axes[0, 1].set_xlabel('Precision')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].grid(True, alpha=0.3)

        # Add model name annotations
        for i, name in enumerate(model_names):
            axes[0, 1].annotate(name, (precisions[i], recalls[i]),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=8, alpha=0.8)

        # 3. Performance metrics radar chart simulation
        metrics_to_compare = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        # Create bar chart for all metrics
        x_pos = np.arange(len(metric_labels))
        width = 0.8 / len(model_names)

        for i, name in enumerate(model_names[:5]):  # Limit to top 5 models for clarity
            values = []
            for metric in metrics_to_compare:
                if metric in model_results[name]:
                    values.append(model_results[name][metric])
                else:
                    values.append(model_results[name]['accuracy'])

            axes[0, 2].bar(x_pos + i * width, values, width,
                           label=name, alpha=0.8, color=colors[i])

        axes[0, 2].set_title('All Metrics Comparison (Top 5 Models)', fontweight='bold')
        axes[0, 2].set_xlabel('Metrics')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].set_xticks(x_pos + width * 2)
        axes[0, 2].set_xticklabels(metric_labels)
        axes[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Performance distribution box plot
        all_metrics = []
        box_labels = []
        for name in model_names:
            metrics_values = [
                model_results[name]['accuracy'],
                model_results[name].get('precision_weighted', model_results[name]['accuracy']),
                model_results[name].get('recall_weighted', model_results[name]['accuracy']),
                model_results[name].get('f1_weighted', model_results[name]['accuracy'])
            ]
            all_metrics.extend(metrics_values)
            box_labels.extend(metric_labels)

        # Create DataFrame for seaborn
        df_box = pd.DataFrame({'Metric': box_labels, 'Score': all_metrics})
        sns.boxplot(data=df_box, x='Metric', y='Score', ax=axes[1, 0])
        axes[1, 0].set_title('Metric Distribution Across All Models', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Model ranking with performance bands
        rankings = []
        for i, name in enumerate(model_names):
            rank = sorted_names.index(name) + 1
            rankings.append(rank)

        # Create horizontal bar chart with color coding
        colors_rank = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(model_names)))
        bars_rank = axes[1, 1].barh(range(len(model_names)), rankings, color=colors_rank, alpha=0.8)
        axes[1, 1].set_title('Model Ranking (1 = Best)', fontweight='bold')
        axes[1, 1].set_xlabel('Rank')
        axes[1, 1].set_ylabel('Models')
        axes[1, 1].set_yticks(range(len(model_names)))
        axes[1, 1].set_yticklabels(model_names)
        axes[1, 1].grid(True, alpha=0.3)

        # Add rank values and accuracy
        for i, (rank, name) in enumerate(zip(rankings, model_names)):
            acc = model_results[name]['accuracy']
            axes[1, 1].text(rank + 0.1, i, f'{rank} ({acc:.3f})',
                            ha='left', va='center', fontweight='bold')

        # 6. Performance summary statistics
        axes[1, 2].axis('off')

        # Calculate summary statistics
        acc_mean = np.mean(accuracies)
        acc_std = np.std(accuracies)
        best_model = sorted_names[0]
        worst_model = sorted_names[-1]

        summary_text = f"""Performance Summary:

Models Evaluated: {len(model_names)}

Accuracy Statistics:
  Best: {max(accuracies):.4f} ({best_model})
  Worst: {min(accuracies):.4f} ({worst_model})
  Mean: {acc_mean:.4f}
  Std: {acc_std:.4f}
  Range: {max(accuracies) - min(accuracies):.4f}

Top 3 Models:
"""

        for i in range(min(3, len(sorted_names))):
            name = sorted_names[i]
            acc = sorted_accuracies[i]
            summary_text += f"  {i + 1}. {name}: {acc:.4f}\n"

        # Add model distribution info
        summary_text += f"\nModel Types Distribution:\n"
        model_types = {}
        for name in model_names:
            model_type = name.split('_')[0] if '_' in name else name
            model_types[model_type] = model_types.get(model_type, 0) + 1

        for model_type, count in model_types.items():
            summary_text += f"  {model_type}: {count}\n"

        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()

        # Save comparison plot
        output_path = os.path.join(visualization_dir, 'comprehensive_model_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Saved comprehensive model comparison: {output_path}")

        # Save detailed comparison table
        save_detailed_comparison_table(model_results, visualization_dir)

    except Exception as e:
        logging.error(f"Error creating model comparison: {str(e)}")


def save_detailed_comparison_table(model_results, visualization_dir):
    """
    Save detailed model comparison table to CSV.
    """
    try:
        comparison_data = []

        for model_name, results in model_results.items():
            row = {
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision_Weighted': results.get('precision_weighted', results['accuracy']),
                'Recall_Weighted': results.get('recall_weighted', results['accuracy']),
                'F1_Score_Weighted': results.get('f1_weighted', results['accuracy']),
                'Precision_Macro': results.get('precision_macro', results['accuracy']),
                'Recall_Macro': results.get('recall_macro', results['accuracy']),
                'F1_Score_Macro': results.get('f1_macro', results['accuracy'])
            }
            comparison_data.append(row)

        # Convert to DataFrame and sort by accuracy
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)

        # Add ranking
        comparison_df['Rank'] = range(1, len(comparison_df) + 1)

        # Reorder columns
        columns_order = ['Rank', 'Model', 'Accuracy', 'Precision_Weighted', 'Recall_Weighted',
                         'F1_Score_Weighted', 'Precision_Macro', 'Recall_Macro', 'F1_Score_Macro']
        comparison_df = comparison_df[columns_order]

        # Save to CSV
        csv_path = os.path.join(visualization_dir, 'detailed_model_comparison.csv')
        comparison_df.to_csv(csv_path, index=False, float_format='%.4f')

        logging.info(f"Saved detailed model comparison table: {csv_path}")

    except Exception as e:
        logging.error(f"Error saving detailed comparison table: {str(e)}")


def save_evaluation_report(model_results, output_dir):
    """
    Save comprehensive evaluation report to file.
    """
    try:
        report_path = os.path.join(output_dir, 'comprehensive_evaluation_report.txt')

        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE MODEL EVALUATION REPORT\n")
            f.write("=" * 70 + "\n\n")

            # Summary statistics
            accuracies = [results['accuracy'] for results in model_results.values()]
            f.write(f"EVALUATION SUMMARY:\n")
            f.write(f"  Number of models evaluated: {len(model_results)}\n")
            f.write(f"  Best accuracy: {max(accuracies):.4f}\n")
            f.write(f"  Worst accuracy: {min(accuracies):.4f}\n")
            f.write(f"  Mean accuracy: {np.mean(accuracies):.4f}\n")
            f.write(f"  Standard deviation: {np.std(accuracies):.4f}\n")
            f.write(f"  Accuracy range: {max(accuracies) - min(accuracies):.4f}\n\n")

            # Individual model results (sorted by performance)
            sorted_models = sorted(model_results.items(),
                                   key=lambda x: x[1]['accuracy'], reverse=True)

            f.write("DETAILED MODEL RESULTS (Ranked by Accuracy):\n")
            f.write("-" * 70 + "\n")

            for i, (model_name, results) in enumerate(sorted_models, 1):
                f.write(f"{i:2d}. {model_name}\n")
                f.write(f"     Accuracy: {results['accuracy']:.4f}\n")

                if 'precision_weighted' in results:
                    f.write(f"     Precision (weighted): {results['precision_weighted']:.4f}\n")
                    f.write(f"     Recall (weighted): {results['recall_weighted']:.4f}\n")
                    f.write(f"     F1-Score (weighted): {results['f1_weighted']:.4f}\n")

                    if 'precision_macro' in results:
                        f.write(f"     Precision (macro): {results['precision_macro']:.4f}\n")
                        f.write(f"     Recall (macro): {results['recall_macro']:.4f}\n")
                        f.write(f"     F1-Score (macro): {results['f1_macro']:.4f}\n")

                f.write("\n")

            # Performance analysis
            f.write("PERFORMANCE ANALYSIS:\n")
            f.write("-" * 30 + "\n")

            # Top performers
            top_3 = sorted_models[:3]
            f.write("Top 3 performing models:\n")
            for i, (name, results) in enumerate(top_3, 1):
                f.write(f"  {i}. {name}: {results['accuracy']:.4f}\n")

            # Performance gaps
            if len(sorted_models) > 1:
                best_acc = sorted_models[0][1]['accuracy']
                worst_acc = sorted_models[-1][1]['accuracy']
                f.write(f"\nPerformance gap: {best_acc - worst_acc:.4f}\n")

                # Models within 1% of best
                close_performers = [name for name, results in sorted_models
                                    if best_acc - results['accuracy'] <= 0.01]
                f.write(f"Models within 1% of best: {len(close_performers)}\n")
                for name in close_performers:
                    f.write(f"  - {name}\n")

            f.write(f"\nReport generated: {pd.Timestamp.now()}\n")

        logging.info(f"Saved comprehensive evaluation report: {report_path}")

    except Exception as e:
        logging.error(f"Error saving evaluation report: {str(e)}")


def create_model_performance_dashboard(model_results, visualization_dir):
    """
    Create a comprehensive dashboard visualization of all model results.
    """
    try:
        # Create a large dashboard figure
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        fig.suptitle('Tea Particle Analysis - Model Performance Dashboard',
                     fontsize=20, fontweight='bold', y=0.98)

        model_names = list(model_results.keys())
        accuracies = [results['accuracy'] for results in model_results.values()]

        # Sort models by accuracy
        sorted_indices = np.argsort(accuracies)[::-1]
        sorted_names = [model_names[i] for i in sorted_indices]
        sorted_accuracies = [accuracies[i] for i in sorted_indices]

        # 1. Main accuracy comparison (top-left, spans 2x2)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        colors = plt.cm.Set3(np.linspace(0, 1, len(sorted_names)))
        bars = ax1.bar(range(len(sorted_names)), sorted_accuracies, color=colors, alpha=0.8)
        ax1.set_title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Models', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_xticks(range(len(sorted_names)))
        ax1.set_xticklabels(sorted_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)

        # Add accuracy values on bars
        for bar, acc in zip(bars, sorted_accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                     f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

        # 2. Performance metrics comparison (top-right)
        ax2 = fig.add_subplot(gs[0:2, 2:4])
        metrics_to_show = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        # Show top 5 models
        top_5_models = sorted_names[:5]
        x_pos = np.arange(len(metric_labels))
        width = 0.15

        for i, model_name in enumerate(top_5_models):
            values = []
            for metric in metrics_to_show:
                if metric in model_results[model_name]:
                    values.append(model_results[model_name][metric])
                else:
                    values.append(model_results[model_name]['accuracy'])

            ax2.bar(x_pos + i * width, values, width,
                    label=model_name, alpha=0.8, color=colors[i])

        ax2.set_title('Performance Metrics - Top 5 Models', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Metrics', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_xticks(x_pos + width * 2)
        ax2.set_xticklabels(metric_labels)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        # 3. Accuracy distribution histogram (bottom-left)
        ax3 = fig.add_subplot(gs[2, 0:2])
        ax3.hist(accuracies, bins=10, alpha=0.7, color='lightblue', edgecolor='black')
        ax3.axvline(np.mean(accuracies), color='red', linestyle='--',
                    label=f'Mean: {np.mean(accuracies):.3f}')
        ax3.axvline(np.median(accuracies), color='green', linestyle='--',
                    label=f'Median: {np.median(accuracies):.3f}')
        ax3.set_title('Accuracy Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Accuracy')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Model ranking (bottom-right)
        ax4 = fig.add_subplot(gs[2, 2:4])
        rankings = list(range(1, len(model_names) + 1))
        colors_rank = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(model_names)))

        bars_rank = ax4.barh(range(len(sorted_names)),
                             [len(sorted_names) - i for i in range(len(sorted_names))],
                             color=colors_rank, alpha=0.8)
        ax4.set_title('Model Performance Ranking', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Performance Score (Higher = Better)')
        ax4.set_ylabel('Models')
        ax4.set_yticks(range(len(sorted_names)))
        ax4.set_yticklabels(sorted_names)
        ax4.grid(True, alpha=0.3)

        # Add accuracy values
        for i, (name, acc) in enumerate(zip(sorted_names, sorted_accuracies)):
            ax4.text(len(sorted_names) - i + 0.1, i, f'{acc:.3f}',
                     ha='left', va='center', fontweight='bold')

        # 5. Summary statistics (bottom, spans full width)
        ax5 = fig.add_subplot(gs[3, :])
        ax5.axis('off')

        # Calculate comprehensive statistics
        acc_stats = {
            'count': len(accuracies),
            'mean': np.mean(accuracies),
            'std': np.std(accuracies),
            'min': np.min(accuracies),
            'max': np.max(accuracies),
            'range': np.max(accuracies) - np.min(accuracies),
            'median': np.median(accuracies)
        }

        summary_text = f"""
COMPREHENSIVE MODEL EVALUATION SUMMARY

Total Models Evaluated: {acc_stats['count']}     Best Model: {sorted_names[0]} ({acc_stats['max']:.4f})     Worst Model: {sorted_names[-1]} ({acc_stats['min']:.4f})

Accuracy Statistics:  Mean: {acc_stats['mean']:.4f}  |  Std: {acc_stats['std']:.4f}  |  Median: {acc_stats['median']:.4f}  |  Range: {acc_stats['range']:.4f}

Top 3 Performers:  1. {sorted_names[0]}: {sorted_accuracies[0]:.4f}  |  2. {sorted_names[1]}: {sorted_accuracies[1]:.4f}  |  3. {sorted_names[2]}: {sorted_accuracies[2]:.4f}

Performance Categories:  Excellent (>90%): {sum(1 for acc in accuracies if acc > 0.9)}  |  Good (80-90%): {sum(1 for acc in accuracies if 0.8 <= acc <= 0.9)}  |  Fair (<80%): {sum(1 for acc in accuracies if acc < 0.8)}
        """

        ax5.text(0.5, 0.5, summary_text, transform=ax5.transAxes,
                 fontsize=14, ha='center', va='center', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))

        # Save dashboard
        dashboard_path = os.path.join(visualization_dir, 'model_performance_dashboard.png')
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        logging.info(f"Saved comprehensive performance dashboard: {dashboard_path}")

    except Exception as e:
        logging.error(f"Error creating model performance dashboard: {str(e)}")


def generate_model_evaluation_summary(model_results, output_dir):
    """
    Generate comprehensive model evaluation outputs.
    """
    try:
        logging.info("Generating comprehensive model evaluation summary...")

        # Create visualization directory
        eval_viz_dir = os.path.join(output_dir, "model_evaluation")
        os.makedirs(eval_viz_dir, exist_ok=True)

        # 1. Create comparison visualizations
        compare_models_performance(model_results, eval_viz_dir)

        # 2. Create performance dashboard
        create_model_performance_dashboard(model_results, eval_viz_dir)

        # 3. Save detailed evaluation report
        save_evaluation_report(model_results, output_dir)

        # 4. Save detailed comparison table
        save_detailed_comparison_table(model_results, eval_viz_dir)

        # 5. Create individual model summaries
        create_individual_model_summaries(model_results, eval_viz_dir)

        logging.info("Model evaluation summary generation completed")

    except Exception as e:
        logging.error(f"Error generating model evaluation summary: {str(e)}")


def create_individual_model_summaries(model_results, visualization_dir):
    """
    Create individual summary files for each model.
    """
    try:
        summaries_dir = os.path.join(visualization_dir, "individual_summaries")
        os.makedirs(summaries_dir, exist_ok=True)

        for model_name, results in model_results.items():
            # Create individual model summary
            summary_path = os.path.join(summaries_dir, f"{model_name}_summary.txt")

            with open(summary_path, 'w') as f:
                f.write(f"MODEL EVALUATION SUMMARY - {model_name}\n")
                f.write("=" * 50 + "\n\n")

                f.write("PERFORMANCE METRICS:\n")
                f.write(f"  Accuracy: {results['accuracy']:.4f}\n")

                if 'precision_weighted' in results:
                    f.write(f"  Precision (weighted): {results['precision_weighted']:.4f}\n")
                    f.write(f"  Recall (weighted): {results['recall_weighted']:.4f}\n")
                    f.write(f"  F1-Score (weighted): {results['f1_weighted']:.4f}\n")

                    if 'precision_macro' in results:
                        f.write(f"  Precision (macro): {results['precision_macro']:.4f}\n")
                        f.write(f"  Recall (macro): {results['recall_macro']:.4f}\n")
                        f.write(f"  F1-Score (macro): {results['f1_macro']:.4f}\n")

                # Add ranking information
                all_accuracies = [r['accuracy'] for r in model_results.values()]
                sorted_accuracies = sorted(all_accuracies, reverse=True)
                rank = sorted_accuracies.index(results['accuracy']) + 1

                f.write(f"\nRANKING:\n")
                f.write(f"  Rank: {rank} out of {len(model_results)} models\n")
                f.write(f"  Percentile: {100 - (rank / len(model_results)) * 100:.1f}%\n")

                # Performance category
                acc = results['accuracy']
                if acc >= 0.9:
                    category = "Excellent"
                elif acc >= 0.8:
                    category = "Good"
                elif acc >= 0.7:
                    category = "Fair"
                else:
                    category = "Poor"

                f.write(f"  Performance Category: {category}\n")

                # Comparison with best and average
                best_acc = max(all_accuracies)
                avg_acc = np.mean(all_accuracies)

                f.write(f"\nCOMPARATIVE ANALYSIS:\n")
                f.write(f"  Gap from best model: {best_acc - acc:.4f}\n")
                f.write(f"  Difference from average: {acc - avg_acc:+.4f}\n")

                if acc >= avg_acc:
                    f.write(f"  Performance: Above average\n")
                else:
                    f.write(f"  Performance: Below average\n")

        logging.info(f"Created individual model summaries in {summaries_dir}")

    except Exception as e:
        logging.error(f"Error creating individual model summaries: {str(e)}")