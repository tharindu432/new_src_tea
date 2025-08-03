import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import cv2
import logging
from sklearn.metrics import confusion_matrix, classification_report
from config import Config


class ComprehensiveVisualizer:
    """Advanced visualization system for tea particle analysis results."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def visualize_performance(self, model_names, accuracies, reports, y_test, y_pred_dict, visualization_dir):
        """
        Create comprehensive model performance visualizations.
        """
        try:
            # Create performance comparison dashboard
            self.create_performance_dashboard(model_names, accuracies, reports, visualization_dir)

            # Create confusion matrices for all models
            self.create_confusion_matrices(model_names, y_test, y_pred_dict, visualization_dir)

            # Create detailed classification reports
            self.create_classification_report_heatmaps(model_names, reports, visualization_dir)

            # Create model comparison analysis
            self.create_model_comparison_analysis(model_names, accuracies, reports, visualization_dir)

            # Create performance metrics radar chart
            self.create_performance_radar_chart(model_names, reports, visualization_dir)

            self.logger.info("Created comprehensive performance visualizations")

        except Exception as e:
            self.logger.error(f"Error creating performance visualizations: {str(e)}")

    def create_performance_dashboard(self, model_names, accuracies, reports, visualization_dir):
        """
        Create a comprehensive performance dashboard.
        """
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Model Performance Dashboard', fontsize=20, fontweight='bold')

            # 1. Accuracy comparison bar chart
            bars = axes[0, 0].bar(model_names, accuracies,
                                  color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            axes[0, 0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

            # 2. Precision comparison
            precisions = []
            for model_name in model_names:
                if model_name in reports:
                    precision = reports[model_name]['weighted avg']['precision']
                    precisions.append(precision)
                else:
                    precisions.append(0)

            axes[0, 1].bar(model_names, precisions, color='lightgreen', alpha=0.8)
            axes[0, 1].set_title('Model Precision Comparison', fontsize=14, fontweight='bold')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].tick_params(axis='x', rotation=45)

            # 3. Recall comparison
            recalls = []
            for model_name in model_names:
                if model_name in reports:
                    recall = reports[model_name]['weighted avg']['recall']
                    recalls.append(recall)
                else:
                    recalls.append(0)

            axes[0, 2].bar(model_names, recalls, color='lightcoral', alpha=0.8)
            axes[0, 2].set_title('Model Recall Comparison', fontsize=14, fontweight='bold')
            axes[0, 2].set_ylabel('Recall')
            axes[0, 2].set_ylim(0, 1)
            axes[0, 2].tick_params(axis='x', rotation=45)

            # 4. F1-Score comparison
            f1_scores = []
            for model_name in model_names:
                if model_name in reports:
                    f1 = reports[model_name]['weighted avg']['f1-score']
                    f1_scores.append(f1)
                else:
                    f1_scores.append(0)

            axes[1, 0].bar(model_names, f1_scores, color='gold', alpha=0.8)
            axes[1, 0].set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
            axes[1, 0].set_ylabel('F1-Score')
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].tick_params(axis='x', rotation=45)

            # 5. Combined metrics comparison
            metrics_df = pd.DataFrame({
                'Model': model_names,
                'Accuracy': accuracies,
                'Precision': precisions,
                'Recall': recalls,
                'F1-Score': f1_scores
            })

            x = np.arange(len(model_names))
            width = 0.15

            axes[1, 1].bar(x - 1.5 * width, accuracies, width, label='Accuracy', alpha=0.8)
            axes[1, 1].bar(x - 0.5 * width, precisions, width, label='Precision', alpha=0.8)
            axes[1, 1].bar(x + 0.5 * width, recalls, width, label='Recall', alpha=0.8)
            axes[1, 1].bar(x + 1.5 * width, f1_scores, width, label='F1-Score', alpha=0.8)

            axes[1, 1].set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(model_names, rotation=45)
            axes[1, 1].legend()
            axes[1, 1].set_ylim(0, 1)

            # 6. Performance ranking
            ranking = sorted(zip(model_names, accuracies), key=lambda x: x[1], reverse=True)
            ranks, scores = zip(*ranking)

            axes[1, 2].barh(range(len(ranks)), scores, color='lightblue', alpha=0.8)
            axes[1, 2].set_yticks(range(len(ranks)))
            axes[1, 2].set_yticklabels(ranks)
            axes[1, 2].set_xlabel('Accuracy')
            axes[1, 2].set_title('Model Ranking by Accuracy', fontsize=14, fontweight='bold')
            axes[1, 2].invert_yaxis()

            # Add rank numbers
            for i, (rank, score) in enumerate(ranking):
                axes[1, 2].text(score + 0.01, i, f'#{i + 1}',
                                va='center', fontweight='bold')

            plt.tight_layout()

            # Save dashboard
            output_path = os.path.join(visualization_dir, 'performance_dashboard.png')
            plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
            plt.close()

            # Save metrics CSV
            csv_path = os.path.join(visualization_dir, 'model_metrics.csv')
            metrics_df.to_csv(csv_path, index=False)

            self.logger.info(f"Saved performance dashboard: {output_path}")
            self.logger.info(f"Saved metrics CSV: {csv_path}")

        except Exception as e:
            self.logger.error(f"Error creating performance dashboard: {str(e)}")

    def create_confusion_matrices(self, model_names, y_test, y_pred_dict, visualization_dir):
        """
        Create detailed confusion matrices for all models.
        """
        try:
            n_models = len(model_names)
            cols = min(3, n_models)
            rows = (n_models + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
            fig.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold')

            if n_models == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.reshape(1, -1)

            for idx, model_name in enumerate(model_names):
                row = idx // cols
                col = idx % cols

                if model_name in y_pred_dict:
                    y_pred = y_pred_dict[model_name]
                    cm = confusion_matrix(y_test, y_pred)

                    # Normalize confusion matrix
                    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

                    # Create heatmap
                    im = axes[row, col].imshow(cm_normalized, interpolation='nearest', cmap='Blues')
                    axes[row, col].set_title(f'{model_name}\nAccuracy: {np.trace(cm) / np.sum(cm):.3f}',
                                             fontweight='bold')

                    # Add text annotations
                    thresh = cm_normalized.max() / 2.
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            axes[row, col].text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                                                ha="center", va="center",
                                                color="white" if cm_normalized[i, j] > thresh else "black",
                                                fontsize=10)

                    axes[row, col].set_xlabel('Predicted Label')
                    axes[row, col].set_ylabel('True Label')

                    # Add colorbar
                    plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

            # Hide empty subplots
            for idx in range(n_models, rows * cols):
                row = idx // cols
                col = idx % cols
                axes[row, col].axis('off')

            plt.tight_layout()

            # Save confusion matrices
            output_path = os.path.join(visualization_dir, 'confusion_matrices_comparison.png')
            plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Saved confusion matrices: {output_path}")

        except Exception as e:
            self.logger.error(f"Error creating confusion matrices: {str(e)}")

    def create_classification_report_heatmaps(self, model_names, reports, visualization_dir):
        """
        Create heatmaps from classification reports.
        """
        try:
            # Extract metrics for all models
            all_metrics = []
            classes = []

            for model_name in model_names:
                if model_name in reports:
                    report = reports[model_name]
                    model_metrics = []

                    for class_name, metrics in report.items():
                        if class_name not in ['accuracy', 'macro avg', 'weighted avg'] and isinstance(metrics, dict):
                            if not classes or class_name not in classes:
                                classes.append(class_name)
                            model_metrics.append([
                                metrics['precision'],
                                metrics['recall'],
                                metrics['f1-score']
                            ])

                    if model_metrics:
                        all_metrics.append(model_metrics)

            if not all_metrics or not classes:
                self.logger.warning("No classification report data available for heatmap")
                return

            # Create heatmap for each metric
            metrics_names = ['Precision', 'Recall', 'F1-Score']

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Classification Metrics Heatmaps', fontsize=16, fontweight='bold')

            for metric_idx, metric_name in enumerate(metrics_names):
                # Prepare data matrix
                data_matrix = []
                for model_idx, model_name in enumerate(model_names):
                    if model_idx < len(all_metrics):
                        metric_values = [metrics[metric_idx] for metrics in all_metrics[model_idx]]
                        data_matrix.append(metric_values)

                if data_matrix:
                    data_df = pd.DataFrame(data_matrix, index=model_names[:len(data_matrix)], columns=classes)

                    # Create heatmap
                    sns.heatmap(data_df, annot=True, fmt='.3f', cmap='YlOrRd',
                                ax=axes[metric_idx], cbar_kws={'label': metric_name})
                    axes[metric_idx].set_title(f'{metric_name} by Class and Model')
                    axes[metric_idx].set_xlabel('Tea Classes')
                    axes[metric_idx].set_ylabel('Models')

            plt.tight_layout()

            # Save heatmaps
            output_path = os.path.join(visualization_dir, 'classification_metrics_heatmaps.png')
            plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Saved classification heatmaps: {output_path}")

        except Exception as e:
            self.logger.error(f"Error creating classification heatmaps: {str(e)}")

    def create_model_comparison_analysis(self, model_names, accuracies, reports, visualization_dir):
        """
        Create detailed model comparison analysis.
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Detailed Model Comparison Analysis', fontsize=16, fontweight='bold')

            # 1. Accuracy vs Model complexity (simplified representation)
            model_complexity = {'RandomForest': 3, 'SVM': 2, 'NeuralNetwork': 4, 'XGBoost': 3, 'Ensemble': 5}
            complexities = [model_complexity.get(name, 1) for name in model_names]

            scatter = axes[0, 0].scatter(complexities, accuracies,
                                         s=100, alpha=0.7, c=range(len(model_names)), cmap='viridis')

            for i, name in enumerate(model_names):
                axes[0, 0].annotate(name, (complexities[i], accuracies[i]),
                                    xytext=(5, 5), textcoords='offset points', fontsize=10)

            axes[0, 0].set_xlabel('Model Complexity (Relative)')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_title('Accuracy vs Model Complexity')
            axes[0, 0].grid(True, alpha=0.3)

            # 2. Performance consistency analysis
            if len(model_names) > 1:
                # Calculate performance variance across different metrics
                performance_data = []
                for model_name in model_names:
                    if model_name in reports:
                        report = reports[model_name]
                        if 'weighted avg' in report:
                            metrics = [
                                report['weighted avg']['precision'],
                                report['weighted avg']['recall'],
                                report['weighted avg']['f1-score']
                            ]
                            performance_data.append(metrics)

                if performance_data:
                    consistency_scores = [np.std(metrics) for metrics in performance_data]
                    axes[0, 1].bar(model_names[:len(consistency_scores)], consistency_scores,
                                   color='lightsteelblue', alpha=0.8)
                    axes[0, 1].set_title('Model Consistency (Lower is Better)')
                    axes[0, 1].set_ylabel('Standard Deviation of Metrics')
                    axes[0, 1].tick_params(axis='x', rotation=45)

            # 3. Class-wise performance comparison
            if model_names and reports:
                # Get all unique classes
                all_classes = set()
                for model_name in model_names:
                    if model_name in reports:
                        for class_name in reports[model_name].keys():
                            if class_name not in ['accuracy', 'macro avg', 'weighted avg'] and isinstance(
                                    reports[model_name][class_name], dict):
                                all_classes.add(class_name)

                if all_classes:
                    all_classes = sorted(list(all_classes))

                    # Create class-wise F1-score comparison
                    class_f1_data = []
                    for class_name in all_classes:
                        class_f1_scores = []
                        for model_name in model_names:
                            if model_name in reports and class_name in reports[model_name]:
                                f1_score = reports[model_name][class_name]['f1-score']
                                class_f1_scores.append(f1_score)
                            else:
                                class_f1_scores.append(0)
                        class_f1_data.append(class_f1_scores)

                    # Plot grouped bar chart
                    x = np.arange(len(all_classes))
                    width = 0.15

                    for i, model_name in enumerate(model_names):
                        model_scores = [class_f1_data[j][i] for j in range(len(all_classes))]
                        axes[1, 0].bar(x + i * width, model_scores, width, label=model_name, alpha=0.8)

                    axes[1, 0].set_xlabel('Tea Classes')
                    axes[1, 0].set_ylabel('F1-Score')
                    axes[1, 0].set_title('Class-wise F1-Score Comparison')
                    axes[1, 0].set_xticks(x + width * (len(model_names) - 1) / 2)
                    axes[1, 0].set_xticklabels(all_classes, rotation=45)
                    axes[1, 0].legend()
                    axes[1, 0].grid(True, alpha=0.3)

            # 4. Model performance summary statistics
            summary_stats = []
            for model_name in model_names:
                if model_name in reports and 'weighted avg' in reports[model_name]:
                    precision = reports[model_name]['weighted avg']['precision']
                    recall = reports[model_name]['weighted avg']['recall']
                    f1 = reports[model_name]['weighted avg']['f1-score']

                    summary_stats.append({
                        'Model': model_name,
                        'Precision': precision,
                        'Recall': recall,
                        'F1-Score': f1,
                        'Accuracy': accuracies[model_names.index(model_name)]
                    })

            if summary_stats:
                stats_df = pd.DataFrame(summary_stats)

                # Create table visualization
                axes[1, 1].axis('tight')
                axes[1, 1].axis('off')

                table_data = []
                for _, row in stats_df.iterrows():
                    table_data.append([
                        row['Model'],
                        f"{row['Accuracy']:.3f}",
                        f"{row['Precision']:.3f}",
                        f"{row['Recall']:.3f}",
                        f"{row['F1-Score']:.3f}"
                    ])

                table = axes[1, 1].table(cellText=table_data,
                                         colLabels=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
                                         cellLoc='center',
                                         loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 2)

                # Color code the best performances
                for i in range(1, 5):  # Skip model name column
                    col_values = [float(table_data[j][i]) for j in range(len(table_data))]
                    max_val = max(col_values)
                    for j in range(len(table_data)):
                        if float(table_data[j][i]) == max_val:
                            table[(j + 1, i)].set_facecolor('#90EE90')  # Light green for best

                axes[1, 1].set_title('Performance Summary Table')

            plt.tight_layout()

            # Save comparison analysis
            output_path = os.path.join(visualization_dir, 'model_comparison_analysis.png')
            plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Saved model comparison analysis: {output_path}")

        except Exception as e:
            self.logger.error(f"Error creating model comparison analysis: {str(e)}")

    def create_performance_radar_chart(self, model_names, reports, visualization_dir):
        """
        Create radar chart for model performance comparison.
        """
        try:
            # Prepare data
            metrics = ['Precision', 'Recall', 'F1-Score']

            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

            # Number of variables
            N = len(metrics)

            # Angle for each metric
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Complete the circle

            # Colors for different models
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

            for i, model_name in enumerate(model_names):
                if model_name in reports and 'weighted avg' in reports[model_name]:
                    values = [
                        reports[model_name]['weighted avg']['precision'],
                        reports[model_name]['weighted avg']['recall'],
                        reports[model_name]['weighted avg']['f1-score']
                    ]
                    values += values[:1]  # Complete the circle

                    ax.plot(angles, values, 'o-', linewidth=2,
                            label=model_name, color=colors[i % len(colors)])
                    ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

            # Add labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
            ax.grid(True)

            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.title('Model Performance Radar Chart', size=16, fontweight='bold', pad=20)

            # Save radar chart
            output_path = os.path.join(visualization_dir, 'performance_radar_chart.png')
            plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Saved performance radar chart: {output_path}")

        except Exception as e:
            self.logger.error(f"Error creating performance radar chart: {str(e)}")

    def create_overlap_summary_visualization(self, overlap_data, visualization_dir):
        """
        Create comprehensive overlap analysis summary.
        """
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Overlap Analysis Summary', fontsize=16, fontweight='bold')

            if not overlap_data:
                axes[0, 0].text(0.5, 0.5, 'No overlap data available',
                                ha='center', va='center', transform=axes[0, 0].transAxes)
                return

            # Convert overlap data to DataFrame
            df = pd.DataFrame(overlap_data)

            # 1. Distribution of overlap counts
            axes[0, 0].hist(df['overlap_count'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_xlabel('Overlap Count')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Distribution of Overlap Counts')
            axes[0, 0].grid(True, alpha=0.3)

            # Add statistics
            mean_overlap = df['overlap_count'].mean()
            std_overlap = df['overlap_count'].std()
            axes[0, 0].axvline(mean_overlap, color='red', linestyle='--',
                               label=f'Mean: {mean_overlap:.1f}')
            axes[0, 0].axvline(mean_overlap + std_overlap, color='orange', linestyle='--',
                               label=f'+1σ: {mean_overlap + std_overlap:.1f}')
            axes[0, 0].legend()

            # 2. Train vs Test overlap comparison
            if 'dataset' in df.columns:
                train_data = df[df['dataset'] == 'train']['overlap_count']
                test_data = df[df['dataset'] == 'test']['overlap_count']

                axes[0, 1].boxplot([train_data, test_data], labels=['Train', 'Test'])
                axes[0, 1].set_title('Overlap Count Distribution by Dataset')
                axes[0, 1].set_ylabel('Overlap Count')
                axes[0, 1].grid(True, alpha=0.3)

            # 3. Cumulative distribution
            sorted_overlaps = np.sort(df['overlap_count'])
            cumulative = np.arange(1, len(sorted_overlaps) + 1) / len(sorted_overlaps)

            axes[0, 2].plot(sorted_overlaps, cumulative, marker='o', linestyle='-', alpha=0.7)
            axes[0, 2].set_xlabel('Overlap Count')
            axes[0, 2].set_ylabel('Cumulative Probability')
            axes[0, 2].set_title('Cumulative Distribution of Overlaps')
            axes[0, 2].grid(True, alpha=0.3)

            # 4. Overlap count vs sample index
            axes[1, 0].plot(df.index, df['overlap_count'], marker='o', alpha=0.7)
            axes[1, 0].set_xlabel('Sample Index')
            axes[1, 0].set_ylabel('Overlap Count')
            axes[1, 0].set_title('Overlap Count by Sample')
            axes[1, 0].grid(True, alpha=0.3)

            # 5. Summary statistics table
            stats = df['overlap_count'].describe()

            axes[1, 1].axis('tight')
            axes[1, 1].axis('off')

            stats_data = [
                ['Count', f"{stats['count']:.0f}"],
                ['Mean', f"{stats['mean']:.2f}"],
                ['Std', f"{stats['std']:.2f}"],
                ['Min', f"{stats['min']:.0f}"],
                ['25%', f"{stats['25%']:.0f}"],
                ['50%', f"{stats['50%']:.0f}"],
                ['75%', f"{stats['75%']:.0f}"],
                ['Max', f"{stats['max']:.0f}"]
            ]

            table = axes[1, 1].table(cellText=stats_data,
                                     colLabels=['Statistic', 'Value'],
                                     cellLoc='center',
                                     loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 2)
            axes[1, 1].set_title('Overlap Statistics Summary')

            # 6. Overlap categories
            # Categorize overlaps: Low (1-2), Medium (3-5), High (6+)
            categories = []
            for count in df['overlap_count']:
                if count <= 2:
                    categories.append('Low (1-2)')
                elif count <= 5:
                    categories.append('Medium (3-5)')
                else:
                    categories.append('High (6+)')

            category_counts = pd.Series(categories).value_counts()

            axes[1, 2].pie(category_counts.values, labels=category_counts.index,
                           autopct='%1.1f%%', startangle=90)
            axes[1, 2].set_title('Overlap Categories Distribution')

            plt.tight_layout()

            # Save overlap summary
            output_path = os.path.join(visualization_dir, 'overlap_analysis_summary.png')
            plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Saved overlap analysis summary: {output_path}")

        except Exception as e:
            self.logger.error(f"Error creating overlap summary visualization: {str(e)}")

    def create_feature_analysis_visualization(self, X_train, X_test, feature_names, visualization_dir):
        """
        Create feature analysis and distribution visualizations.
        """
        try:
            if X_train is None or len(X_train) == 0:
                self.logger.warning("No training data available for feature analysis")
                return

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Feature Analysis and Distribution', fontsize=16, fontweight='bold')

            # 1. Feature correlation heatmap
            if X_train.shape[1] > 1:
                correlation_matrix = np.corrcoef(X_train.T)
                im = axes[0, 0].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
                axes[0, 0].set_title('Feature Correlation Matrix')
                axes[0, 0].set_xlabel('Features')
                axes[0, 0].set_ylabel('Features')
                plt.colorbar(im, ax=axes[0, 0])
            else:
                axes[0, 0].text(0.5, 0.5, 'Insufficient features\nfor correlation analysis',
                                ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Feature Correlation Matrix')

            # 2. Feature distributions
            if X_train.shape[1] >= 3:
                for i in range(min(3, X_train.shape[1])):
                    axes[0, i].hist(X_train[:, i], bins=30, alpha=0.7,
                                    color='skyblue', edgecolor='black')
                    axes[0, i].set_title(f'Feature {i + 1} Distribution')
                    axes[0, i].set_xlabel('Feature Value')
                    axes[0, i].set_ylabel('Frequency')
                    axes[0, i].grid(True, alpha=0.3)

            # Fill remaining subplots with additional analysis
            if X_train.shape[1] >= 1:
                # Feature statistics comparison
                train_means = np.mean(X_train, axis=0)
                test_means = np.mean(X_test, axis=0) if X_test is not None and len(X_test) > 0 else np.zeros_like(
                    train_means)

                x = np.arange(min(10, len(train_means)))  # Show first 10 features
                width = 0.35

                axes[1, 0].bar(x - width / 2, train_means[:len(x)], width, label='Train', alpha=0.8)
                axes[1, 0].bar(x + width / 2, test_means[:len(x)], width, label='Test', alpha=0.8)
                axes[1, 0].set_xlabel('Feature Index')
                axes[1, 0].set_ylabel('Mean Value')
                axes[1, 0].set_title('Feature Means Comparison')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

                # Feature variance analysis
                train_vars = np.var(X_train, axis=0)
                axes[1, 1].bar(range(min(10, len(train_vars))), train_vars[:10], alpha=0.7)
                axes[1, 1].set_xlabel('Feature Index')
                axes[1, 1].set_ylabel('Variance')
                axes[1, 1].set_title('Feature Variance Analysis')
                axes[1, 1].grid(True, alpha=0.3)

                # Feature range analysis
                train_ranges = np.max(X_train, axis=0) - np.min(X_train, axis=0)
                axes[1, 2].bar(range(min(10, len(train_ranges))), train_ranges[:10],
                               alpha=0.7, color='lightcoral')
                axes[1, 2].set_xlabel('Feature Index')
                axes[1, 2].set_ylabel('Range')
                axes[1, 2].set_title('Feature Range Analysis')
                axes[1, 2].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save feature analysis
            output_path = os.path.join(visualization_dir, 'feature_analysis.png')
            plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Saved feature analysis: {output_path}")

        except Exception as e:
            self.logger.error(f"Error creating feature analysis visualization: {str(e)}")


def visualize_performance(model_names, accuracies, reports, y_test, y_pred_dict, visualization_dir):
    """
    Wrapper function for backward compatibility.
    """
    visualizer = ComprehensiveVisualizer()
    visualizer.visualize_performance(model_names, accuracies, reports, y_test, y_pred_dict, visualization_dir)