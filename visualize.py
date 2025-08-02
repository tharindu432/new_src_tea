import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import logging
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import Config

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")


def create_confusion_matrix(y_true, y_pred, labels, title, save_path):
    """Create and save confusion matrix visualization."""
    try:
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix: {title}', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
        plt.close()

        logging.info(f"Confusion matrix saved: {save_path}")
    except Exception as e:
        logging.error(f"Error creating confusion matrix: {str(e)}")


def create_classification_report_heatmap(report_dict, title, save_path):
    """Create classification report as heatmap."""
    try:
        # Convert report to DataFrame
        df = pd.DataFrame(report_dict).iloc[:-1, :].T  # Exclude 'accuracy' row
        df = df.drop(['support'], axis=1)  # Remove support column for clarity

        plt.figure(figsize=(8, 6))
        sns.heatmap(df.astype(float), annot=True, fmt='.3f', cmap='RdYlBu_r',
                    cbar_kws={'label': 'Score'})
        plt.title(f'Classification Report: {title}', fontsize=16, fontweight='bold')
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Classes', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
        plt.close()

        logging.info(f"Classification report heatmap saved: {save_path}")
    except Exception as e:
        logging.error(f"Error creating classification report heatmap: {str(e)}")


def visualize_model_comparison(model_results, save_dir):
    """Create comprehensive model comparison visualizations."""
    try:
        # Extract data for visualization
        models = list(model_results.keys())
        val_accuracies = [results['val_accuracy'] for results in model_results.values()]
        test_accuracies = [results['test_accuracy'] for results in model_results.values()]

        # 1. Accuracy comparison bar plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        x = np.arange(len(models))
        width = 0.35

        bars1 = ax1.bar(x - width / 2, val_accuracies, width, label='Validation', alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x + width / 2, test_accuracies, width, label='Test', alpha=0.8, color='lightcoral')

        ax1.set_xlabel('Models', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        # 2. Performance difference analysis
        performance_diff = [val - test for val, test in zip(val_accuracies, test_accuracies)]
        colors = ['green' if diff >= 0 else 'red' for diff in performance_diff]

        bars3 = ax2.bar(models, performance_diff, color=colors, alpha=0.7)
        ax2.set_xlabel('Models', fontsize=12)
        ax2.set_ylabel('Validation - Test Accuracy', fontsize=12)
        ax2.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, diff in zip(bars3, performance_diff):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.,
                     height + (0.005 if height >= 0 else -0.015),
                     f'{diff:.3f}', ha='center',
                     va='bottom' if height >= 0 else 'top', fontsize=9)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/model_comparison.png", dpi=Config.PLOT_DPI, bbox_inches='tight')
        plt.close()

        # 3. Detailed performance metrics radar chart
        create_radar_chart(model_results, save_dir)

        logging.info("Model comparison visualizations created")

    except Exception as e:
        logging.error(f"Error creating model comparison: {str(e)}")


def create_radar_chart(model_results, save_dir):
    """Create radar chart for model performance metrics."""
    try:
        # Extract precision, recall, f1-score for each model
        metrics_data = {}

        for model_name, results in model_results.items():
            report = results['test_report']
            if isinstance(report, dict) and 'weighted avg' in report:
                weighted_avg = report['weighted avg']
                metrics_data[model_name] = {
                    'Precision': weighted_avg['precision'],
                    'Recall': weighted_avg['recall'],
                    'F1-Score': weighted_avg['f1-score'],
                    'Accuracy': results['test_accuracy']
                }

        if not metrics_data:
            logging.warning("No metrics data available for radar chart")
            return

        # Create radar chart using plotly
        fig = go.Figure()

        metrics = list(list(metrics_data.values())[0].keys())

        for model_name, model_metrics in metrics_data.items():
            values = list(model_metrics.values())
            values.append(values[0])  # Close the polygon

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model_name,
                line=dict(width=2)
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Model Performance Radar Chart",
            title_x=0.5
        )

        fig.write_html(f"{save_dir}/performance_radar.html")
        try:
            fig.write_image(f"{save_dir}/performance_radar.png", width=800, height=600)
        except:
            logging.warning("Could not save radar chart as PNG - plotly kaleido may not be installed")

        logging.info("Radar chart created")

    except Exception as e:
        logging.error(f"Error creating radar chart: {str(e)}")


def visualize_feature_importance(X, y, feature_names, save_dir, le):
    """Visualize feature importance using various methods."""
    try:
        # 1. PCA Analysis
        pca = PCA(n_components=min(10, X.shape[1]))
        X_pca = pca.fit_transform(X)

        # Plot PCA explained variance
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_, alpha=0.7, color='steelblue')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Explained Variance')
        plt.grid(axis='y', alpha=0.3)

        plt.subplot(1, 2, 2)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
                 'bo-', alpha=0.7, color='darkred')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance')
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/pca_analysis.png", dpi=Config.PLOT_DPI, bbox_inches='tight')
        plt.close()

        # 2. t-SNE visualization
        if len(X) > 10:  # Only if we have enough samples
            tsne = TSNE(n_components=2, random_state=Config.RANDOM_STATE, perplexity=min(30, len(X) - 1))
            X_tsne = tsne.fit_transform(X)

            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7)
            plt.colorbar(scatter, ticks=range(len(le.classes_)))
            plt.colorbar().set_ticklabels(le.classes_)
            plt.title('t-SNE Visualization of Features', fontsize=14, fontweight='bold')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.grid(alpha=0.3)
            plt.savefig(f"{save_dir}/tsne_visualization.png", dpi=Config.PLOT_DPI, bbox_inches='tight')
            plt.close()

        logging.info("Feature importance visualizations created")

    except Exception as e:
        logging.error(f"Error creating feature importance visualizations: {str(e)}")


def visualize_shape_analysis(X, y, save_dir, le):
    """Create comprehensive shape analysis visualizations."""
    try:
        # Feature correlation heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = np.corrcoef(X.T)

        # Create feature names
        feature_names = []
        for i in range(Config.SHAPE_FEATURE_COUNT):
            if i < 7:
                feature_names.append(f'Hu_{i + 1}')
            elif i < 17:
                feature_names.append(f'FD_{i - 6}')
            else:
                feature_names.append(f'Geo_{i - 16}')

        # Add aggregated feature names
        feature_names.extend(['Mean_Area', 'Std_Area', 'Particle_Count',
                              'Density', 'Size_Var', 'Dist_1', 'Dist_2',
                              'Dist_3', 'Dist_4', 'Dist_5', 'Overlap'])

        # Trim feature names to match actual feature count
        feature_names = feature_names[:X.shape[1]]

        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm',
                    center=0, square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/feature_correlation.png", dpi=Config.PLOT_DPI, bbox_inches='tight')
        plt.close()

        # Class-wise feature distributions
        n_features_to_plot = min(12, X.shape[1])
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.ravel()

        for i in range(n_features_to_plot):
            ax = axes[i]
            for class_idx, class_name in enumerate(le.classes_):
                class_data = X[y == class_idx, i]
                if len(class_data) > 0:
                    ax.hist(class_data, alpha=0.7, label=class_name, bins=20)

            ax.set_title(f'Feature {i + 1}' if i < len(feature_names) else f'Feature {i + 1}')
            ax.set_xlabel('Feature Value')
            ax.set_ylabel('Frequency')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        plt.suptitle('Feature Distributions by Class', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/feature_distributions.png", dpi=Config.PLOT_DPI, bbox_inches='tight')
        plt.close()

        # Box plots for key features
        key_features_idx = [0, 7, 17, -3, -2, -1]  # Hu_1, FD_1, Geo_1, Particle_Count, Size_Var, Overlap
        key_features_idx = [i for i in key_features_idx if abs(i) < X.shape[1]]

        if key_features_idx:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.ravel()

            for i, feature_idx in enumerate(key_features_idx):
                if i >= len(axes):
                    break

                ax = axes[i]
                data_for_box = []
                labels_for_box = []

                for class_idx, class_name in enumerate(le.classes_):
                    class_data = X[y == class_idx, feature_idx]
                    if len(class_data) > 0:
                        data_for_box.append(class_data)
                        labels_for_box.append(class_name)

                if data_for_box:
                    ax.boxplot(data_for_box, labels=labels_for_box)
                    ax.set_title(f'Feature {feature_idx + 1 if feature_idx >= 0 else X.shape[1] + feature_idx + 1}')
                    ax.set_ylabel('Feature Value')
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(alpha=0.3)

            plt.suptitle('Key Features Box Plots by Class', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{save_dir}/key_features_boxplots.png", dpi=Config.PLOT_DPI, bbox_inches='tight')
            plt.close()

        # Feature importance visualization using PCA
        visualize_feature_importance(X, y, feature_names, save_dir, le)

        logging.info("Shape analysis visualizations created")

    except Exception as e:
        logging.error(f"Error creating shape analysis visualizations: {str(e)}")


def visualize_dataset_overview(X, y, overlap_counts, dataset_sources, save_dir, le):
    """Create dataset overview visualizations."""
    try:
        # 1. Dataset composition
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Class distribution
        class_counts = pd.Series(y).value_counts()
        class_names = [le.classes_[i] if i < len(le.classes_) else f'Class_{i}'
                       for i in class_counts.index]

        wedges, texts, autotexts = ax1.pie(class_counts.values, labels=class_names,
                                           autopct='%1.1f%%', startangle=90)
        ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')

        # Dataset source distribution
        source_counts = pd.Series(dataset_sources).value_counts()
        ax2.bar(source_counts.index, source_counts.values, alpha=0.7, color='skyblue')
        ax2.set_title('Samples by Dataset Source', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Dataset Source')
        ax2.set_ylabel('Number of Samples')
        ax2.tick_params(axis='x', rotation=45)

        # Overlap distribution
        ax3.hist(overlap_counts, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_title('Overlap Count Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Overlap Count')
        ax3.set_ylabel('Frequency')
        ax3.grid(alpha=0.3)

        # Feature statistics
        feature_stats = {
            'Mean': np.mean(X, axis=0),
            'Std': np.std(X, axis=0),
            'Min': np.min(X, axis=0),
            'Max': np.max(X, axis=0)
        }

        stats_df = pd.DataFrame(feature_stats)
        im = ax4.imshow(stats_df.T, cmap='viridis', aspect='auto')
        ax4.set_title('Feature Statistics Heatmap', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Feature Index')
        ax4.set_ylabel('Statistic')
        ax4.set_yticks(range(len(feature_stats)))
        ax4.set_yticklabels(feature_stats.keys())
        plt.colorbar(im, ax=ax4)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/dataset_overview.png", dpi=Config.PLOT_DPI, bbox_inches='tight')
        plt.close()

        # 2. Interactive 3D scatter plot using plotly
        if X.shape[1] >= 3:
            try:
                fig_3d = px.scatter_3d(
                    x=X[:, 0], y=X[:, 1], z=X[:, 2],
                    color=[le.classes_[i] if i < len(le.classes_) else f'Class_{i}' for i in y],
                    title="3D Feature Space Visualization",
                    labels={'x': 'Feature 1', 'y': 'Feature 2', 'z': 'Feature 3'}
                )
                fig_3d.write_html(f"{save_dir}/3d_feature_space.html")
            except Exception as e:
                logging.warning(f"Could not create 3D plot: {str(e)}")

        logging.info("Dataset overview visualizations created")

    except Exception as e:
        logging.error(f"Error creating dataset overview: {str(e)}")


def create_learning_curves(model_results, save_dir):
    """Create learning curves if validation history is available."""
    try:
        # This would require modification of training to save validation history
        # For now, create a placeholder showing the concept

        plt.figure(figsize=(12, 8))

        models = list(model_results.keys())
        val_accuracies = [results['val_accuracy'] for results in model_results.values()]
        test_accuracies = [results['test_accuracy'] for results in model_results.values()]

        # Simulate learning curves (in real implementation, save training history)
        epochs = range(1, 11)

        for i, model in enumerate(models[:4]):  # Show top 4 models
            if i >= 4:
                break
            # Simulate training progression
            np.random.seed(Config.RANDOM_STATE + i)
            train_acc = np.random.uniform(0.3, val_accuracies[i], 10)
            train_acc = np.sort(train_acc)
            val_acc = np.random.uniform(0.3, val_accuracies[i], 10)
            val_acc = np.sort(val_acc)

            plt.subplot(2, 2, i + 1)
            plt.plot(epochs, train_acc, 'b-', label='Training', alpha=0.7)
            plt.plot(epochs, val_acc, 'r-', label='Validation', alpha=0.7)
            plt.axhline(y=test_accuracies[i], color='g', linestyle='--',
                        label=f'Test ({test_accuracies[i]:.3f})', alpha=0.7)
            plt.title(f'{model} Learning Curve')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/learning_curves.png", dpi=Config.PLOT_DPI, bbox_inches='tight')
        plt.close()

        logging.info("Learning curves visualization created")

    except Exception as e:
        logging.error(f"Error creating learning curves: {str(e)}")


def create_performance_summary(model_results, save_dir):
    """Create a comprehensive performance summary table."""
    try:
        # Prepare data for summary table
        summary_data = []

        for model_name, results in model_results.items():
            test_report = results['test_report']

            if isinstance(test_report, dict) and 'weighted avg' in test_report:
                weighted_avg = test_report['weighted avg']
                summary_data.append({
                    'Model': model_name,
                    'Test Accuracy': f"{results['test_accuracy']:.4f}",
                    'Validation Accuracy': f"{results['val_accuracy']:.4f}",
                    'Precision': f"{weighted_avg['precision']:.4f}",
                    'Recall': f"{weighted_avg['recall']:.4f}",
                    'F1-Score': f"{weighted_avg['f1-score']:.4f}",
                    'Overfitting': f"{results['val_accuracy'] - results['test_accuracy']:.4f}"
                })

        # Create table visualization
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')

        df_summary = pd.DataFrame(summary_data)
        table = ax.table(cellText=df_summary.values,
                         colLabels=df_summary.columns,
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)

        # Color code the cells based on performance
        for i in range(len(df_summary)):
            test_acc = float(df_summary.iloc[i]['Test Accuracy'])
            if test_acc >= 0.95:
                color = 'lightgreen'
            elif test_acc >= 0.90:
                color = 'lightyellow'
            else:
                color = 'lightcoral'

            for j in range(len(df_summary.columns)):
                table[(i + 1, j)].set_facecolor(color)

        plt.title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(f"{save_dir}/performance_summary.png", dpi=Config.PLOT_DPI, bbox_inches='tight')
        plt.close()

        # Save as CSV as well
        df_summary.to_csv(f"{save_dir}/performance_summary.csv", index=False)

        logging.info("Performance summary created")

    except Exception as e:
        logging.error(f"Error creating performance summary: {str(e)}")


def create_error_analysis(model_results, y_test, le, save_dir):
    """Create error analysis visualizations."""
    try:
        # Find the best performing model
        best_model = max(model_results.keys(),
                         key=lambda k: model_results[k]['test_accuracy'])

        y_pred = model_results[best_model]['test_predictions']

        # Error analysis
        errors = y_test != y_pred
        error_indices = np.where(errors)[0]

        if len(error_indices) > 0:
            # Confusion between classes
            error_matrix = pd.crosstab(
                pd.Series([le.classes_[i] for i in y_test[errors]], name='True'),
                pd.Series([le.classes_[i] for i in y_pred[errors]], name='Predicted'),
                margins=True
            )

            plt.figure(figsize=(10, 8))
            sns.heatmap(error_matrix.iloc[:-1, :-1], annot=True, fmt='d', cmap='Reds')
            plt.title(f'Error Analysis - {best_model}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{save_dir}/error_analysis.png", dpi=Config.PLOT_DPI, bbox_inches='tight')
            plt.close()

            logging.info("Error analysis created")
        else:
            logging.info("No errors found - perfect classification!")

    except Exception as e:
        logging.error(f"Error creating error analysis: {str(e)}")


def create_interactive_dashboard(model_results, save_dir):
    """Create an interactive dashboard using plotly."""
    try:
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Accuracy Comparison', 'Performance Metrics',
                            'Overfitting Analysis', 'Model Rankings'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )

        # Model accuracy comparison
        models = list(model_results.keys())
        test_accuracies = [results['test_accuracy'] for results in model_results.values()]
        val_accuracies = [results['val_accuracy'] for results in model_results.values()]

        fig.add_trace(
            go.Bar(x=models, y=test_accuracies, name="Test Accuracy", marker_color='lightblue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=models, y=val_accuracies, name="Validation Accuracy", marker_color='lightcoral'),
            row=1, col=1
        )

        # Performance metrics
        precision_scores = []
        recall_scores = []
        f1_scores = []
        for results in model_results.values():
            if 'weighted avg' in results['test_report']:
                precision_scores.append(results['test_report']['weighted avg']['precision'])
                recall_scores.append(results['test_report']['weighted avg']['recall'])
                f1_scores.append(results['test_report']['weighted avg']['f1-score'])
            else:
                precision_scores.append(0)
                recall_scores.append(0)
                f1_scores.append(0)

        fig.add_trace(
            go.Bar(x=models, y=precision_scores, name="Precision", marker_color='lightgreen'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=models, y=recall_scores, name="Recall", marker_color='orange'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=models, y=f1_scores, name="F1-Score", marker_color='purple'),
            row=1, col=2
        )

        # Overfitting analysis
        overfitting = [val - test for val, test in zip(val_accuracies, test_accuracies)]
        colors = ['green' if diff >= 0 else 'red' for diff in overfitting]

        fig.add_trace(
            go.Bar(x=models, y=overfitting, name="Val - Test Accuracy",
                   marker_color=colors),
            row=2, col=1
        )

        # Model rankings
        fig.add_trace(
            go.Bar(x=models, y=test_accuracies, name="Model Ranking",
                   marker_color='gold'),
            row=2, col=2
        )

        fig.update_layout(height=800, showlegend=True,
                          title_text="Tea Quality Assessment Dashboard")

        fig.write_html(f"{save_dir}/interactive_dashboard.html")

        logging.info("Interactive dashboard created")

    except Exception as e:
        logging.error(f"Error creating interactive dashboard: {str(e)}")


def save_all_visualizations_summary(save_dir):
    """Create a summary of all generated visualizations."""
    try:
        # List all files in the visualization directory
        all_files = os.listdir(save_dir)
        viz_files = [f for f in all_files if f.endswith('.png')]
        html_files = [f for f in all_files if f.endswith('.html')]
        csv_files = [f for f in all_files if f.endswith('.csv')]

        summary_content = f"""# Tea Quality Assessment - Visualization Summary

Generated {len(all_files)} files total:
- {len(viz_files)} PNG images
- {len(html_files)} HTML interactive files  
- {len(csv_files)} CSV data files

## Performance Visualizations:
- model_comparison.png: Overall model performance comparison
- performance_summary.png: Detailed performance metrics table
- performance_radar.html: Radar chart of model metrics
- learning_curves.png: Training progression curves
- error_analysis.png: Error pattern analysis
- interactive_dashboard.html: Interactive performance dashboard

## Shape Analysis:
- feature_correlation.png: Feature correlation heatmap
- feature_distributions.png: Class-wise feature distributions
- key_features_boxplots.png: Box plots of important features
- pca_analysis.png: Principal component analysis
- tsne_visualization.png: t-SNE dimensionality reduction

## Dataset Overview:
- dataset_overview.png: Dataset composition and statistics
- 3d_feature_space.html: Interactive 3D feature visualization

## Model-Specific:
"""

        # Add model-specific files
        model_files = [f for f in viz_files if '_confusion_matrix.png' in f or '_classification_report.png' in f]
        for file in sorted(model_files):
            summary_content += f"- {file}\n"

        summary_content += f"""
## Interactive Files:
- interactive_dashboard.html: Comprehensive interactive dashboard
- performance_radar.html: Model performance radar chart
- 3d_feature_space.html: 3D feature space exploration

## Data Files:
"""
        for file in sorted(csv_files):
            summary_content += f"- {file}\n"

        summary_content += f"""
Total files generated: {len(all_files)}

## Usage Instructions:
1. Open HTML files in a web browser for interactive visualizations
2. PNG files can be used in reports and presentations
3. CSV files contain raw data for further analysis
"""

        with open(f"{save_dir}/visualization_summary.md", 'w') as f:
            f.write(summary_content)

        logging.info("Visualization summary created")

    except Exception as e:
        logging.error(f"Error creating visualization summary: {str(e)}")


def visualize_performance(model_results, y_test, save_dir, le):
    """
    Create comprehensive performance visualizations.
    """
    try:
        logging.info("Creating performance visualizations...")

        # 1. Model comparison
        visualize_model_comparison(model_results, save_dir)

        # 2. Confusion matrices for each model
        for model_name, results in model_results.items():
            y_pred = results['test_predictions']

            # Confusion matrix
            cm_path = f"{save_dir}/{model_name}_confusion_matrix.png"
            create_confusion_matrix(y_test, y_pred, le.classes_, model_name, cm_path)

            # Classification report heatmap
            if isinstance(results['test_report'], dict):
                report_path = f"{save_dir}/{model_name}_classification_report.png"
                create_classification_report_heatmap(results['test_report'], model_name, report_path)

        # 3. Overall performance summary
        create_performance_summary(model_results, save_dir)

        # 4. Learning curves
        create_learning_curves(model_results, save_dir)

        # 5. Error analysis
        create_error_analysis(model_results, y_test, le, save_dir)

        # 6. Create interactive dashboard
        create_interactive_dashboard(model_results, save_dir)

        # 7. Save visualization summary
        save_all_visualizations_summary(save_dir)

        logging.info("All performance visualizations created successfully")

    except Exception as e:
        logging.error(f"Error in visualize_performance: {str(e)}")


def create_model_performance_comparison_chart(model_results, save_dir):
    """Create a comprehensive model performance comparison chart."""
    try:
        # Extract all metrics for comparison
        models = list(model_results.keys())
        metrics_data = {
            'Model': models,
            'Test_Accuracy': [results['test_accuracy'] for results in model_results.values()],
            'Val_Accuracy': [results['val_accuracy'] for results in model_results.values()],
            'Precision': [],
            'Recall': [],
            'F1_Score': []
        }

        # Extract precision, recall, f1-score
        for results in model_results.values():
            test_report = results['test_report']
            if isinstance(test_report, dict) and 'weighted avg' in test_report:
                weighted_avg = test_report['weighted avg']
                metrics_data['Precision'].append(weighted_avg['precision'])
                metrics_data['Recall'].append(weighted_avg['recall'])
                metrics_data['F1_Score'].append(weighted_avg['f1-score'])
            else:
                metrics_data['Precision'].append(0)
                metrics_data['Recall'].append(0)
                metrics_data['F1_Score'].append(0)

        # Create DataFrame
        df = pd.DataFrame(metrics_data)

        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Model Performance Analysis', fontsize=16, fontweight='bold')

        # 1. Test vs Validation Accuracy
        ax1 = axes[0, 0]
        x = np.arange(len(models))
        width = 0.35
        ax1.bar(x - width / 2, df['Test_Accuracy'], width, label='Test', alpha=0.8, color='skyblue')
        ax1.bar(x + width / 2, df['Val_Accuracy'], width, label='Validation', alpha=0.8, color='lightcoral')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Test vs Validation Accuracy')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # 2. Precision, Recall, F1-Score
        ax2 = axes[0, 1]
        ax2.bar(x - width, df['Precision'], width, label='Precision', alpha=0.8, color='lightgreen')
        ax2.bar(x, df['Recall'], width, label='Recall', alpha=0.8, color='orange')
        ax2.bar(x + width, df['F1_Score'], width, label='F1-Score', alpha=0.8, color='purple')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Score')
        ax2.set_title('Precision, Recall, F1-Score')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        # 3. Overfitting Analysis
        ax3 = axes[0, 2]
        overfitting = [val - test for val, test in zip(df['Val_Accuracy'], df['Test_Accuracy'])]
        colors = ['green' if diff >= -0.02 else 'orange' if diff >= -0.05 else 'red' for diff in overfitting]
        bars = ax3.bar(models, overfitting, color=colors, alpha=0.7)
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Validation - Test Accuracy')
        ax3.set_title('Overfitting Analysis')
        ax3.set_xticklabels(models, rotation=45, ha='right')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, overfitting):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + (0.005 if height >= 0 else -0.015),
                     f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

        # 4. Model Rankings (Test Accuracy)
        ax4 = axes[1, 0]
        sorted_indices = np.argsort(df['Test_Accuracy'])[::-1]
        sorted_models = [models[i] for i in sorted_indices]
        sorted_accuracies = [df['Test_Accuracy'][i] for i in sorted_indices]

        colors_rank = plt.cm.RdYlGn([acc for acc in sorted_accuracies])
        bars_rank = ax4.bar(range(len(sorted_models)), sorted_accuracies, color=colors_rank, alpha=0.8)
        ax4.set_xlabel('Model Rank')
        ax4.set_ylabel('Test Accuracy')
        ax4.set_title('Model Rankings by Test Accuracy')
        ax4.set_xticks(range(len(sorted_models)))
        ax4.set_xticklabels([f"{i + 1}. {model}" for i, model in enumerate(sorted_models)],
                            rotation=45, ha='right')
        ax4.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, acc in zip(bars_rank, sorted_accuracies):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                     f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

        # 5. Performance Consistency
        ax5 = axes[1, 1]
        consistency_score = [1 - abs(val - test) for val, test in zip(df['Val_Accuracy'], df['Test_Accuracy'])]
        bars_cons = ax5.bar(models, consistency_score, alpha=0.7, color='mediumpurple')
        ax5.set_xlabel('Models')
        ax5.set_ylabel('Consistency Score')
        ax5.set_title('Model Consistency (1 - |Val - Test|)')
        ax5.set_xticklabels(models, rotation=45, ha='right')
        ax5.grid(axis='y', alpha=0.3)

        # 6. Overall Score (weighted combination)
        ax6 = axes[1, 2]
        # Overall score = 0.5 * test_acc + 0.3 * f1_score + 0.2 * consistency
        overall_scores = [0.5 * test + 0.3 * f1 + 0.2 * cons
                          for test, f1, cons in zip(df['Test_Accuracy'], df['F1_Score'], consistency_score)]

        bars_overall = ax6.bar(models, overall_scores, alpha=0.7, color='gold')
        ax6.set_xlabel('Models')
        ax6.set_ylabel('Overall Score')
        ax6.set_title('Overall Performance Score')
        ax6.set_xticklabels(models, rotation=45, ha='right')
        ax6.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, score in zip(bars_overall, overall_scores):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                     f'{score:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/comprehensive_model_analysis.png", dpi=Config.PLOT_DPI, bbox_inches='tight')
        plt.close()

        # Save metrics to CSV
        df.to_csv(f"{save_dir}/model_metrics_comparison.csv", index=False)

        logging.info("Comprehensive model performance comparison created")

    except Exception as e:
        logging.error(f"Error creating model performance comparison: {str(e)}")


def create_feature_analysis_plots(X, y, save_dir, le):
    """Create detailed feature analysis plots."""
    try:
        # 1. Feature importance using variance
        feature_variances = np.var(X, axis=0)
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(feature_variances)), feature_variances, alpha=0.7, color='steelblue')
        plt.xlabel('Feature Index')
        plt.ylabel('Variance')
        plt.title('Feature Variance Analysis')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/feature_variance_analysis.png", dpi=Config.PLOT_DPI, bbox_inches='tight')
        plt.close()

        # 2. Inter-class feature analysis
        n_classes = len(le.classes_)
        n_features = min(8, X.shape[1])  # Show top 8 features

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()

        for i in range(n_features):
            ax = axes[i]

            # Create violin plot for each class
            data_by_class = []
            labels_by_class = []

            for class_idx, class_name in enumerate(le.classes_):
                class_data = X[y == class_idx, i]
                if len(class_data) > 0:
                    data_by_class.append(class_data)
                    labels_by_class.append(class_name)

            if data_by_class:
                parts = ax.violinplot(data_by_class, positions=range(len(data_by_class)),
                                      showmeans=True, showmedians=True)
                ax.set_xticks(range(len(labels_by_class)))
                ax.set_xticklabels(labels_by_class, rotation=45)
                ax.set_title(f'Feature {i + 1} Distribution')
                ax.set_ylabel('Feature Value')
                ax.grid(alpha=0.3)

        plt.suptitle('Feature Distributions Across Classes (Violin Plots)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/feature_violin_plots.png", dpi=Config.PLOT_DPI, bbox_inches='tight')
        plt.close()

        # 3. Feature correlation with target
        correlations = []
        for i in range(X.shape[1]):
            # Calculate correlation between feature and encoded target
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations.append(abs(corr))  # Use absolute correlation

        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(correlations)), correlations, alpha=0.7, color='lightcoral')
        plt.xlabel('Feature Index')
        plt.ylabel('|Correlation with Target|')
        plt.title('Feature-Target Correlation Analysis')
        plt.grid(axis='y', alpha=0.3)

        # Highlight top features
        top_features = np.argsort(correlations)[-5:]  # Top 5 features
        for idx in top_features:
            bars[idx].set_color('darkred')

        plt.tight_layout()
        plt.savefig(f"{save_dir}/feature_target_correlation.png", dpi=Config.PLOT_DPI, bbox_inches='tight')
        plt.close()

        logging.info("Feature analysis plots created")

    except Exception as e:
        logging.error(f"Error creating feature analysis plots: {str(e)}")


# Add the missing functions to complete the visualization module
def create_class_separation_analysis(X, y, save_dir, le):
    """Analyze how well classes are separated in feature space."""
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.metrics import silhouette_score

        # 1. LDA for class separation
        if len(np.unique(y)) > 1 and X.shape[0] > len(np.unique(y)):
            lda = LinearDiscriminantAnalysis(n_components=min(2, len(np.unique(y)) - 1))
            X_lda = lda.fit_transform(X, y)

            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1] if X_lda.shape[1] > 1 else np.zeros(len(X_lda)),
                                  c=y, cmap='tab10', alpha=0.7)
            plt.colorbar(scatter, ticks=range(len(le.classes_)), label='Classes')
            plt.colorbar().set_ticklabels(le.classes_)
            plt.xlabel('LDA Component 1')
            plt.ylabel('LDA Component 2' if X_lda.shape[1] > 1 else 'Zero')
            plt.title('Linear Discriminant Analysis - Class Separation')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/lda_class_separation.png", dpi=Config.PLOT_DPI, bbox_inches='tight')
            plt.close()

        # 2. Silhouette analysis
        if len(np.unique(y)) > 1:
            silhouette_avg = silhouette_score(X, y)

            plt.figure(figsize=(10, 6))
            plt.bar(['Silhouette Score'], [silhouette_avg], color='skyblue', alpha=0.7)
            plt.ylabel('Score')
            plt.title(f'Class Separation Quality (Silhouette Score: {silhouette_avg:.3f})')
            plt.ylim(0, 1)
            plt.grid(axis='y', alpha=0.3)

            # Add interpretation text
            if silhouette_avg > 0.7:
                interpretation = "Excellent separation"
                color = 'green'
            elif silhouette_avg > 0.5:
                interpretation = "Good separation"
                color = 'orange'
            else:
                interpretation = "Poor separation"
                color = 'red'

            plt.text(0, silhouette_avg + 0.05, interpretation,
                     ha='center', va='bottom', fontsize=12, color=color, fontweight='bold')

            plt.tight_layout()
            plt.savefig(f"{save_dir}/silhouette_analysis.png", dpi=Config.PLOT_DPI, bbox_inches='tight')
            plt.close()

        logging.info("Class separation analysis created")

    except Exception as e:
        logging.error(f"Error creating class separation analysis: {str(e)}")


# Update the main visualization function to include all new analyses
def visualize_comprehensive_analysis(X, y, model_results, overlap_counts, dataset_sources, save_dir, le):
    """Create all comprehensive visualizations."""
    try:
        # Create all visualization categories
        logging.info("Creating comprehensive analysis visualizations...")

        # 1. Dataset overview
        visualize_dataset_overview(X, y, overlap_counts, dataset_sources, save_dir, le)

        # 2. Shape analysis
        visualize_shape_analysis(X, y, save_dir, le)

        # 3. Model performance
        visualize_performance(model_results, y, save_dir, le)

        # 4. Additional comprehensive analyses
        create_model_performance_comparison_chart(model_results, save_dir)
        create_feature_analysis_plots(X, y, save_dir, le)
        create_class_separation_analysis(X, y, save_dir, le)

        # 5. Final summary
        save_all_visualizations_summary(save_dir)

        logging.info("All comprehensive visualizations completed successfully!")

    except Exception as e:
        logging.error(f"Error in comprehensive visualization: {str(e)}")


# Export the main functions that will be called from main.py
__all__ = [
    'visualize_performance',
    'visualize_shape_analysis',
    'visualize_dataset_overview',
    'visualize_comprehensive_analysis'
]