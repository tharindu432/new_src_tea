import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Import all modules
from config import Config
from preprocess import preprocess_dataset
from feature_extraction import AdvancedFeatureExtractor
from train_model import train_and_fine_tune_models
from evaluate import evaluate_model
from visualize import ComprehensiveVisualizer
from utils import ensure_dir
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class TeaParticleAnalysisPipeline:
    """Complete tea particle analysis pipeline with comprehensive outputs."""

    def __init__(self):
        self.logger = None
        self.setup_logging()
        self.feature_extractor = AdvancedFeatureExtractor()
        self.visualizer = ComprehensiveVisualizer()

    def setup_logging(self):
        """Initialize logging system."""
        try:
            # Create output directories first
            Config.create_output_directories()
            Config.setup_logging()
            self.logger = logging.getLogger(__name__)
            self.logger.info("=" * 80)
            self.logger.info("TEA PARTICLE ANALYSIS PIPELINE STARTED")
            self.logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info("=" * 80)
        except Exception as e:
            print(f"Error setting up logging: {str(e)}")
            sys.exit(1)

    def validate_datasets(self):
        """Validate dataset structure and labels."""
        try:
            self.logger.info("Validating dataset structure...")

            # Check dataset directories
            dataset_info = {}
            for name, directory in [
                ('train', Config.TRAIN_DIR),
                ('test', Config.TEST_DIR),
                ('train_new', Config.TRAIN_NEW_DIR),
                ('test_new', Config.TEST_NEW_DIR)
            ]:
                if os.path.exists(directory) and os.listdir(directory):
                    # Count files
                    file_count = 0
                    for root, _, files in os.walk(directory):
                        file_count += len([f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))])

                    dataset_info[name] = {
                        'exists': True,
                        'path': directory,
                        'file_count': file_count
                    }
                    self.logger.info(f"Dataset {name}: {file_count} images found")
                else:
                    dataset_info[name] = {
                        'exists': False,
                        'path': directory,
                        'file_count': 0
                    }
                    self.logger.warning(f"Dataset {name}: directory missing or empty")

            # Validate labels file
            if not os.path.exists(Config.LABELS_FILE):
                raise FileNotFoundError(f"Labels file not found: {Config.LABELS_FILE}")

            labels_df = pd.read_csv(Config.LABELS_FILE)
            if labels_df.empty:
                raise ValueError(f"Labels file is empty: {Config.LABELS_FILE}")

            self.logger.info(f"Labels file: {len(labels_df)} entries found")
            self.logger.info(f"Tea variants: {sorted(labels_df['tea_variant'].unique())}")
            self.logger.info(f"Elevations: {sorted(labels_df['elevation'].unique())}")

            # Check if we have any valid datasets
            total_files = sum(info['file_count'] for info in dataset_info.values())
            if total_files == 0:
                raise ValueError("No valid image files found in any dataset directory")

            return dataset_info, labels_df

        except Exception as e:
            self.logger.error(f"Dataset validation failed: {str(e)}")
            raise

    def preprocess_all_datasets(self, dataset_info):
        """Preprocess all available datasets."""
        try:
            self.logger.info("Starting dataset preprocessing...")

            preprocessing_results = {}

            for name, info in dataset_info.items():
                if not info['exists'] or info['file_count'] == 0:
                    self.logger.info(f"Skipping {name} - no data available")
                    continue

                try:
                    # Determine output directory
                    if 'new' in name.lower():
                        if 'train' in name.lower():
                            output_dir = os.path.join(Config.PREPROCESSED_NEW_DIR, 'train')
                        else:
                            output_dir = os.path.join(Config.PREPROCESSED_NEW_DIR, 'test')
                    else:
                        if 'train' in name.lower():
                            output_dir = os.path.join(Config.PREPROCESSED_DIR, 'train')
                        else:
                            output_dir = os.path.join(Config.PREPROCESSED_DIR, 'test')

                    ensure_dir(output_dir)

                    self.logger.info(f"Preprocessing {name} dataset...")
                    preprocess_dataset(info['path'], output_dir)

                    preprocessing_results[name] = {
                        'success': True,
                        'input_dir': info['path'],
                        'output_dir': output_dir
                    }

                    self.logger.info(f"Successfully preprocessed {name} dataset")

                except Exception as e:
                    self.logger.error(f"Error preprocessing {name} dataset: {str(e)}")
                    preprocessing_results[name] = {
                        'success': False,
                        'error': str(e)
                    }

            return preprocessing_results

        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            raise

    def extract_all_features(self, preprocessing_results):
        """Extract features from all preprocessed datasets."""
        try:
            self.logger.info("Starting feature extraction...")

            all_features = []
            all_labels = []
            all_overlap_counts = []
            feature_extraction_info = {}

            # Process each successfully preprocessed dataset
            for name, result in preprocessing_results.items():
                if not result['success']:
                    continue

                try:
                    self.logger.info(f"Extracting features from {name} dataset...")

                    features, labels, overlap_counts = self.feature_extractor.extract_features(
                        result['output_dir'],
                        Config.LABELS_FILE,
                        os.path.join(Config.VISUALIZATIONS_DIR, f"{name}_analysis")
                    )

                    if len(features) > 0:
                        all_features.append(features)
                        all_labels.extend(labels)
                        all_overlap_counts.extend(overlap_counts)

                        feature_extraction_info[name] = {
                            'samples': len(features),
                            'features_per_sample': features.shape[1],
                            'unique_labels': len(set(labels))
                        }

                        self.logger.info(f"Extracted {len(features)} samples from {name} "
                                         f"with {features.shape[1]} features each")
                    else:
                        self.logger.warning(f"No features extracted from {name}")

                except Exception as e:
                    self.logger.error(f"Error extracting features from {name}: {str(e)}")
                    continue

            if not all_features:
                raise ValueError("No features extracted from any dataset")

            # Combine all features
            combined_features = np.vstack(all_features)

            self.logger.info(f"Total features extracted: {len(combined_features)} samples, "
                             f"{combined_features.shape[1]} features per sample")
            self.logger.info(f"Unique labels: {len(set(all_labels))}")

            return combined_features, all_labels, all_overlap_counts, feature_extraction_info

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            raise

    def prepare_training_data(self, features, labels, overlap_counts):
        """Prepare and split data for training."""
        try:
            self.logger.info("Preparing training data...")

            # Encode labels
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)

            # Save label encoder
            joblib.dump(label_encoder, os.path.join(Config.MODELS_DIR, 'label_encoder.pkl'))

            self.logger.info(f"Label encoding completed. Classes: {list(label_encoder.classes_)}")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, encoded_labels,
                test_size=Config.TEST_SIZE,
                random_state=Config.RANDOM_STATE,
                stratify=encoded_labels
            )

            self.logger.info(f"Data split completed:")
            self.logger.info(f"  Training samples: {len(X_train)}")
            self.logger.info(f"  Testing samples: {len(X_test)}")

            # Save training data info
            data_info = {
                'total_samples': len(features),
                'training_samples': len(X_train),
                'testing_samples': len(X_test),
                'features_per_sample': features.shape[1],
                'unique_classes': len(label_encoder.classes_),
                'class_names': list(label_encoder.classes_),
                'feature_names': [f'feature_{i}' for i in range(features.shape[1])]
            }

            # Save data info
            pd.DataFrame([data_info]).to_csv(
                os.path.join(Config.RESULTS_DIR, 'training_data_info.csv'),
                index=False
            )

            return X_train, X_test, y_train, y_test, label_encoder, data_info

        except Exception as e:
            self.logger.error(f"Data preparation failed: {str(e)}")
            raise

    def train_models(self, X_train, y_train):
        """Train all models."""
        try:
            self.logger.info("Starting model training...")

            best_models, ensemble = train_and_fine_tune_models(X_train, y_train)

            # Get list of all trained models
            model_names = list(best_models.keys()) + ['Ensemble']

            self.logger.info(f"Training completed. Models trained: {model_names}")

            return best_models, ensemble, model_names

        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            raise

    def evaluate_models(self, model_names, X_test, y_test, label_encoder):
        """Evaluate all trained models."""
        try:
            self.logger.info("Starting model evaluation...")

            evaluation_results = {}
            accuracies = []
            reports = {}
            y_pred_dict = {}

            for model_name in model_names:
                try:
                    model_path = os.path.join(Config.MODELS_DIR, f'{model_name}.pkl')

                    if not os.path.exists(model_path):
                        self.logger.warning(f"Model file not found: {model_path}")
                        continue

                    # Load model
                    model = joblib.load(model_path)

                    # Evaluate model
                    accuracy, report, y_pred = evaluate_model(
                        model, X_test, y_test, model_name, Config.VISUALIZATIONS_DIR
                    )

                    accuracies.append(accuracy)
                    reports[model_name] = report
                    y_pred_dict[model_name] = y_pred

                    evaluation_results[model_name] = {
                        'accuracy': accuracy,
                        'report': report
                    }

                    self.logger.info(f"{model_name} evaluation completed - Accuracy: {accuracy:.4f}")

                except Exception as e:
                    self.logger.error(f"Error evaluating {model_name}: {str(e)}")
                    continue

            if not evaluation_results:
                raise ValueError("No models successfully evaluated")

            # Save evaluation results
            eval_summary = []
            for model_name, results in evaluation_results.items():
                eval_summary.append({
                    'model': model_name,
                    'accuracy': results['accuracy'],
                    'precision': results['report']['weighted avg']['precision'],
                    'recall': results['report']['weighted avg']['recall'],
                    'f1_score': results['report']['weighted avg']['f1-score']
                })

            eval_df = pd.DataFrame(eval_summary)
            eval_df.to_csv(os.path.join(Config.RESULTS_DIR, 'model_evaluation_summary.csv'), index=False)

            return evaluation_results, accuracies, reports, y_pred_dict

        except Exception as e:
            self.logger.error(f"Model evaluation failed: {str(e)}")
            raise

    def create_comprehensive_visualizations(self, model_names, accuracies, reports,
                                            y_test, y_pred_dict, overlap_counts,
                                            X_train, X_test, data_info):
        """Create all visualization outputs."""
        try:
            self.logger.info("Creating comprehensive visualizations...")

            # 1. Model performance visualizations
            self.visualizer.visualize_performance(
                model_names, accuracies, reports, y_test, y_pred_dict, Config.VISUALIZATIONS_DIR
            )

            # 2. Overlap analysis visualization
            overlap_data = [
                {'sample_id': f'sample_{i}', 'overlap_count': count, 'dataset': 'combined'}
                for i, count in enumerate(overlap_counts)
            ]
            self.visualizer.create_overlap_summary_visualization(
                overlap_data, Config.VISUALIZATIONS_DIR
            )

            # 3. Feature analysis visualization
            feature_names = data_info.get('feature_names', [])
            self.visualizer.create_feature_analysis_visualization(
                X_train, X_test, feature_names, Config.VISUALIZATIONS_DIR
            )

            self.logger.info("All visualizations created successfully")

        except Exception as e:
            self.logger.error(f"Visualization creation failed: {str(e)}")
            raise

    def save_comprehensive_results(self, overlap_counts, evaluation_results, data_info):
        """Save all pipeline results."""
        try:
            self.logger.info("Saving comprehensive results...")

            # 1. Save overlap analysis results
            overlap_df = pd.DataFrame({
                'sample_id': [f'sample_{i}' for i in range(len(overlap_counts))],
                'overlap_count': overlap_counts
            })
            overlap_path = os.path.join(Config.RESULTS_DIR, 'overlap_analysis_results.csv')
            overlap_df.to_csv(overlap_path, index=False)

            # 2. Create final pipeline report
            report_path = os.path.join(Config.RESULTS_DIR, 'pipeline_final_report.txt')

            with open(report_path, 'w') as f:
                f.write("TEA PARTICLE ANALYSIS PIPELINE - FINAL REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("DATASET SUMMARY:\n")
                f.write(f"  Total Samples Processed: {data_info['total_samples']}\n")
                f.write(f"  Training Samples: {data_info['training_samples']}\n")
                f.write(f"  Testing Samples: {data_info['testing_samples']}\n")
                f.write(f"  Features per Sample: {data_info['features_per_sample']}\n")
                f.write(f"  Unique Classes: {data_info['unique_classes']}\n")
                f.write(f"  Class Names: {', '.join(data_info['class_names'])}\n\n")

                f.write("OVERLAP ANALYSIS SUMMARY:\n")
                f.write(f"  Total Overlap Counts: {len(overlap_counts)}\n")
                f.write(f"  Average Overlap Count: {np.mean(overlap_counts):.2f}\n")
                f.write(f"  Overlap Count Range: {np.min(overlap_counts)} - {np.max(overlap_counts)}\n\n")

                f.write("MODEL PERFORMANCE SUMMARY:\n")
                best_model = max(evaluation_results.items(), key=lambda x: x[1]['accuracy'])
                f.write(f"  Best Performing Model: {best_model[0]}\n")
                f.write(f"  Best Accuracy: {best_model[1]['accuracy']:.4f}\n\n")

                f.write("DETAILED MODEL RESULTS:\n")
                for model_name, results in evaluation_results.items():
                    f.write(f"  {model_name}:\n")
                    f.write(f"    Accuracy: {results['accuracy']:.4f}\n")
                    f.write(f"    Precision: {results['report']['weighted avg']['precision']:.4f}\n")
                    f.write(f"    Recall: {results['report']['weighted avg']['recall']:.4f}\n")
                    f.write(f"    F1-Score: {results['report']['weighted avg']['f1-score']:.4f}\n\n")

                f.write("OUTPUT FILES GENERATED:\n")
                f.write("  Models: output/models/\n")
                f.write("  Results: output/results/\n")
                f.write("  Visualizations: output/visualizations/\n")
                f.write("  Segmented Images: output/segmented_images/\n")
                f.write("  Overlap Analysis: output/overlap_analysis/\n")
                f.write("  Logs: output/logs/\n")

            self.logger.info(f"Final pipeline report saved: {report_path}")
            self.logger.info(f"Overlap analysis results saved: {overlap_path}")

        except Exception as e:
            self.logger.error(f"Error saving final results: {str(e)}")
            raise

    def run_complete_pipeline(self):
        """Execute the complete tea particle analysis pipeline."""
        try:
            start_time = datetime.now()
            self.logger.info("Starting complete pipeline execution...")

            # Step 1: Validate datasets
            dataset_info, labels_df = self.validate_datasets()

            # Step 2: Preprocess datasets
            preprocessing_results = self.preprocess_all_datasets(dataset_info)

            # Step 3: Extract features
            features, labels, overlap_counts, feature_info = self.extract_all_features(preprocessing_results)

            # Step 4: Prepare training data
            X_train, X_test, y_train, y_test, label_encoder, data_info = self.prepare_training_data(
                features, labels, overlap_counts
            )

            # Step 5: Train models
            best_models, ensemble, model_names = self.train_models(X_train, y_train)

            # Step 6: Evaluate models
            evaluation_results, accuracies, reports, y_pred_dict = self.evaluate_models(
                model_names, X_test, y_test, label_encoder
            )

            # Step 7: Create comprehensive visualizations
            self.create_comprehensive_visualizations(
                model_names, accuracies, reports, y_test, y_pred_dict,
                overlap_counts, X_train, X_test, data_info
            )

            # Step 8: Save comprehensive results
            self.save_comprehensive_results(overlap_counts, evaluation_results, data_info)

            # Pipeline completion summary
            end_time = datetime.now()
            duration = end_time - start_time

            self.logger.info("=" * 80)
            self.logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
            self.logger.info(f"Total execution time: {duration}")
            self.logger.info(f"Samples processed: {len(features)}")
            self.logger.info(f"Models trained: {len(model_names)}")
            self.logger.info(f"Best model accuracy: {max(accuracies):.4f}")
            self.logger.info("=" * 80)

            # Print output directory structure
            self.print_output_summary()

            return {
                'success': True,
                'duration': duration,
                'samples_processed': len(features),
                'models_trained': len(model_names),
                'best_accuracy': max(accuracies),
                'output_directories': {
                    'models': Config.MODELS_DIR,
                    'results': Config.RESULTS_DIR,
                    'visualizations': Config.VISUALIZATIONS_DIR,
                    'segmentation': Config.SEGMENTATION_DIR,
                    'overlap_analysis': Config.OVERLAP_DIR,
                    'logs': Config.LOGS_DIR
                }
            }

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            self.logger.error("Pipeline terminated with errors")
            raise

    def print_output_summary(self):
        """Print summary of generated output files."""
        try:
            self.logger.info("\nOUTPUT FILES GENERATED:")
            self.logger.info("-" * 40)

            output_dirs = [
                (Config.MODELS_DIR, "Trained Models"),
                (Config.RESULTS_DIR, "Analysis Results"),
                (Config.VISUALIZATIONS_DIR, "Visualizations"),
                (Config.SEGMENTATION_DIR, "Segmented Images"),
                (Config.OVERLAP_DIR, "Overlap Analysis"),
                (Config.LOGS_DIR, "Log Files")
            ]

            for directory, description in output_dirs:
                if os.path.exists(directory):
                    file_count = sum(len(files) for _, _, files in os.walk(directory))
                    self.logger.info(f"{description:.<30} {file_count:>3} files")
                    self.logger.info(f"  Location: {directory}")
                else:
                    self.logger.info(f"{description:.<30} No files (directory not created)")

            self.logger.info("-" * 40)

        except Exception as e:
            self.logger.error(f"Error printing output summary: {str(e)}")


def main():
    """Main function to run the complete pipeline."""
    try:
        # Initialize and run pipeline
        pipeline = TeaParticleAnalysisPipeline()
        results = pipeline.run_complete_pipeline()

        print("\n" + "=" * 80)
        print("TEA PARTICLE ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Execution time: {results['duration']}")
        print(f"Samples processed: {results['samples_processed']}")
        print(f"Models trained: {results['models_trained']}")
        print(f"Best accuracy achieved: {results['best_accuracy']:.4f}")
        print("\nOutput directories:")
        for name, path in results['output_directories'].items():
            print(f"  {name.title()}: {path}")
        print("=" * 80)

        return 0

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        return 1

    except Exception as e:
        print(f"\nPipeline failed with error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)