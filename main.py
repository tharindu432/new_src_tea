import os
import sys
import numpy as np
import pandas as pd
import logging
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Import custom modules
from preprocess import preprocess_dataset
from feature_extraction import extract_features
from train_model import train_and_fine_tune_models
from evaluate import evaluate_model
from visualize import visualize_performance
from config import Config
from utils import ensure_dir


def main():
    """
    Orchestrate the entire pipeline with visualizations and robust error handling.
    """
    try:
        print("Starting Tea Quality Assessment Pipeline...")

        # Create output directories first
        directories_to_create = [
            Config.OUTPUT_DIR,
            Config.MODELS_DIR,
            Config.RESULTS_DIR,
            Config.LOGS_DIR,
            Config.VISUALIZATIONS_DIR,
            Config.PREPROCESSED_DIR,
            f"{Config.PREPROCESSED_DIR}/train",
            f"{Config.PREPROCESSED_DIR}/test",
            Config.PREPROCESSED_NEW_DIR,
            f"{Config.PREPROCESSED_NEW_DIR}/train",
            f"{Config.PREPROCESSED_NEW_DIR}/test"
        ]

        for directory in directories_to_create:
            ensure_dir(directory)

        # Setup logging after directories are created
        Config.setup_logging()
        logging.info("Starting Tea Quality Assessment Pipeline")

        # Validate dataset directories
        available_dirs = []
        dataset_dirs = [
            (Config.TRAIN_DIR, "dataset/train"),
            (Config.TEST_DIR, "dataset/test"),
            (Config.TRAIN_NEW_DIR, "dataset_New/train"),
            (Config.TEST_NEW_DIR, "dataset_New/test")
        ]

        for directory, name in dataset_dirs:
            if os.path.exists(directory) and os.listdir(directory):
                available_dirs.append((directory, name))
                logging.info(f"Found dataset directory: {name}")
            else:
                logging.warning(f"Dataset directory not found or empty: {name}")

        if not available_dirs:
            raise ValueError("No valid dataset directories found. Please check your dataset structure.")

        # Validate labels file
        if not os.path.exists(Config.LABELS_FILE):
            raise FileNotFoundError(f"Labels file not found: {Config.LABELS_FILE}")

        if os.stat(Config.LABELS_FILE).st_size == 0:
            raise ValueError(f"Labels file is empty: {Config.LABELS_FILE}")

        labels_df = pd.read_csv(Config.LABELS_FILE)
        logging.info(f"Loaded labels.csv with {len(labels_df)} entries")
        logging.info(f"Tea variants: {sorted(labels_df['tea_variant'].unique())}")
        logging.info(f"Elevations: {sorted(labels_df['elevation'].unique())}")

        # Preprocessing phase
        logging.info("=" * 50)
        logging.info("PREPROCESSING PHASE")
        logging.info("=" * 50)

        preprocessing_tasks = [
            (Config.TRAIN_DIR, f"{Config.PREPROCESSED_DIR}/train"),
            (Config.TEST_DIR, f"{Config.PREPROCESSED_DIR}/test"),
            (Config.TRAIN_NEW_DIR, f"{Config.PREPROCESSED_NEW_DIR}/train"),
            (Config.TEST_NEW_DIR, f"{Config.PREPROCESSED_NEW_DIR}/test")
        ]

        for input_dir, output_dir in preprocessing_tasks:
            if os.path.exists(input_dir):
                logging.info(f"Preprocessing {input_dir}...")
                preprocess_dataset(input_dir, output_dir)
            else:
                logging.info(f"Skipping preprocessing for {input_dir} (not found)")

        # Feature Extraction phase
        logging.info("=" * 50)
        logging.info("FEATURE EXTRACTION PHASE")
        logging.info("=" * 50)

        all_features = []
        all_labels = []
        all_overlap_counts = []

        extraction_tasks = [
            (f"{Config.PREPROCESSED_DIR}/train", "dataset/train"),
            (f"{Config.PREPROCESSED_DIR}/test", "dataset/test"),
            (f"{Config.PREPROCESSED_NEW_DIR}/train", "dataset_New/train"),
            (f"{Config.PREPROCESSED_NEW_DIR}/test", "dataset_New/test")
        ]

        for preprocessed_dir, name in extraction_tasks:
            if os.path.exists(preprocessed_dir):
                logging.info(f"Extracting features from {name}...")
                try:
                    X, y, overlap_counts = extract_features(
                        preprocessed_dir, Config.LABELS_FILE, Config.VISUALIZATIONS_DIR
                    )
                    if len(X) > 0:
                        all_features.append(X)
                        all_labels.extend(y)
                        all_overlap_counts.extend(overlap_counts)
                        logging.info(f"Extracted {len(X)} samples from {name}")
                    else:
                        logging.warning(f"No features extracted from {name}")
                except Exception as e:
                    logging.error(f"Failed to extract features from {name}: {str(e)}")
                    continue
            else:
                logging.info(f"Skipping feature extraction for {name} (preprocessed directory not found)")

        # Combine all features
        if not all_features:
            raise ValueError("No features were extracted from any dataset")

        # Combine feature arrays
        X_combined = np.vstack(all_features)
        y_combined = all_labels

        logging.info(f"Combined dataset: {len(X_combined)} samples with {X_combined.shape[1]} features")
        logging.info(f"Class distribution: {pd.Series(y_combined).value_counts().to_dict()}")

        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_combined,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE,
            stratify=y_combined
        )

        logging.info(f"Training set: {len(X_train)} samples")
        logging.info(f"Test set: {len(X_test)} samples")

        # Label Encoding
        logging.info("Encoding labels...")
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)

        # Save label encoder
        joblib.dump(le, f"{Config.MODELS_DIR}/label_encoder.pkl")
        logging.info(f"Label encoder saved. Classes: {le.classes_}")

        # Model Training phase
        logging.info("=" * 50)
        logging.info("MODEL TRAINING PHASE")
        logging.info("=" * 50)

        best_models, ensemble = train_and_fine_tune_models(X_train, y_train_encoded)

        if not best_models:
            raise ValueError("No models were successfully trained")

        # Model Evaluation phase
        logging.info("=" * 50)
        logging.info("MODEL EVALUATION PHASE")
        logging.info("=" * 50)

        accuracies = []
        reports = {}
        y_pred_dict = {}
        model_names = list(best_models.keys()) + ['Ensemble']

        for model_name in model_names:
            try:
                model_path = f"{Config.MODELS_DIR}/{model_name}.pkl"
                if not os.path.exists(model_path):
                    logging.warning(f"Model file {model_path} not found, skipping")
                    continue

                model = joblib.load(model_path)
                accuracy, report, y_pred = evaluate_model(
                    model, X_test, y_test_encoded, model_name, Config.VISUALIZATIONS_DIR
                )

                accuracies.append(accuracy)
                reports[model_name] = report
                y_pred_dict[model_name] = y_pred

            except Exception as e:
                logging.error(f"Error evaluating {model_name}: {str(e)}")
                continue

        if not accuracies:
            raise ValueError("No models were successfully evaluated")

        # Visualization phase
        logging.info("=" * 50)
        logging.info("VISUALIZATION PHASE")
        logging.info("=" * 50)

        try:
            evaluated_models = [name for name in model_names if name in y_pred_dict]
            evaluated_accuracies = [acc for name, acc in zip(model_names, accuracies) if name in y_pred_dict]

            visualize_performance(
                evaluated_models, evaluated_accuracies, reports,
                y_test_encoded, y_pred_dict, Config.VISUALIZATIONS_DIR
            )
        except Exception as e:
            logging.error(f"Error in visualization: {str(e)}")

        # Save results
        logging.info("=" * 50)
        logging.info("SAVING RESULTS")
        logging.info("=" * 50)

        # Save overlap counts
        try:
            overlap_df = pd.DataFrame({
                'sample_id': [f"sample_{i}" for i in range(len(all_overlap_counts))],
                'overlap_count': all_overlap_counts
            })
            overlap_path = f"{Config.RESULTS_DIR}/overlap_counts.csv"
            overlap_df.to_csv(overlap_path, index=False)
            logging.info(f"Saved overlap counts to {overlap_path}")
        except Exception as e:
            logging.error(f"Error saving overlap counts: {str(e)}")

        # Save model performance summary
        try:
            performance_df = pd.DataFrame({
                'Model': evaluated_models,
                'Accuracy': evaluated_accuracies
            })
            performance_path = f"{Config.RESULTS_DIR}/model_performance.csv"
            performance_df.to_csv(performance_path, index=False)
            logging.info(f"Saved model performance to {performance_path}")
        except Exception as e:
            logging.error(f"Error saving performance summary: {str(e)}")

        # Final summary
        logging.info("=" * 50)
        logging.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logging.info("=" * 50)

        best_model_idx = np.argmax(evaluated_accuracies)
        best_model_name = evaluated_models[best_model_idx]
        best_accuracy = evaluated_accuracies[best_model_idx]

        logging.info(f"Best performing model: {best_model_name} with accuracy: {best_accuracy:.4f}")
        logging.info(f"Total samples processed: {len(X_combined)}")
        logging.info(f"Total features per sample: {X_combined.shape[1]}")
        logging.info(f"Number of classes: {len(le.classes_)}")

        print("\n" + "=" * 50)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"Best model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        print(f"Results saved in: {Config.OUTPUT_DIR}")
        print(f"Check logs at: {Config.LOGS_DIR}/pipeline.log")

    except Exception as e:
        error_msg = f"Error in main pipeline: {str(e)}"
        logging.error(error_msg)
        print(f"\nERROR: {error_msg}")
        print("Check logs for detailed error information.")
        sys.exit(1)


if __name__ == "__main__":
    main()