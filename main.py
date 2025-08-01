from preprocess import preprocess_dataset
from feature_extraction import extract_features
from train_model import train_and_fine_tune_models
from evaluate import evaluate_model
from visualize import visualize_performance
from config import Config
from utils import ensure_dir
from sklearn.preprocessing import LabelEncoder
import joblib
import logging
import numpy as np
import pandas as pd
import os


def main():
    """
    Orchestrate the entire pipeline with visualizations and robust error handling.
    """
    try:
        # Validate dataset directories
        for directory in [Config.TRAIN_DIR, Config.TEST_DIR, Config.TRAIN_NEW_DIR, Config.TEST_NEW_DIR]:
            if not os.path.exists(directory):
                logging.warning(f"Dataset directory not found, skipping: {directory}")
                continue
            if not os.listdir(directory):
                logging.warning(f"Dataset directory is empty, skipping: {directory}")
                continue

        # Validate single labels file
        if not os.path.exists(Config.LABELS_FILE):
            raise FileNotFoundError(f"Labels file not found: {Config.LABELS_FILE}")
        if os.stat(Config.LABELS_FILE).st_size == 0:
            raise ValueError(f"Labels file is empty: {Config.LABELS_FILE}")
        labels_df = pd.read_csv(Config.LABELS_FILE)
        logging.info(f"Loaded labels.csv with {len(labels_df)} entries")
        logging.info(f"Tea variants: {labels_df['tea_variant'].unique()}")
        logging.info(f"Elevations: {labels_df['elevation'].unique()}")

        # Create output directories
        ensure_dir(Config.PREPROCESSED_DIR)
        ensure_dir(f"{Config.PREPROCESSED_DIR}/train")
        ensure_dir(f"{Config.PREPROCESSED_DIR}/test")
        ensure_dir(Config.PREPROCESSED_NEW_DIR)
        ensure_dir(f"{Config.PREPROCESSED_NEW_DIR}/train")
        ensure_dir(f"{Config.PREPROCESSED_NEW_DIR}/test")
        ensure_dir(Config.MODELS_DIR)
        ensure_dir(Config.RESULTS_DIR)
        ensure_dir(Config.LOGS_DIR)
        ensure_dir(Config.VISUALIZATIONS_DIR)

        # Setup logging after directories are created
        Config.setup_logging()
        logging.info("Starting pipeline")

        # Preprocessing
        logging.info("Preprocessing datasets...")
        preprocess_dataset(Config.TRAIN_DIR, f"{Config.PREPROCESSED_DIR}/train")
        preprocess_dataset(Config.TEST_DIR, f"{Config.PREPROCESSED_DIR}/test")
        preprocess_dataset(Config.TRAIN_NEW_DIR, f"{Config.PREPROCESSED_NEW_DIR}/train")
        preprocess_dataset(Config.TEST_NEW_DIR, f"{Config.PREPROCESSED_NEW_DIR}/test")

        # Feature Extraction
        logging.info("Extracting features...")
        X_train, y_train, train_overlap_counts = [], [], []
        X_test, y_test, test_overlap_counts = [], [], []
        X_train_new, y_train_new, train_new_overlap_counts = [], [], []
        X_test_new, y_test_new, test_new_overlap_counts = [], [], []

        # Extract features from dataset
        try:
            X_train, y_train, train_overlap_counts = extract_features(
                f"{Config.PREPROCESSED_DIR}/train", Config.LABELS_FILE, Config.VISUALIZATIONS_DIR
            )
            logging.info(f"Extracted {len(X_train)} training samples from dataset/train")
        except Exception as e:
            logging.warning(f"No features extracted from dataset/train: {str(e)}")
        try:
            X_test, y_test, test_overlap_counts = extract_features(
                f"{Config.PREPROCESSED_DIR}/test", Config.LABELS_FILE, Config.VISUALIZATIONS_DIR
            )
            logging.info(f"Extracted {len(X_test)} test samples from dataset/test")
        except Exception as e:
            logging.warning(f"No features extracted from dataset/test: {str(e)}")

        # Extract features from dataset_New
        try:
            X_train_new, y_train_new, train_new_overlap_counts = extract_features(
                f"{Config.PREPROCESSED_NEW_DIR}/train", Config.LABELS_FILE, Config.VISUALIZATIONS_DIR
            )
            logging.info(f"Extracted {len(X_train_new)} training samples from dataset_New/train")
        except Exception as e:
            logging.warning(f"No features extracted from dataset_New/train: {str(e)}")
        try:
            X_test_new, y_test_new, test_new_overlap_counts = extract_features(
                f"{Config.PREPROCESSED_NEW_DIR}/test", Config.LABELS_FILE, Config.VISUALIZATIONS_DIR
            )
            logging.info(f"Extracted {len(X_test_new)} test samples from dataset_New/test")
        except Exception as e:
            logging.warning(f"No features extracted from dataset_New/test: {str(e)}")

        # Combine features and labels
        X_train_combined = []
        y_train_combined = []
        X_test_combined = []
        y_test_combined = []
        train_overlap_counts_combined = []
        test_overlap_counts_combined = []

        if X_train and X_train_new:
            if X_train.shape[1] != X_train_new.shape[1]:
                raise ValueError("Feature dimensions mismatch between dataset and dataset_New for training")
            X_train_combined = np.vstack([X_train, X_train_new])
            y_train_combined = y_train + y_train_new
            train_overlap_counts_combined = train_overlap_counts + train_new_overlap_counts
        elif X_train:
            X_train_combined = X_train
            y_train_combined = y_train
            train_overlap_counts_combined = train_overlap_counts
        elif X_train_new:
            X_train_combined = X_train_new
            y_train_combined = y_train_new
            train_overlap_counts_combined = train_new_overlap_counts
        else:
            raise ValueError("No training features extracted from either dataset")

        if X_test and X_test_new:
            if X_test.shape[1] != X_test_new.shape[1]:
                raise ValueError("Feature dimensions mismatch between dataset and dataset_New for testing")
            X_test_combined = np.vstack([X_test, X_test_new])
            y_test_combined = y_test + y_test_new
            test_overlap_counts_combined = test_overlap_counts + test_new_overlap_counts
        elif X_test:
            X_test_combined = X_test
            y_test_combined = y_test
            test_overlap_counts_combined = test_overlap_counts
        elif X_test_new:
            X_test_combined = X_test_new
            y_test_combined = y_test_new
            test_overlap_counts_combined = test_new_overlap_counts
        else:
            raise ValueError("No test features extracted from either dataset")

        logging.info(f"Combined {len(X_train_combined)} training samples and {len(X_test_combined)} test samples")

        # Label Encoding
        le = LabelEncoder()
        y_train_combined = le.fit_transform(y_train_combined)
        y_test_combined = le.transform(y_test_combined)
        logging.info("Encoded labels")
        joblib.dump(le, f"{Config.MODELS_DIR}/label_encoder.pkl")

        # Model Training
        logging.info("Training models...")
        best_models, ensemble = train_and_fine_tune_models(X_train_combined, y_train_combined)

        # Evaluation
        logging.info("Evaluating models...")
        accuracies = []
        reports = {}
        y_pred_dict = {}
        model_names = list(best_models.keys()) + ['Ensemble']

        for model_name in model_names:
            model_path = f"{Config.MODELS_DIR}/{model_name}.pkl"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file {model_path} not found")
            model = joblib.load(model_path)
            accuracy, report, y_pred = evaluate_model(model, X_test_combined, y_test_combined, model_name,
                                                      Config.VISUALIZATIONS_DIR)
            accuracies.append(accuracy)
            reports[model_name] = report
            y_pred_dict[model_name] = y_pred

        # Visualize Performance
        logging.info("Visualizing performance...")
        visualize_performance(model_names, accuracies, reports, y_test_combined, y_pred_dict, Config.VISUALIZATIONS_DIR)

        # Save Overlap Counts
        overlap_df = pd.DataFrame({
            'sample_id': [f"train_{i}" for i in range(len(train_overlap_counts_combined))] + [f"test_{i}" for i in
                                                                                              range(
                                                                                                  len(test_overlap_counts_combined))],
            'overlap_count': train_overlap_counts_combined + test_overlap_counts_combined,
            'dataset': ['train'] * len(train_overlap_counts_combined) + ['test'] * len(test_overlap_counts_combined)
        })
        overlap_path = f"{Config.RESULTS_DIR}/overlap_counts.csv"
        overlap_df.to_csv(overlap_path, index=False)
        logging.info(f"Saved overlap counts to {overlap_path}")

        logging.info("Pipeline completed successfully")
    except Exception as e:
        logging.error(f"Error in main pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()