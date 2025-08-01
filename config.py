import logging
import os
from pathlib import Path


class Config:
    # Define project root (C:\L4S1\PythonProject2)
    PROJECT_ROOT = str(Path(__file__).parent.parent)

    # Dataset paths
    DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
    DATASET_NEW_DIR = os.path.join(PROJECT_ROOT, "dataset_New")
    TRAIN_DIR = os.path.join(DATASET_DIR, "train")
    TEST_DIR = os.path.join(DATASET_DIR, "test")
    TRAIN_NEW_DIR = os.path.join(DATASET_NEW_DIR, "train")
    TEST_NEW_DIR = os.path.join(DATASET_NEW_DIR, "test")
    PREPROCESSED_DIR = os.path.join(PROJECT_ROOT, "dataset/preprocessed")
    PREPROCESSED_NEW_DIR = os.path.join(PROJECT_ROOT, "dataset_New/preprocessed")
    LABELS_FILE = os.path.join(PROJECT_ROOT, "dataset/labels.csv")  # Single labels file

    # Output paths
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
    MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
    RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
    LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
    VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

    # Preprocessing parameters
    MIN_CONTOUR_AREA = 100  # Minimum contour area for valid particles
    MORPH_KERNEL_SIZE = (3, 3)  # Kernel size for morphological operations
    CONTRAST_ALPHA = 1.5  # Contrast stretching factor
    CONTRAST_BETA = 0  # Contrast offset

    # Overlap handling parameters
    MAX_CLUSTERS = 5  # Maximum number of clusters for K-means
    AVG_PARTICLE_AREA = 500  # Estimated average particle area
    HISTOGRAM_BINS = 256  # Number of bins for histogram peak detection
    MIN_PEAK_HEIGHT = 100  # Minimum height for histogram peaks
    MIN_CLUSTER_SIZE = 10  # Minimum points per cluster
    MATCH_SHAPE_THRESHOLD = 0.1  # Threshold for contour matching
    OVERLAP_VALIDATION_THRESHOLD = 0.05  # For cross-image validation

    # Model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5  # Cross-validation folds

    # Visualization parameters
    VISUALIZE_SAMPLES = 5  # Number of samples to visualize
    PLOT_DPI = 300  # DPI for saved plots

    @staticmethod
    def setup_logging():
        """
        Configure logging for the system.
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{Config.LOGS_DIR}/pipeline.log"),
                logging.StreamHandler()
            ]
        )