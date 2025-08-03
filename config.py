import logging
import os
from pathlib import Path


class Config:
    # Define project root
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
    LABELS_FILE = os.path.join(PROJECT_ROOT, "dataset/labels.csv")

    # Output paths
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
    MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
    RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
    LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
    VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
    SEGMENTATION_DIR = os.path.join(OUTPUT_DIR, "segmented_images")
    OVERLAP_DIR = os.path.join(OUTPUT_DIR, "overlap_analysis")

    # Segmentation parameters
    MIN_CONTOUR_AREA = 50  # Minimum contour area for valid particles
    MAX_CONTOUR_AREA = 10000  # Maximum contour area to filter noise
    MORPH_KERNEL_SIZE = (3, 3)  # Kernel size for morphological operations

    # Morphological operation parameters
    DILATE_ITERATIONS = 2  # Number of dilation iterations
    ERODE_ITERATIONS = 2  # Number of erosion iterations
    CLOSE_ITERATIONS = 2  # Number of closing iterations
    OPEN_ITERATIONS = 1  # Number of opening iterations

    # Thresholding parameters
    BINARY_THRESHOLD = 127  # Binary threshold value
    ADAPTIVE_BLOCK_SIZE = 15  # Block size for adaptive thresholding
    ADAPTIVE_THRESH_BLOCK_SIZE = 15  # Alternative name for adaptive thresholding block size
    ADAPTIVE_C = 8  # Constant subtracted from mean in adaptive thresholding
    ADAPTIVE_THRESH_C = 8  # Alternative name for adaptive thresholding constant

    # Contrast enhancement parameters
    CONTRAST_ALPHA = 1.2  # Contrast stretching factor
    CONTRAST_BETA = 10  # Contrast offset
    GAMMA_CORRECTION = 1.2  # Gamma correction value

    # Overlap handling parameters
    MAX_CLUSTERS = 8  # Maximum number of clusters for K-means
    AVG_PARTICLE_AREA = 800  # Estimated average particle area
    HISTOGRAM_BINS = 256  # Number of bins for histogram peak analysis
    MIN_PEAK_HEIGHT = 50  # Minimum height for histogram peaks
    MIN_PEAK_DISTANCE = 20  # Minimum distance between peaks
    MIN_CLUSTER_SIZE = 5  # Minimum points per cluster
    MATCH_SHAPE_THRESHOLD = 0.3  # Threshold for contour matching
    OVERLAP_VALIDATION_THRESHOLD = 0.1  # For cross-image validation

    # Shape analysis parameters
    N_FOURIER_DESCRIPTORS = 10  # Number of Fourier descriptors to extract
    MIN_CONTOUR_POINTS = 10  # Minimum number of points in contour for Fourier analysis
    HU_MOMENTS_COUNT = 7  # Number of Hu moments to extract

    # Noise filtering parameters
    GAUSSIAN_BLUR_KERNEL = (5, 5)  # Gaussian blur kernel size (FIXED)
    GAUSSIAN_KERNEL = (5, 5)  # Alternative name for compatibility
    MEDIAN_FILTER_SIZE = 3  # Median filter kernel size
    BILATERAL_D = 9  # Bilateral filter diameter
    BILATERAL_SIGMA_COLOR = 75  # Bilateral filter sigma color
    BILATERAL_SIGMA_SPACE = 75  # Bilateral filter sigma space

    # Model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5

    # Visualization parameters
    VISUALIZE_SAMPLES = 10  # Number of samples to visualize
    PLOT_DPI = 300  # DPI for saved plots
    FIGURE_SIZE = (12, 8)  # Default figure size

    # Color schemes for visualizations
    COLORS = {
        'particles': (0, 255, 0),  # Green for particles
        'overlaps': (0, 0, 255),  # Red for overlaps
        'background': (255, 0, 0),  # Blue for background
        'contours': (255, 255, 0)  # Cyan for contours
    }

    @staticmethod
    def setup_logging():
        """Configure logging for the system."""
        os.makedirs(Config.LOGS_DIR, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{Config.LOGS_DIR}/pipeline.log"),
                logging.StreamHandler()
            ]
        )

    @staticmethod
    def create_output_directories():
        """Create all necessary output directories."""
        directories = [
            Config.OUTPUT_DIR,
            Config.MODELS_DIR,
            Config.RESULTS_DIR,
            Config.LOGS_DIR,
            Config.VISUALIZATIONS_DIR,
            Config.SEGMENTATION_DIR,
            Config.OVERLAP_DIR,
            Config.PREPROCESSED_DIR,
            Config.PREPROCESSED_NEW_DIR
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logging.info(f"Created directory: {directory}")