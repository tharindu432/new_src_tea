import os
import pandas as pd
import logging
from config import Config
from utils import get_sample_groups


def setup_logging():
    """
    Configure logging for verify_dataset.py.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("verify_dataset.log"),
            logging.StreamHandler()
        ]
    )


def verify_dataset():
    """
    Verify dataset structure and labels.csv consistency.
    """
    setup_logging()
    logging.info("Starting dataset verification")

    # Check directories
    for directory in [Config.TRAIN_DIR, Config.TEST_DIR, Config.TRAIN_NEW_DIR, Config.TEST_NEW_DIR]:
        if not os.path.exists(directory):
            logging.warning(f"Directory does not exist: {directory}")
            continue
        if not os.listdir(directory):
            logging.warning(f"Directory is empty: {directory}")
            continue
        logging.info(f"Checking {directory}")
        sample_groups = get_sample_groups(directory)
        logging.info(f"Found {len(sample_groups)} samples in {directory}")
        for sample_id, image_paths in sample_groups:
            logging.info(f"Sample {sample_id}: {len(image_paths)} images")

    # Check labels.csv
    if not os.path.exists(Config.LABELS_FILE):
        logging.error(f"Labels file not found: {Config.LABELS_FILE}")
        return
    if os.stat(Config.LABELS_FILE).st_size == 0:
        logging.error(f"Labels file is empty: {Config.LABELS_FILE}")
        return
    labels_df = pd.read_csv(Config.LABELS_FILE)
    logging.info(f"Labels file contains {len(labels_df)} entries")

    # Verify label consistency
    tea_variants = labels_df['tea_variant'].unique()
    elevations = labels_df['elevation'].unique()
    logging.info(f"Tea variants found: {tea_variants}")
    logging.info(f"Elevations found: {elevations}")

    # Check for missing images in labels.csv
    for directory in [Config.TRAIN_DIR, Config.TRAIN_NEW_DIR]:
        for root, _, files in os.walk(directory):
            if "PARTICLE" not in root:
                continue
            for f in files:
                if f.endswith(('.png', '.jpg', '.jpeg')):
                    relative_path = os.path.join(root, f).replace(directory + os.sep, '').replace('\\', '/')
                    if relative_path not in labels_df['filename'].values:
                        logging.warning(f"Image {relative_path} not found in labels.csv")


if __name__ == "__main__":
    verify_dataset()