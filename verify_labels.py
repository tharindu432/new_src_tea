import os
import pandas as pd
import logging
from config import Config


def setup_logging():
    """
    Configure logging for verify_labels.py.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("verify_labels.log"),
            logging.StreamHandler()
        ]
    )


def verify_labels():
    """
    Verify the single labels.csv and check for duplicates.
    """
    setup_logging()
    logging.info("Starting labels.csv verification")

    # Check for single labels.csv
    if not os.path.exists(Config.LABELS_FILE):
        logging.error(f"Labels file not found: {Config.LABELS_FILE}")
        return
    if os.stat(Config.LABELS_FILE).st_size == 0:
        logging.error(f"Labels file is empty: {Config.LABELS_FILE}")
        return
    labels_df = pd.read_csv(Config.LABELS_FILE)
    logging.info(f"Labels file at {Config.LABELS_FILE} contains {len(labels_df)} entries")
    logging.info(f"Tea variants: {labels_df['tea_variant'].unique()}")
    logging.info(f"Elevations: {labels_df['elevation'].unique()}")

    # Check for duplicate labels.csv files
    potential_duplicates = []
    for root, _, files in os.walk(Config.PROJECT_ROOT):
        for f in files:
            if f == "labels.csv" and os.path.join(root, f) != Config.LABELS_FILE:
                potential_duplicates.append(os.path.join(root, f))
    if potential_duplicates:
        logging.warning(f"Found potential duplicate labels.csv files: {potential_duplicates}")
        for dup_file in potential_duplicates:
            dup_df = pd.read_csv(dup_file)
            logging.info(f"Duplicate file {dup_file} contains {len(dup_df)} entries")

    # Verify label consistency with datasets
    for directory in [Config.TRAIN_DIR, Config.TEST_DIR, Config.TRAIN_NEW_DIR, Config.TEST_NEW_DIR]:
        if not os.path.exists(directory):
            continue
        for root, _, files in os.walk(directory):
            if "PARTICLE" not in root:
                continue
            for f in files:
                if f.endswith(('.png', '.jpg', '.jpeg')):
                    relative_path = os.path.join(root, f).replace(directory + os.sep, '').replace('\\', '/')
                    if relative_path not in labels_df['filename'].values:
                        logging.warning(f"Image {relative_path} in {directory} not found in labels.csv")


if __name__ == "__main__":
    verify_labels()