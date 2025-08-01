import os
import cv2
import logging
from pathlib import Path


def ensure_dir(directory):
    """
    Create directory if it doesn't exist.
    """
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logging.info(f"Ensured directory exists: {directory}")
    except Exception as e:
        logging.error(f"Error creating directory {directory}: {str(e)}")
        raise


def load_image(image_path):
    """
    Load an image in grayscale.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return img
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {str(e)}")
        raise


def get_sample_groups(input_dir):
    """
    Group images by sample or take in PARTICLE directories, allowing any number of images.
    Returns list of tuples: (sample_id, [image_paths]).
    """
    try:
        sample_groups = []
        for root, _, files in os.walk(input_dir):
            if "PARTICLE" not in root:
                continue
            # Old dataset: PARTICLE/SAMPLE_XX
            if os.path.basename(root).startswith("SAMPLE_"):
                image_files = [os.path.join(root, f) for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
                if image_files:  # Accept any number of images
                    sample_id = os.path.basename(root)
                    sample_groups.append((sample_id, sorted(image_files)))
                    logging.info(f"Found sample {sample_id} with {len(image_files)} images")
                else:
                    logging.warning(f"Sample {root} has no valid images")
            # New dataset: PARTICLE/SAMPLE_XX/TAKE_XX
            elif os.path.basename(root).startswith("TAKE_"):
                image_files = [os.path.join(root, f) for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
                if image_files:  # Accept any number of images
                    sample_id = f"{os.path.basename(os.path.dirname(root))}_{os.path.basename(root)}"
                    sample_groups.append((sample_id, sorted(image_files)))
                    logging.info(f"Found take {sample_id} with {len(image_files)} images")
                else:
                    logging.warning(f"Take {root} has no valid images")

        if not sample_groups:
            logging.warning(f"No valid sample groups found in {input_dir}")
            return []

        logging.info(f"Found {len(sample_groups)} sample groups in {input_dir}")
        return sample_groups
    except Exception as e:
        logging.error(f"Error getting sample groups from {input_dir}: {str(e)}")
        return []