import cv2
import numpy as np
import pandas as pd
import logging
from config import Config
from utils import load_image, get_sample_groups
from segmentation import segment_image
from overlap_handling import count_overlaps
from shape_analysis import extract_shape_features


def extract_features(input_dir, label_file, visualization_dir):
    """
    Extract features and labels for all samples in input_dir, handling variable image counts.
    Returns feature matrix, labels, and overlap counts.
    """
    try:
        # Load labels
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Labels file {label_file} not found")
        labels_df = pd.read_csv(label_file)
        if labels_df.empty:
            raise ValueError("Labels file is empty")

        # Get sample groups
        sample_groups = get_sample_groups(input_dir)
        if not sample_groups:
            raise ValueError(f"No valid samples found in {input_dir}")

        features = []
        labels = []
        overlap_counts = []

        for idx, (sample_id, image_paths) in enumerate(sample_groups):
            if idx >= Config.VISUALIZE_SAMPLES:
                break
            try:
                # Load images and segment
                images = [load_image(path) for path in image_paths]
                contours_list = [segment_image(img) for img in images]

                # Count overlaps
                overlap_count = count_overlaps(image_paths, visualization_dir, sample_id)
                overlap_counts.append(overlap_count)

                # Extract shape features
                shape_features = [extract_shape_features(contours) for contours in contours_list]
                shape_features = [sf for sf in shape_features if len(sf) > 0]

                # Aggregate features
                if shape_features:
                    sample_features = np.mean([np.mean(sf, axis=0) for sf in shape_features], axis=0)
                else:
                    sample_features = np.zeros(19)  # Default feature vector if no contours
                sample_features = np.append(sample_features, overlap_count)

                # Get label
                sample_labels = labels_df[labels_df['filename'].str.contains(sample_id.split('_')[0])]
                if sample_labels.empty:
                    logging.warning(f"No labels found for {sample_id}")
                    continue
                label = sample_labels.iloc[0]['tea_variant']
                elevation = sample_labels.iloc[0]['elevation']
                labels.append(f"{label}_{elevation}")

                features.append(sample_features)

            except Exception as e:
                logging.error(f"Error processing sample {sample_id}: {str(e)}")
                continue

        if not features:
            raise ValueError("No valid features extracted")

        return np.array(features), labels, overlap_counts
    except Exception as e:
        logging.error(f"Error extracting features from {input_dir}: {str(e)}")
        raise