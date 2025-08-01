import cv2
import numpy as np
import pandas as pd
import os
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
        # Check if input directory exists
        if not os.path.exists(input_dir):
            logging.warning(f"Input directory does not exist: {input_dir}")
            return np.array([]), [], []

        # Load labels
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Labels file {label_file} not found")

        labels_df = pd.read_csv(label_file)
        if labels_df.empty:
            raise ValueError("Labels file is empty")

        # Get sample groups
        sample_groups = get_sample_groups(input_dir)
        if not sample_groups:
            logging.warning(f"No valid samples found in {input_dir}")
            return np.array([]), [], []

        features = []
        labels = []
        overlap_counts = []
        processed_samples = 0

        for idx, (sample_id, image_paths) in enumerate(sample_groups):
            # Process all samples, not just first VISUALIZE_SAMPLES
            try:
                # Load images and segment
                images = []
                for path in image_paths:
                    try:
                        img = load_image(path)
                        images.append(img)
                    except Exception as e:
                        logging.warning(f"Failed to load image {path}: {str(e)}")
                        continue

                if not images:
                    logging.warning(f"No valid images for sample {sample_id}")
                    continue

                contours_list = []
                for img in images:
                    contours = segment_image(img)
                    contours_list.append(contours)

                # Count overlaps
                overlap_count = count_overlaps(image_paths, visualization_dir, sample_id)
                overlap_counts.append(overlap_count)

                # Extract shape features
                shape_features = []
                for contours in contours_list:
                    if contours:
                        sf = extract_shape_features(contours)
                        if sf.size > 0:
                            shape_features.append(sf)

                # Aggregate features
                if shape_features:
                    # Calculate mean features across all images and contours
                    all_features = []
                    for sf in shape_features:
                        if sf.ndim == 2:  # Multiple contours
                            all_features.extend(sf)
                        elif sf.ndim == 1:  # Single contour
                            all_features.append(sf)

                    if all_features:
                        sample_features = np.mean(all_features, axis=0)
                    else:
                        sample_features = np.zeros(19)  # Default feature vector
                else:
                    sample_features = np.zeros(19)  # Default feature vector if no contours

                # Add overlap count as feature
                sample_features = np.append(sample_features, overlap_count)

                # Get label - extract base sample name for matching
                base_sample_name = sample_id.split('_')[0]  # Extract base name (e.g., SAMPLE_01 from SAMPLE_01_TAKE_01)

                # Find matching labels
                sample_labels = labels_df[labels_df['filename'].str.contains(base_sample_name, case=False, na=False)]

                if sample_labels.empty:
                    logging.warning(f"No labels found for {sample_id} (base: {base_sample_name})")
                    continue

                # Use first matching label
                label_row = sample_labels.iloc[0]
                label = label_row['tea_variant']
                elevation = label_row['elevation']
                combined_label = f"{label}_{elevation}"

                labels.append(combined_label)
                features.append(sample_features)
                processed_samples += 1

                logging.info(
                    f"Processed sample {sample_id}: {len(image_paths)} images, overlap_count={overlap_count}, label={combined_label}")

            except Exception as e:
                logging.error(f"Error processing sample {sample_id}: {str(e)}")
                continue

        if not features:
            logging.warning(f"No valid features extracted from {input_dir}")
            return np.array([]), [], []

        logging.info(f"Successfully extracted features from {processed_samples} samples in {input_dir}")
        return np.array(features), labels, overlap_counts

    except Exception as e:
        logging.error(f"Error extracting features from {input_dir}: {str(e)}")
        return np.array([]), [], []