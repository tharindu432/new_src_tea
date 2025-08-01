import cv2
import numpy as np
import os
import logging
from utils import ensure_dir, load_image


def preprocess_image(image_path, output_dir):
    """
    Preprocess a single image: grayscale, thresholding, morphological operations, contrast stretching.
    """
    try:
        # Load image
        img = load_image(image_path)

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Contrast stretching on original image
        min_val, max_val = np.min(img), np.max(img)
        if max_val > min_val:
            img_stretched = (img - min_val) * (255.0 / (max_val - min_val))
            img_stretched = img_stretched.astype(np.uint8)
        else:
            img_stretched = img

        # Save preprocessed image (using thresholded version for particle detection)
        ensure_dir(output_dir)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, thresh)
        logging.info(f"Preprocessed image saved to {output_path}")

    except Exception as e:
        logging.error(f"Error preprocessing image {image_path}: {str(e)}")
        raise


def preprocess_dataset(input_dir, output_dir):
    """
    Preprocess all images in input_dir recursively, saving to output_dir with same structure.
    """
    try:
        if not os.path.exists(input_dir):
            logging.warning(f"Input directory does not exist: {input_dir}")
            return

        ensure_dir(output_dir)
        processed_count = 0

        for root, _, files in os.walk(input_dir):
            # Calculate relative path from input_dir
            relative_path = os.path.relpath(root, input_dir)
            if relative_path == '.':
                current_output_dir = output_dir
            else:
                current_output_dir = os.path.join(output_dir, relative_path)

            # Process image files
            image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                ensure_dir(current_output_dir)

            for filename in image_files:
                try:
                    image_path = os.path.join(root, filename)
                    preprocess_image(image_path, current_output_dir)
                    processed_count += 1
                except Exception as e:
                    logging.error(f"Failed to preprocess {image_path}: {str(e)}")
                    continue

        logging.info(f"Completed preprocessing for directory {input_dir}, processed {processed_count} images")

    except Exception as e:
        logging.error(f"Error preprocessing dataset in {input_dir}: {str(e)}")
        raise