import cv2
import numpy as np
import os
import logging
from utils import ensure_dir, load_image
from config import Config


def enhance_particle_outline(image):
    """
    Enhanced preprocessing to highlight tea particle outlines without grayscale conversion.
    """
    try:
        # Convert to LAB color space for better color separation
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(l_channel, Config.GAUSSIAN_BLUR_KERNEL, 0)

        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        # Apply bilateral filter to preserve edges while smoothing
        bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # Use adaptive thresholding for better particle separation
        thresh = cv2.adaptiveThreshold(
            bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, Config.ADAPTIVE_THRESH_BLOCK_SIZE, Config.ADAPTIVE_THRESH_C
        )

        # Morphological operations to enhance particle shapes
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, Config.MORPH_KERNEL_SIZE)
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

        # Close small gaps
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_small, iterations=1)

        # Remove small noise
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel_small, iterations=1)

        # Dilate to enhance particle boundaries
        morphed = cv2.dilate(morphed, kernel_large, iterations=Config.DILATE_ITERATIONS)

        # Erode to restore original size
        morphed = cv2.erode(morphed, kernel_large, iterations=Config.ERODE_ITERATIONS)

        return morphed

    except Exception as e:
        logging.error(f"Error enhancing particle outline: {str(e)}")
        raise


def create_outline_visualization(original_image, processed_image, contours):
    """
    Create a visualization showing particle outlines on the original image.
    """
    try:
        # Create a copy of the original image
        vis_image = original_image.copy()

        # Draw contours with different colors for better visibility
        if contours:
            # Draw filled contours with semi-transparency
            overlay = vis_image.copy()
            cv2.fillPoly(overlay, contours, (0, 255, 255))  # Yellow fill
            vis_image = cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0)

            # Draw contour outlines
            cv2.drawContours(vis_image, contours, -1, (0, 0, 255), 2)  # Red outline

            # Add contour points
            for contour in contours:
                if len(contour) > 10:  # Only for significant contours
                    for point in contour[::5]:  # Every 5th point to avoid clutter
                        cv2.circle(vis_image, tuple(point[0]), 2, (255, 0, 0), -1)  # Blue points

        return vis_image

    except Exception as e:
        logging.error(f"Error creating outline visualization: {str(e)}")
        return original_image


def preprocess_image(image_path, output_dir, visualization_dir=None):
    """
    Enhanced preprocessing pipeline for tea particle images.
    """
    try:
        # Load image in color
        img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_color is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Enhance particle outlines
        processed = enhance_particle_outline(img_color)

        # Find contours for visualization
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > Config.MIN_CONTOUR_AREA]

        # Create visualization if directory provided
        if visualization_dir and valid_contours:
            vis_image = create_outline_visualization(img_color, processed, valid_contours)
            vis_filename = f"outlined_{os.path.basename(image_path)}"
            vis_path = os.path.join(visualization_dir, vis_filename)
            cv2.imwrite(vis_path, vis_image)
            logging.info(f"Saved outline visualization: {vis_path}")

        # Save preprocessed image
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, processed)
        logging.info(f"Preprocessed image saved: {output_path}")

        return len(valid_contours)  # Return number of detected particles

    except Exception as e:
        logging.error(f"Error preprocessing image {image_path}: {str(e)}")
        raise


def preprocess_dataset(input_dir, output_dir, visualization_dir=None):
    """
    Preprocess all images in input_dir with enhanced particle detection.
    """
    try:
        if not os.path.exists(input_dir):
            logging.warning(f"Input directory does not exist: {input_dir}")
            return

        ensure_dir(output_dir)
        if visualization_dir:
            ensure_dir(visualization_dir)

        total_particles = 0
        processed_images = 0

        for root, _, files in os.walk(input_dir):
            if not files:
                continue

            relative_path = os.path.relpath(root, input_dir)
            current_output_dir = os.path.join(output_dir, relative_path)
            ensure_dir(current_output_dir)

            current_vis_dir = None
            if visualization_dir:
                current_vis_dir = os.path.join(visualization_dir, relative_path)
                ensure_dir(current_vis_dir)

            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, filename)
                    try:
                        particle_count = preprocess_image(image_path, current_output_dir, current_vis_dir)
                        total_particles += particle_count
                        processed_images += 1
                    except Exception as e:
                        logging.error(f"Failed to process {image_path}: {str(e)}")
                        continue

        logging.info(f"Preprocessing completed: {processed_images} images, {total_particles} particles detected")

    except Exception as e:
        logging.error(f"Error preprocessing dataset in {input_dir}: {str(e)}")
        raise