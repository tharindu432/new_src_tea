import cv2
import numpy as np
import logging
from config import Config


def segment_image(image):
    """
    Segment particles in a grayscale image and return contours.
    """
    try:
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > Config.MIN_CONTOUR_AREA]

        if not valid_contours:
            logging.warning(f"No valid contours found in image")
            return []

        logging.info(f"Found {len(valid_contours)} valid contours")
        return valid_contours
    except Exception as e:
        logging.error(f"Error segmenting image: {str(e)}")
        return []