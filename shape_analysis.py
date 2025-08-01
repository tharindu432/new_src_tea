import cv2
import numpy as np
from scipy.fft import fft
import logging


def extract_shape_features(contours):
    """
    Extract shape features from contours: Hu Moments, Fourier Descriptors, aspect ratio, circularity.
    """
    try:
        features = []
        for contour in contours:
            # Hu Moments
            moments = cv2.moments(contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

            # Fourier Descriptors
            contour_points = contour.squeeze()
            if len(contour_points) > 10:
                coeffs = fft(contour_points[:, 0] + 1j * contour_points[:, 1])
                fd = np.abs(coeffs[:10])
            else:
                fd = np.zeros(10)

            # Aspect Ratio and Circularity
            (x, y), (w, h), _ = cv2.minAreaRect(contour)
            aspect_ratio = w / h if h > 0 else 1.0
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

            features.append(np.concatenate([hu_moments, fd, [aspect_ratio, circularity]]))
        return np.array(features)
    except Exception as e:
        logging.error(f"Error extracting shape features: {str(e)}")
        raise