import cv2
import numpy as np
from scipy.fft import fft
import logging


def extract_shape_features(contours):
    """
    Extract shape features from contours: Hu Moments, Fourier Descriptors, aspect ratio, circularity.
    """
    try:
        if not contours:
            return np.array([])

        features = []
        for contour in contours:
            try:
                # Hu Moments
                moments = cv2.moments(contour)
                hu_moments = cv2.HuMoments(moments).flatten()
                # Handle log of zero or negative values
                hu_moments = np.where(hu_moments != 0,
                                      -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10),
                                      0)

                # Fourier Descriptors
                contour_points = contour.squeeze()
                if len(contour_points.shape) == 2 and contour_points.shape[0] > 10:
                    complex_coords = contour_points[:, 0] + 1j * contour_points[:, 1]
                    coeffs = fft(complex_coords)
                    fd = np.abs(coeffs[:10]) if len(coeffs) >= 10 else np.pad(np.abs(coeffs), (0, 10 - len(coeffs)),
                                                                              'constant')
                else:
                    fd = np.zeros(10)

                # Aspect Ratio and Circularity
                (x, y), (w, h), _ = cv2.minAreaRect(contour)
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

                # Combine all features
                feature_vector = np.concatenate([hu_moments, fd, [aspect_ratio, circularity]])
                features.append(feature_vector)

            except Exception as e:
                logging.warning(f"Error processing contour: {str(e)}")
                # Return a default feature vector if contour processing fails
                default_features = np.zeros(19)  # 7 Hu moments + 10 Fourier + 2 geometric
                features.append(default_features)

        return np.array(features) if features else np.array([])

    except Exception as e:
        logging.error(f"Error extracting shape features: {str(e)}")
        return np.array([])