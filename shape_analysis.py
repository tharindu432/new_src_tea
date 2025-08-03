import cv2
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from config import Config


def calculate_shape_descriptors(contour):
    """
    Calculate comprehensive shape descriptors for a contour.
    """
    try:
        # Basic measurements
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area == 0 or perimeter == 0:
            return None

        # Bounding rectangle and ellipse
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0

        # Minimum enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * radius * radius
        solidity_circle = area / circle_area if circle_area > 0 else 0

        # Convex hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        # Aspect ratio
        aspect_ratio = w / h if h > 0 else 0

        # Circularity (compactness)
        circularity = (4 * np.pi * area) / (perimeter ** 2)

        # Rectangularity
        rectangularity = area / rect_area if rect_area > 0 else 0

        # Elongation - Fixed calculation
        if area > 0:
            # Get moments
            moments = cv2.moments(contour)

            # Create covariance matrix from central moments
            mu20 = moments['mu20']
            mu02 = moments['mu02']
            mu11 = moments['mu11']

            # Construct covariance matrix
            cov_matrix = np.array([[mu20, mu11],
                                   [mu11, mu02]]) / moments['m00'] if moments['m00'] > 0 else np.eye(2)

            # Calculate eigenvalues
            eigenvalues = np.linalg.eigvals(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order

            if len(eigenvalues) >= 2 and eigenvalues[1] > 0:
                elongation = eigenvalues[0] / eigenvalues[1]
            else:
                elongation = 1.0
        else:
            elongation = 1.0

        return {
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': aspect_ratio,
            'circularity': circularity,
            'solidity': solidity,
            'extent': extent,
            'rectangularity': rectangularity,
            'elongation': elongation,
            'solidity_circle': solidity_circle
        }

    except Exception as e:
        logging.error(f"Error calculating shape descriptors: {str(e)}")
        return None


def extract_fourier_descriptors(contour, n_descriptors=None):
    """
    Extract Fourier descriptors from contour.
    """
    try:
        if n_descriptors is None:
            n_descriptors = Config.N_FOURIER_DESCRIPTORS

        # Get contour points
        contour_points = contour.squeeze()
        if len(contour_points.shape) == 1 or len(contour_points) < Config.MIN_CONTOUR_POINTS:
            return np.zeros(n_descriptors)

        # Create complex representation
        complex_contour = contour_points[:, 0] + 1j * contour_points[:, 1]

        # Apply FFT
        fft_result = fft(complex_contour)

        # Get magnitude of first n_descriptors coefficients (excluding DC component)
        descriptors = np.abs(fft_result[1:n_descriptors + 1])

        # Normalize by the first descriptor to achieve scale invariance
        if len(descriptors) > 0 and descriptors[0] > 0:
            descriptors = descriptors / descriptors[0]

        return descriptors

    except Exception as e:
        logging.error(f"Error extracting Fourier descriptors: {str(e)}")
        return np.zeros(n_descriptors if n_descriptors else Config.N_FOURIER_DESCRIPTORS)


def extract_hu_moments(contour):
    """
    Extract Hu moments from contour.
    """
    try:
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()

        # Apply log transform for numerical stability
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

        return hu_moments

    except Exception as e:
        logging.error(f"Error extracting Hu moments: {str(e)}")
        return np.zeros(7)


def extract_shape_features(contours):
    """
    Extract comprehensive shape features from all contours.
    """
    try:
        if not contours:
            return np.array([])

        features_list = []

        for contour in contours:
            if cv2.contourArea(contour) < Config.MIN_CONTOUR_AREA:
                continue

            # Shape descriptors
            shape_desc = calculate_shape_descriptors(contour)
            if shape_desc is None:
                continue

            # Hu moments
            hu_moments = extract_hu_moments(contour)

            # Fourier descriptors
            fourier_desc = extract_fourier_descriptors(contour)

            # Combine all features
            feature_vector = np.concatenate([
                list(shape_desc.values()),
                hu_moments,
                fourier_desc
            ])

            features_list.append(feature_vector)

        if not features_list:
            # Return zero vector with expected dimensions
            expected_dim = 9 + 7 + Config.N_FOURIER_DESCRIPTORS  # shape_desc + hu + fourier
            return np.zeros((1, expected_dim))

        return np.array(features_list)

    except Exception as e:
        logging.error(f"Error extracting shape features: {str(e)}")
        expected_dim = 9 + 7 + Config.N_FOURIER_DESCRIPTORS
        return np.zeros((1, expected_dim))


def visualize_shape_analysis(contours, features, sample_id, visualization_dir):
    """
    Create comprehensive shape analysis visualizations.
    """
    try:
        if not contours or len(features) == 0:
            logging.warning(f"No contours or features to visualize for {sample_id}")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Shape Analysis - {sample_id}', fontsize=16)

        # Extract shape descriptors from features
        shape_names = ['area', 'perimeter', 'aspect_ratio', 'circularity',
                       'solidity', 'extent', 'rectangularity', 'elongation', 'solidity_circle']

        if features.shape[1] >= len(shape_names):
            shape_features = features[:, :len(shape_names)]

            # 1. Area distribution
            axes[0, 0].hist(shape_features[:, 0], bins=10, alpha=0.7, color='blue')
            axes[0, 0].set_title('Area Distribution')
            axes[0, 0].set_xlabel('Area (pixels²)')
            axes[0, 0].set_ylabel('Frequency')

            # 2. Circularity vs Solidity
            axes[0, 1].scatter(shape_features[:, 3], shape_features[:, 4], alpha=0.7, c='red')
            axes[0, 1].set_title('Circularity vs Solidity')
            axes[0, 1].set_xlabel('Circularity')
            axes[0, 1].set_ylabel('Solidity')

            # 3. Aspect ratio distribution
            axes[0, 2].hist(shape_features[:, 2], bins=10, alpha=0.7, color='green')
            axes[0, 2].set_title('Aspect Ratio Distribution')
            axes[0, 2].set_xlabel('Aspect Ratio')
            axes[0, 2].set_ylabel('Frequency')

            # 4. Shape feature correlation heatmap
            if len(shape_features) > 1:
                corr_matrix = np.corrcoef(shape_features.T)
                im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                axes[1, 0].set_title('Feature Correlation')
                axes[1, 0].set_xticks(range(len(shape_names)))
                axes[1, 0].set_yticks(range(len(shape_names)))
                axes[1, 0].set_xticklabels(shape_names, rotation=45, ha='right')
                axes[1, 0].set_yticklabels(shape_names)
                plt.colorbar(im, ax=axes[1, 0])

            # 5. Elongation vs Rectangularity
            axes[1, 1].scatter(shape_features[:, 7], shape_features[:, 6], alpha=0.7, c='purple')
            axes[1, 1].set_title('Elongation vs Rectangularity')
            axes[1, 1].set_xlabel('Elongation')
            axes[1, 1].set_ylabel('Rectangularity')

            # 6. Feature summary statistics
            mean_values = np.mean(shape_features, axis=0)
            std_values = np.std(shape_features, axis=0)

            x_pos = np.arange(len(shape_names))
            axes[1, 2].bar(x_pos, mean_values, yerr=std_values, alpha=0.7, capsize=5)
            axes[1, 2].set_title('Mean Feature Values (±std)')
            axes[1, 2].set_xticks(x_pos)
            axes[1, 2].set_xticklabels(shape_names, rotation=45, ha='right')
            axes[1, 2].set_ylabel('Value')

        plt.tight_layout()

        # Save plot
        output_path = os.path.join(visualization_dir, f'{sample_id}_shape_analysis.png')
        plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
        plt.close()

        logging.info(f"Saved shape analysis visualization: {output_path}")

    except Exception as e:
        logging.error(f"Error creating shape analysis visualization for {sample_id}: {str(e)}")


def create_fourier_descriptor_plot(features_dict, visualization_dir):
    """
    Create plots showing Fourier descriptor patterns across different tea types.
    """
    try:
        if not features_dict:
            return

        plt.figure(figsize=(12, 8))

        colors = plt.cm.Set3(np.linspace(0, 1, len(features_dict)))

        for i, (tea_type, features) in enumerate(features_dict.items()):
            if len(features) == 0:
                continue

            # Extract Fourier descriptors (last 15 features)
            fourier_features = features[:, -Config.N_FOURIER_DESCRIPTORS:]
            mean_fourier = np.mean(fourier_features, axis=0)
            std_fourier = np.std(fourier_features, axis=0)

            x = np.arange(1, len(mean_fourier) + 1)
            plt.errorbar(x, mean_fourier, yerr=std_fourier,
                         label=tea_type, color=colors[i],
                         marker='o', capsize=3, alpha=0.8)

        plt.xlabel('Fourier Descriptor Index')
        plt.ylabel('Normalized Magnitude')
        plt.title('Fourier Descriptors by Tea Type')
        plt.legend()
        plt.grid(True, alpha=0.3)

        output_path = os.path.join(visualization_dir, 'fourier_descriptors_comparison.png')
        plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
        plt.close()

        logging.info(f"Saved Fourier descriptor comparison: {output_path}")

    except Exception as e:
        logging.error(f"Error creating Fourier descriptor plot: {str(e)}")


def analyze_shape_distribution(all_features, all_labels, visualization_dir):
    """
    Analyze and visualize shape feature distributions across different tea types.
    """
    try:
        if len(all_features) == 0:
            return

        # Group features by tea type
        features_by_type = {}
        for features, label in zip(all_features, all_labels):
            if label not in features_by_type:
                features_by_type[label] = []
            features_by_type[label].append(features)

        # Convert to arrays
        for tea_type in features_by_type:
            features_by_type[tea_type] = np.vstack(features_by_type[tea_type])

        # Create Fourier descriptor comparison
        create_fourier_descriptor_plot(features_by_type, visualization_dir)

        # Create comprehensive comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Shape Feature Analysis Across Tea Types', fontsize=16)

        shape_indices = {
            'circularity': 3,
            'solidity': 4,
            'aspect_ratio': 2,
            'area': 0
        }

        plot_configs = [
            ('circularity', 'Circularity Distribution', axes[0, 0]),
            ('solidity', 'Solidity Distribution', axes[0, 1]),
            ('aspect_ratio', 'Aspect Ratio Distribution', axes[1, 0]),
            ('area', 'Area Distribution', axes[1, 1])
        ]

        colors = plt.cm.Set3(np.linspace(0, 1, len(features_by_type)))

        for feature_name, title, ax in plot_configs:
            feature_idx = shape_indices[feature_name]

            for i, (tea_type, features) in enumerate(features_by_type.items()):
                if features.shape[1] > feature_idx:
                    feature_values = features[:, feature_idx]
                    ax.hist(feature_values, bins=15, alpha=0.6,
                            label=tea_type, color=colors[i])

            ax.set_title(title)
            ax.set_xlabel(feature_name.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = os.path.join(visualization_dir, 'shape_distribution_analysis.png')
        plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
        plt.close()

        logging.info(f"Saved shape distribution analysis: {output_path}")

    except Exception as e:
        logging.error(f"Error analyzing shape distribution: {str(e)}")