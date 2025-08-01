import cv2
import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
import logging
from config import Config
from utils import load_image


def detect_peaks(image):
    """
    Detect intensity peaks in the image histogram.
    """
    try:
        hist, _ = np.histogram(image.ravel(), bins=Config.HISTOGRAM_BINS, range=(0, 255))
        peaks, _ = find_peaks(hist, height=Config.MIN_PEAK_HEIGHT)
        return len(peaks)
    except Exception as e:
        logging.error(f"Error detecting peaks: {str(e)}")
        raise


def cluster_contours(contours):
    """
    Cluster contours to identify overlapping particles.
    """
    try:
        if not contours:
            return []
        centers = [cv2.moments(cnt)['m00'] and np.array([cv2.moments(cnt)['m10'] / cv2.moments(cnt)['m00'],
                                                         cv2.moments(cnt)['m01'] / cv2.moments(cnt)['m00']])
                   for cnt in contours]
        centers = np.array(centers)
        kmeans = KMeans(n_clusters=min(len(centers), Config.MAX_CLUSTERS), random_state=Config.RANDOM_STATE)
        kmeans.fit(centers)
        labels = kmeans.labels_
        clustered_contours = []
        for i in range(min(len(centers), Config.MAX_CLUSTERS)):
            cluster = [contours[j] for j in range(len(contours)) if labels[j] == i]
            if len(cluster) >= Config.MIN_CLUSTER_SIZE:
                clustered_contours.append(cluster)
        return clustered_contours
    except Exception as e:
        logging.error(f"Error clustering contours: {str(e)}")
        raise


def match_contours(contours1, contours2):
    """
    Match contours between two images to validate overlaps.
    """
    try:
        matches = []
        for c1 in contours1:
            min_diff = float('inf')
            match = None
            for c2 in contours2:
                diff = cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0)
                if diff < min_diff and diff < Config.MATCH_SHAPE_THRESHOLD:
                    min_diff = diff
                    match = c2
            if match is not None:
                matches.append((c1, match))
        return matches
    except Exception as e:
        logging.error(f"Error matching contours: {str(e)}")
        raise


def count_overlaps(image_paths, visualization_dir, sample_id):
    """
    Count overlapping particles across three images and visualize results.
    """
    try:
        images = [load_image(path) for path in image_paths]
        contours_list = [segment_image(img) for img in images]

        # Estimate overlaps using histogram peaks
        peak_counts = [detect_peaks(img) for img in images]
        estimated_overlaps = max(1, int(np.mean(peak_counts)))

        # Cluster contours to refine overlap count
        clustered_contours = [cluster_contours(contours) for contours in contours_list]
        min_clusters = min([len(clusters) for clusters in clustered_contours if clusters])
        estimated_overlaps = min(estimated_overlaps, min_clusters)

        # Cross-image contour matching for validation
        matches_12 = match_contours(contours_list[0], contours_list[1])
        matches_23 = match_contours(contours_list[1], contours_list[2])
        overlap_count = len(set(matches_12).intersection(set(matches_23)))

        # Adjust overlap count based on area
        total_area = sum(cv2.contourArea(cnt) for contours in contours_list for cnt in contours)
        avg_contour_area = total_area / sum(
            len(contours) for contours in contours_list) if contours_list else Config.AVG_PARTICLE_AREA
        if avg_contour_area > Config.AVG_PARTICLE_AREA:
            overlap_count = max(1, int(overlap_count * (avg_contour_area / Config.AVG_PARTICLE_AREA)))

        # Visualize overlaps
        vis_image = cv2.cvtColor(images[0], cv2.COLOR_GRAY2BGR)
        for contours in contours_list:
            cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 1)
        for match in matches_12:
            cv2.drawContours(vis_image, [match[0]], -1, (0, 0, 255), 2)
        output_path = os.path.join(visualization_dir, f"{sample_id}_overlap.png")
        cv2.imwrite(output_path, vis_image)
        logging.info(f"Saved overlap visualization to {output_path}")

        return max(1, overlap_count)
    except Exception as e:
        logging.error(f"Error counting overlaps for {sample_id}: {str(e)}")
        raise