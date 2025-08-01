import cv2
import numpy as np
import os
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
import logging
from config import Config
from utils import load_image
from segmentation import segment_image


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
        return 1


def cluster_contours(contours):
    """
    Cluster contours to identify overlapping particles.
    """
    try:
        if not contours:
            return []

        centers = []
        for cnt in contours:
            moments = cv2.moments(cnt)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                centers.append([cx, cy])

        if not centers:
            return []

        centers = np.array(centers)
        n_clusters = min(len(centers), Config.MAX_CLUSTERS)

        if n_clusters <= 1:
            return [contours]

        kmeans = KMeans(n_clusters=n_clusters, random_state=Config.RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(centers)

        clustered_contours = []
        for i in range(n_clusters):
            cluster = [contours[j] for j in range(len(contours)) if labels[j] == i]
            if len(cluster) >= Config.MIN_CLUSTER_SIZE:
                clustered_contours.append(cluster)

        return clustered_contours
    except Exception as e:
        logging.error(f"Error clustering contours: {str(e)}")
        return []


def match_contours(contours1, contours2):
    """
    Match contours between two images to validate overlaps.
    """
    try:
        if not contours1 or not contours2:
            return []

        matches = []
        for c1 in contours1:
            min_diff = float('inf')
            match = None
            for c2 in contours2:
                try:
                    diff = cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0)
                    if diff < min_diff and diff < Config.MATCH_SHAPE_THRESHOLD:
                        min_diff = diff
                        match = c2
                except Exception:
                    continue
            if match is not None:
                matches.append((c1, match))
        return matches
    except Exception as e:
        logging.error(f"Error matching contours: {str(e)}")
        return []


def count_overlaps(image_paths, visualization_dir, sample_id):
    """
    Count overlapping particles across images and visualize results.
    """
    try:
        if not image_paths:
            return 1

        images = []
        for path in image_paths:
            try:
                img = load_image(path)
                images.append(img)
            except Exception as e:
                logging.warning(f"Failed to load image {path}: {str(e)}")
                continue

        if not images:
            return 1

        contours_list = []
        for img in images:
            contours = segment_image(img)
            contours_list.append(contours)

        # Estimate overlaps using histogram peaks
        peak_counts = [detect_peaks(img) for img in images]
        estimated_overlaps = max(1, int(np.mean(peak_counts)))

        # Cluster contours to refine overlap count
        clustered_contours = [cluster_contours(contours) for contours in contours_list]
        valid_clusters = [clusters for clusters in clustered_contours if clusters]

        if valid_clusters:
            min_clusters = min([len(clusters) for clusters in valid_clusters])
            estimated_overlaps = min(estimated_overlaps, max(1, min_clusters))

        # Cross-image contour matching for validation if we have multiple images
        overlap_count = estimated_overlaps
        if len(contours_list) >= 2:
            matches_12 = match_contours(contours_list[0], contours_list[1])
            if len(contours_list) >= 3:
                matches_23 = match_contours(contours_list[1], contours_list[2])
                # Find common matches
                common_matches = 0
                for match1 in matches_12:
                    for match2 in matches_23:
                        if np.array_equal(match1[1], match2[0]):  # Same contour in image 2
                            common_matches += 1
                            break
                overlap_count = max(1, common_matches)
            else:
                overlap_count = max(1, len(matches_12))

        # Adjust overlap count based on area
        total_area = sum(cv2.contourArea(cnt) for contours in contours_list for cnt in contours)
        total_contours = sum(len(contours) for contours in contours_list)

        if total_contours > 0:
            avg_contour_area = total_area / total_contours
            if avg_contour_area > Config.AVG_PARTICLE_AREA:
                overlap_count = max(1, int(overlap_count * (avg_contour_area / Config.AVG_PARTICLE_AREA)))

        # Visualize overlaps
        if images and visualization_dir:
            try:
                os.makedirs(visualization_dir, exist_ok=True)
                vis_image = cv2.cvtColor(images[0], cv2.COLOR_GRAY2BGR)

                # Draw all contours in green
                for contours in contours_list:
                    cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 1)

                # Draw matched contours in red if available
                if len(contours_list) >= 2:
                    matches = match_contours(contours_list[0], contours_list[1])
                    for match in matches:
                        cv2.drawContours(vis_image, [match[0]], -1, (0, 0, 255), 2)

                output_path = os.path.join(visualization_dir, f"{sample_id}_overlap.png")
                cv2.imwrite(output_path, vis_image)
                logging.info(f"Saved overlap visualization to {output_path}")
            except Exception as e:
                logging.warning(f"Failed to save visualization for {sample_id}: {str(e)}")

        return max(1, overlap_count)

    except Exception as e:
        logging.error(f"Error counting overlaps for {sample_id}: {str(e)}")
        return 1