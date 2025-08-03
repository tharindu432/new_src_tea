import cv2
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from config import Config
import pandas as pd


class OverlapAnalyzer:
    """Advanced overlap detection and analysis for tea particles."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_contour_overlap(self, contours):
        """
        Analyze potential overlaps between contours based on spatial proximity and size.
        """
        try:
            if len(contours) < 2:
                return 0, []

            overlap_count = 0
            overlap_pairs = []

            # Calculate bounding rectangles and centroids
            contour_info = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < Config.MIN_CONTOUR_AREA:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                else:
                    cx, cy = x + w / 2, y + h / 2

                contour_info.append({
                    'index': i,
                    'contour': contour,
                    'area': area,
                    'bbox': (x, y, w, h),
                    'centroid': (cx, cy),
                    'perimeter': cv2.arcLength(contour, True)
                })

            # Check for overlaps
            for i in range(len(contour_info)):
                for j in range(i + 1, len(contour_info)):
                    info1, info2 = contour_info[i], contour_info[j]

                    # Distance between centroids
                    dist = np.sqrt((info1['centroid'][0] - info2['centroid'][0]) ** 2 +
                                   (info1['centroid'][1] - info2['centroid'][1]) ** 2)

                    # Expected separation based on particle sizes
                    avg_size = np.sqrt((info1['area'] + info2['area']) / (2 * np.pi))

                    # Check if particles are too close (potential overlap)
                    if dist < avg_size * 1.2:  # Overlap threshold
                        # Additional checks for actual overlap
                        if self.check_bbox_overlap(info1['bbox'], info2['bbox']):
                            overlap_count += 1
                            overlap_pairs.append((i, j))

                    # Check for unusually large particles (potential multiple particles)
                    if info1['area'] > Config.AVG_PARTICLE_AREA * 2:
                        # Analyze if this could be multiple overlapping particles
                        if self.analyze_large_particle(info1['contour']):
                            overlap_count += 1

            return overlap_count, overlap_pairs

        except Exception as e:
            self.logger.error(f"Error analyzing contour overlap: {str(e)}")
            return 0, []

    def check_bbox_overlap(self, bbox1, bbox2):
        """Check if two bounding boxes overlap."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

    def analyze_large_particle(self, contour):
        """Analyze if a large contour might contain multiple overlapping particles."""
        try:
            # Calculate shape descriptors
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            # Circularity (4π*area/perimeter²)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0

            # If circularity is very low, might be overlapping particles
            if circularity < 0.3:
                return True

            # Check for convexity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < 0.7:  # Low solidity suggests overlapping
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Error analyzing large particle: {str(e)}")
            return False

    def cluster_based_overlap_detection(self, contours, image_shape):
        """Use clustering to detect potential overlapping regions."""
        try:
            if len(contours) < 3:
                return 0

            # Extract features for clustering
            features = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < Config.MIN_CONTOUR_AREA:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                else:
                    cx, cy = x + w / 2, y + h / 2

                features.append([cx, cy, area, w * h])

            if len(features) < 3:
                return 0

            features = np.array(features)

            # Normalize features
            scaler = StandardScaler()
            features_normalized = scaler.fit_transform(features)

            # Use DBSCAN to find dense regions (potential overlaps)
            dbscan = DBSCAN(eps=0.5, min_samples=2)
            clusters = dbscan.fit_predict(features_normalized[:, :2])  # Use only position

            # Count dense clusters as potential overlap regions
            unique_clusters = set(clusters)
            overlap_regions = len([c for c in unique_clusters if c != -1 and np.sum(clusters == c) >= 2])

            return overlap_regions

        except Exception as e:
            self.logger.error(f"Error in cluster-based overlap detection: {str(e)}")
            return 0

    def count_overlaps(self, image_paths, sample_id, output_dir):
        """
        Comprehensive overlap analysis for a sample with multiple images.
        """
        try:
            self.logger.info(f"Analyzing overlaps for sample {sample_id}")

            total_overlap_count = 0
            overlap_details = []
            all_contours = []

            for i, image_path in enumerate(image_paths):
                try:
                    # Load image
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        self.logger.warning(f"Could not load image: {image_path}")
                        continue

                    # Simple segmentation for overlap analysis
                    contours = self.simple_segmentation(image)
                    all_contours.extend(contours)

                    # Analyze overlaps in this image
                    overlap_count, overlap_pairs = self.analyze_contour_overlap(contours)

                    # Cluster-based detection
                    cluster_overlaps = self.cluster_based_overlap_detection(contours, image.shape)

                    # Combine results (take maximum to avoid double counting)
                    image_overlaps = max(overlap_count, cluster_overlaps)
                    total_overlap_count += image_overlaps

                    overlap_details.append({
                        'image_index': i,
                        'image_path': os.path.basename(image_path),
                        'contours_found': len(contours),
                        'overlap_count': image_overlaps,
                        'overlap_pairs': len(overlap_pairs),
                        'cluster_overlaps': cluster_overlaps
                    })

                    self.logger.info(f"Image {i + 1}: {len(contours)} contours, {image_overlaps} overlaps")

                except Exception as e:
                    self.logger.error(f"Error processing image {image_path}: {str(e)}")
                    continue

            # Cross-image overlap analysis
            cross_image_overlaps = self.analyze_cross_image_patterns(all_contours)
            total_overlap_count += cross_image_overlaps

            # Save detailed overlap analysis
            self.save_overlap_analysis(sample_id, overlap_details, total_overlap_count,
                                       cross_image_overlaps, output_dir)

            # Create visualization
            self.create_overlap_visualization(sample_id, overlap_details, output_dir)

            self.logger.info(f"Total overlaps detected for {sample_id}: {total_overlap_count}")
            return total_overlap_count

        except Exception as e:
            self.logger.error(f"Error counting overlaps for {sample_id}: {str(e)}")
            return 1  # Return default value

    def simple_segmentation(self, image):
        """Simple segmentation for overlap analysis."""
        try:
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(image, (5, 5), 0)

            # Otsu thresholding
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter by area
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if Config.MIN_CONTOUR_AREA <= area <= Config.MAX_CONTOUR_AREA:
                    valid_contours.append(contour)

            return valid_contours

        except Exception as e:
            self.logger.error(f"Error in simple segmentation: {str(e)}")
            return []

    def analyze_cross_image_patterns(self, all_contours):
        """Analyze patterns across multiple images for consistency."""
        try:
            if len(all_contours) < 5:
                return 0

            # Calculate total area and average particle size
            total_area = sum(cv2.contourArea(cnt) for cnt in all_contours)
            avg_area = total_area / len(all_contours)

            # Check for unusually large particles that might be overlaps
            large_particle_count = sum(1 for cnt in all_contours
                                       if cv2.contourArea(cnt) > avg_area * 2.5)

            # Estimate overlaps based on size distribution
            cross_image_overlaps = int(large_particle_count * 0.7)  # Heuristic

            return cross_image_overlaps

        except Exception as e:
            self.logger.error(f"Error in cross-image pattern analysis: {str(e)}")
            return 0

    def save_overlap_analysis(self, sample_id, overlap_details, total_overlaps,
                              cross_image_overlaps, output_dir):
        """Save detailed overlap analysis results."""
        try:
            os.makedirs(output_dir, exist_ok=True)

            # Save detailed CSV
            df = pd.DataFrame(overlap_details)
            csv_path = os.path.join(output_dir, f'{sample_id}_overlap_details.csv')
            df.to_csv(csv_path, index=False)

            # Save summary report
            report_path = os.path.join(output_dir, f'{sample_id}_overlap_summary.txt')
            with open(report_path, 'w') as f:
                f.write(f"Overlap Analysis Report - {sample_id}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total Images Analyzed: {len(overlap_details)}\n")
                f.write(f"Total Contours Found: {sum(d['contours_found'] for d in overlap_details)}\n")
                f.write(f"Total Overlaps Detected: {total_overlaps}\n")
                f.write(f"Cross-Image Overlaps: {cross_image_overlaps}\n\n")

                f.write("Per-Image Results:\n")
                for detail in overlap_details:
                    f.write(f"  {detail['image_path']}: {detail['overlap_count']} overlaps "
                            f"from {detail['contours_found']} contours\n")

            self.logger.info(f"Saved overlap analysis: {csv_path}, {report_path}")

        except Exception as e:
            self.logger.error(f"Error saving overlap analysis: {str(e)}")

    def create_overlap_visualization(self, sample_id, overlap_details, output_dir):
        """Create comprehensive overlap visualization."""
        try:
            if not overlap_details:
                return

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Overlap Analysis - {sample_id}', fontsize=16, fontweight='bold')

            # 1. Overlaps per image
            image_indices = [d['image_index'] for d in overlap_details]
            overlap_counts = [d['overlap_count'] for d in overlap_details]

            axes[0, 0].bar(image_indices, overlap_counts, color='skyblue', alpha=0.7)
            axes[0, 0].set_xlabel('Image Index')
            axes[0, 0].set_ylabel('Overlap Count')
            axes[0, 0].set_title('Overlaps per Image')
            axes[0, 0].grid(True, alpha=0.3)

            # 2. Contours vs Overlaps scatter
            contour_counts = [d['contours_found'] for d in overlap_details]
            axes[0, 1].scatter(contour_counts, overlap_counts, alpha=0.7, s=50)
            axes[0, 1].set_xlabel('Contours Found')
            axes[0, 1].set_ylabel('Overlaps Detected')
            axes[0, 1].set_title('Contours vs Overlaps Relationship')
            axes[0, 1].grid(True, alpha=0.3)

            # Add trend line
            if len(contour_counts) > 1:
                z = np.polyfit(contour_counts, overlap_counts, 1)
                p = np.poly1d(z)
                axes[0, 1].plot(contour_counts, p(contour_counts), "r--", alpha=0.8)

            # 3. Overlap distribution
            if overlap_counts:
                axes[1, 0].hist(overlap_counts, bins=max(3, len(set(overlap_counts))),
                                alpha=0.7, color='lightcoral', edgecolor='black')
                axes[1, 0].set_xlabel('Overlap Count')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Overlap Count Distribution')
                axes[1, 0].grid(True, alpha=0.3)

            # 4. Summary statistics
            total_contours = sum(contour_counts)
            total_overlaps = sum(overlap_counts)
            avg_overlaps = np.mean(overlap_counts) if overlap_counts else 0

            axes[1, 1].text(0.1, 0.8, f'Total Images: {len(overlap_details)}',
                            transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].text(0.1, 0.7, f'Total Contours: {total_contours}',
                            transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].text(0.1, 0.6, f'Total Overlaps: {total_overlaps}',
                            transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].text(0.1, 0.5, f'Avg Overlaps/Image: {avg_overlaps:.2f}',
                            transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].text(0.1, 0.4, f'Overlap Rate: {total_overlaps / total_contours * 100:.1f}%'
            if total_contours > 0 else 'Overlap Rate: 0%',
                            transform=axes[1, 1].transAxes, fontsize=12)

            axes[1, 1].set_title('Analysis Summary')
            axes[1, 1].axis('off')

            plt.tight_layout()

            # Save visualization
            viz_path = os.path.join(output_dir, f'{sample_id}_overlap_visualization.png')
            plt.savefig(viz_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Saved overlap visualization: {viz_path}")

        except Exception as e:
            self.logger.error(f"Error creating overlap visualization: {str(e)}")


# Backward compatibility function
def count_overlaps(image_paths, sample_id, output_dir):
    """Wrapper function for backward compatibility."""
    analyzer = OverlapAnalyzer()
    return analyzer.count_overlaps(image_paths, sample_id, output_dir)