import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import logging
from config import Config
from utils import load_image
from segmentation import ParticleSegmentation


class OverlapAnalyzer:
    """Advanced overlap detection and analysis for tea particles."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.segmenter = ParticleSegmentation()

    def detect_histogram_peaks(self, image):
        """
        Detect intensity peaks in image histogram for overlap estimation.
        """
        try:
            # Calculate histogram
            hist, bins = np.histogram(image.ravel(), bins=Config.HISTOGRAM_BINS, range=(0, 255))

            # Smooth histogram to reduce noise
            from scipy.ndimage import gaussian_filter1d
            smoothed_hist = gaussian_filter1d(hist, sigma=2)

            # Find peaks
            peaks, properties = find_peaks(
                smoothed_hist,
                height=Config.MIN_PEAK_HEIGHT,
                distance=Config.MIN_PEAK_DISTANCE,
                prominence=20
            )

            return peaks, smoothed_hist, properties

        except Exception as e:
            self.logger.error(f"Error detecting histogram peaks: {str(e)}")
            return [], [], {}

    def cluster_contours_kmeans(self, contours):
        """
        Cluster contours using K-Means to identify overlapping regions.
        """
        try:
            if not contours or len(contours) < 2:
                return []

            # Extract contour features
            features = []
            valid_contours = []

            for contour in contours:
                # Calculate moments
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)

                    # Avoid division by zero
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                    else:
                        circularity = 0

                    features.append([cx, cy, area, perimeter, circularity])
                    valid_contours.append(contour)

            if len(features) < 2:
                return []

            # Normalize features
            scaler = StandardScaler()
            features_normalized = scaler.fit_transform(features)

            # Determine optimal number of clusters
            max_clusters = min(Config.MAX_CLUSTERS, len(features))

            # Apply K-Means clustering
            kmeans = KMeans(n_clusters=max_clusters, random_state=Config.RANDOM_STATE, n_init=10)
            cluster_labels = kmeans.fit_predict(features_normalized)

            # Group contours by cluster
            clustered_contours = []
            for i in range(max_clusters):
                cluster_contours = [valid_contours[j] for j in range(len(valid_contours)) if cluster_labels[j] == i]
                if len(cluster_contours) >= Config.MIN_CLUSTER_SIZE:
                    clustered_contours.append(cluster_contours)

            return clustered_contours

        except Exception as e:
            self.logger.error(f"Error clustering contours with K-Means: {str(e)}")
            return []

    def cluster_contours_dbscan(self, contours):
        """
        Alternative clustering using DBSCAN for density-based overlap detection.
        """
        try:
            if not contours or len(contours) < 2:
                return []

            # Extract spatial features
            centers = []
            valid_contours = []

            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centers.append([cx, cy])
                    valid_contours.append(contour)

            if len(centers) < 2:
                return []

            # Apply DBSCAN clustering
            dbscan = DBSCAN(eps=50, min_samples=2)
            cluster_labels = dbscan.fit_predict(centers)

            # Group contours by cluster (excluding noise points labeled as -1)
            clustered_contours = []
            unique_labels = set(cluster_labels) - {-1}

            for label in unique_labels:
                cluster_contours = [valid_contours[i] for i in range(len(valid_contours)) if cluster_labels[i] == label]
                if len(cluster_contours) >= 2:  # At least 2 contours for overlap
                    clustered_contours.append(cluster_contours)

            return clustered_contours

        except Exception as e:
            self.logger.error(f"Error clustering contours with DBSCAN: {str(e)}")
            return []

    def match_contours_across_images(self, contours_list):
        """
        Match contours across multiple images to validate overlaps.
        """
        try:
            if len(contours_list) < 2:
                return []

            matches = []
            base_contours = contours_list[0]

            for base_contour in base_contours:
                matched_group = [base_contour]

                for other_contours in contours_list[1:]:
                    best_match = None
                    min_distance = float('inf')

                    for other_contour in other_contours:
                        # Use Hu moments for shape matching
                        try:
                            distance = cv2.matchShapes(base_contour, other_contour, cv2.CONTOURS_MATCH_I1, 0)
                            if distance < min_distance and distance < Config.MATCH_SHAPE_THRESHOLD:
                                min_distance = distance
                                best_match = other_contour
                        except:
                            continue

                    if best_match is not None:
                        matched_group.append(best_match)

                # Consider it a match if found in at least 2 images
                if len(matched_group) >= 2:
                    matches.append(matched_group)

            return matches

        except Exception as e:
            self.logger.error(f"Error matching contours across images: {str(e)}")
            return []

    def analyze_overlap_density(self, image, contours):
        """
        Analyze local density to identify overlapping regions.
        """
        try:
            if not contours:
                return []

            # Create density map
            density_map = np.zeros(image.shape[:2], dtype=np.float32)

            for contour in contours:
                # Create mask for current contour
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 1, -1)

                # Add to density map
                density_map += mask.astype(np.float32)

            # Find high-density regions (overlaps)
            overlap_regions = []
            high_density_mask = density_map > 1.5  # Threshold for overlap detection

            # Find contours in high-density regions
            high_density_contours, _ = cv2.findContours(
                high_density_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in high_density_contours:
                if cv2.contourArea(contour) > Config.MIN_CONTOUR_AREA:
                    overlap_regions.append(contour)

            return overlap_regions, density_map

        except Exception as e:
            self.logger.error(f"Error analyzing overlap density: {str(e)}")
            return [], np.zeros(image.shape[:2])

    def count_overlaps(self, image_paths, sample_id, output_dir):
        """
        Comprehensive overlap counting using multiple methods.
        """
        try:
            # Load and process images
            images = []
            contours_list = []

            for path in image_paths:
                img = load_image(path)
                images.append(img)

                # Segment image
                contours = self.segmenter.segment_image(img, f"{sample_id}_{len(images)}", output_dir)[0]
                contours_list.append(contours)

            # Method 1: Histogram peak analysis
            peak_counts = []
            for img in images:
                peaks, hist, properties = self.detect_histogram_peaks(img)
                peak_counts.append(len(peaks))

            histogram_overlap_estimate = max(1, int(np.mean(peak_counts))) if peak_counts else 1

            # Method 2: K-Means clustering
            all_contours = [cnt for contours in contours_list for cnt in contours]
            kmeans_clusters = self.cluster_contours_kmeans(all_contours)
            kmeans_overlap_count = len(kmeans_clusters)

            # Method 3: DBSCAN clustering
            dbscan_clusters = self.cluster_contours_dbscan(all_contours)
            dbscan_overlap_count = len(dbscan_clusters)

            # Method 4: Cross-image matching
            cross_matches = self.match_contours_across_images(contours_list)
            cross_match_count = len(cross_matches)

            # Method 5: Density analysis
            if images:
                overlap_regions, density_map = self.analyze_overlap_density(images[0], all_contours)
                density_overlap_count = len(overlap_regions)
            else:
                density_overlap_count = 0
                density_map = np.zeros((100, 100))

            # Combine results using weighted average
            overlap_estimates = [
                histogram_overlap_estimate * 0.2,
                kmeans_overlap_count * 0.3,
                dbscan_overlap_count * 0.2,
                cross_match_count * 0.2,
                density_overlap_count * 0.1
            ]

            final_overlap_count = max(1, int(np.sum(overlap_estimates)))

            # Create comprehensive visualization
            self.visualize_overlap_analysis(
                images, contours_list, all_contours,
                kmeans_clusters, dbscan_clusters, cross_matches,
                density_map, overlap_regions, sample_id, output_dir,
                {
                    'histogram': histogram_overlap_estimate,
                    'kmeans': kmeans_overlap_count,
                    'dbscan': dbscan_overlap_count,
                    'cross_match': cross_match_count,
                    'density': density_overlap_count,
                    'final': final_overlap_count
                }
            )

            # Save detailed overlap report
            self.save_overlap_report(sample_id, output_dir, {
                'histogram_peaks': histogram_overlap_estimate,
                'kmeans_clusters': kmeans_overlap_count,
                'dbscan_clusters': dbscan_overlap_count,
                'cross_matches': cross_match_count,
                'density_regions': density_overlap_count,
                'final_count': final_overlap_count,
                'total_contours': len(all_contours),
                'images_analyzed': len(images)
            })

            return final_overlap_count

        except Exception as e:
            self.logger.error(f"Error counting overlaps for {sample_id}: {str(e)}")
            return 1

    def visualize_overlap_analysis(self, images, contours_list, all_contours,
                                   kmeans_clusters, dbscan_clusters, cross_matches,
                                   density_map, overlap_regions, sample_id, output_dir, counts):
        """
        Create comprehensive overlap analysis visualization.
        """
        try:
            fig, axes = plt.subplots(3, 4, figsize=(20, 15))
            fig.suptitle(f'Overlap Analysis - {sample_id}', fontsize=16)

            # Row 1: Original images and all contours
            for i, (img, contours) in enumerate(zip(images[:3], contours_list[:3])):
                if i < len(images):
                    # Original image with contours
                    img_with_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    cv2.drawContours(img_with_contours, contours, -1, Config.COLORS['particles'], 2)
                    axes[0, i].imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
                    axes[0, i].set_title(f'Image {i + 1} ({len(contours)} particles)')
                    axes[0, i].axis('off')
                else:
                    axes[0, i].axis('off')

            # All contours combined
            if images:
                combined_img = cv2.cvtColor(images[0], cv2.COLOR_GRAY2BGR)
                cv2.drawContours(combined_img, all_contours, -1, Config.COLORS['particles'], 1)
                axes[0, 3].imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
                axes[0, 3].set_title(f'All Contours ({len(all_contours)})')
                axes[0, 3].axis('off')

            # Row 2: Clustering results
            # K-Means clusters
            if images and kmeans_clusters:
                kmeans_img = cv2.cvtColor(images[0], cv2.COLOR_GRAY2BGR)
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                for i, cluster in enumerate(kmeans_clusters):
                    color = colors[i % len(colors)]
                    cv2.drawContours(kmeans_img, cluster, -1, color, 2)
                axes[1, 0].imshow(cv2.cvtColor(kmeans_img, cv2.COLOR_BGR2RGB))
                axes[1, 0].set_title(f'K-Means Clusters ({counts["kmeans"]})')
                axes[1, 0].axis('off')
            else:
                axes[1, 0].text(0.5, 0.5, 'No K-Means clusters', ha='center', va='center')
                axes[1, 0].set_title('K-Means Clusters (0)')
                axes[1, 0].axis('off')

            # DBSCAN clusters
            if images and dbscan_clusters:
                dbscan_img = cv2.cvtColor(images[0], cv2.COLOR_GRAY2BGR)
                colors = [(255, 100, 0), (100, 255, 0), (0, 100, 255), (255, 0, 100), (100, 0, 255)]
                for i, cluster in enumerate(dbscan_clusters):
                    color = colors[i % len(colors)]
                    cv2.drawContours(dbscan_img, cluster, -1, color, 2)
                axes[1, 1].imshow(cv2.cvtColor(dbscan_img, cv2.COLOR_BGR2RGB))
                axes[1, 1].set_title(f'DBSCAN Clusters ({counts["dbscan"]})')
                axes[1, 1].axis('off')
            else:
                axes[1, 1].text(0.5, 0.5, 'No DBSCAN clusters', ha='center', va='center')
                axes[1, 1].set_title('DBSCAN Clusters (0)')
                axes[1, 1].axis('off')

            # Cross-image matches
            if images and cross_matches:
                match_img = cv2.cvtColor(images[0], cv2.COLOR_GRAY2BGR)
                for i, match_group in enumerate(cross_matches):
                    color = (0, 255, 255)  # Cyan for matches
                    cv2.drawContours(match_img, [match_group[0]], -1, color, 3)
                axes[1, 2].imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
                axes[1, 2].set_title(f'Cross-Image Matches ({counts["cross_match"]})')
                axes[1, 2].axis('off')
            else:
                axes[1, 2].text(0.5, 0.5, 'No cross matches', ha='center', va='center')
                axes[1, 2].set_title('Cross-Image Matches (0)')
                axes[1, 2].axis('off')

            # Density map
            if density_map.size > 0:
                im = axes[1, 3].imshow(density_map, cmap='hot', interpolation='nearest')
                axes[1, 3].set_title(f'Density Map ({counts["density"]} regions)')
                plt.colorbar(im, ax=axes[1, 3])
            else:
                axes[1, 3].text(0.5, 0.5, 'No density map', ha='center', va='center')
                axes[1, 3].set_title('Density Map (0)')
            axes[1, 3].axis('off')

            # Row 3: Analysis and statistics
            # Histogram analysis
            if images:
                hist, bins = np.histogram(images[0].ravel(), bins=50, range=(0, 255))
                axes[2, 0].plot(bins[:-1], hist)
                axes[2, 0].set_title(f'Intensity Histogram\n({counts["histogram"]} peaks)')
                axes[2, 0].set_xlabel('Intensity')
                axes[2, 0].set_ylabel('Frequency')
                axes[2, 0].grid(True, alpha=0.3)

            # Overlap method comparison
            methods = ['Histogram', 'K-Means', 'DBSCAN', 'Cross-Match', 'Density']
            values = [counts['histogram'], counts['kmeans'], counts['dbscan'],
                      counts['cross_match'], counts['density']]

            bars = axes[2, 1].bar(methods, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
            axes[2, 1].set_title('Overlap Detection Methods')
            axes[2, 1].set_ylabel('Detected Overlaps')
            axes[2, 1].tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[2, 1].text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                                f'{value}', ha='center', va='bottom')

            # Final result summary
            axes[2, 2].text(0.1, 0.8, f'Final Overlap Count: {counts["final"]}',
                            fontsize=14, fontweight='bold', transform=axes[2, 2].transAxes)
            axes[2, 2].text(0.1, 0.6, f'Total Contours: {len(all_contours)}',
                            fontsize=12, transform=axes[2, 2].transAxes)
            axes[2, 2].text(0.1, 0.4, f'Images Analyzed: {len(images)}',
                            fontsize=12, transform=axes[2, 2].transAxes)
            axes[2, 2].text(0.1, 0.2, f'Average per Image: {len(all_contours) / max(1, len(images)):.1f}',
                            fontsize=12, transform=axes[2, 2].transAxes)
            axes[2, 2].set_title('Analysis Summary')
            axes[2, 2].axis('off')

            # Particle size distribution
            if all_contours:
                areas = [cv2.contourArea(cnt) for cnt in all_contours]
                axes[2, 3].hist(areas, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
                axes[2, 3].set_xlabel('Particle Area (pixels)')
                axes[2, 3].set_ylabel('Frequency')
                axes[2, 3].set_title('Particle Size Distribution')
                axes[2, 3].grid(True, alpha=0.3)

                # Add mean line
                mean_area = np.mean(areas)
                axes[2, 3].axvline(mean_area, color='red', linestyle='--',
                                   label=f'Mean: {mean_area:.0f}')
                axes[2, 3].legend()

            plt.tight_layout()

            # Save visualization
            output_path = os.path.join(output_dir, f'{sample_id}_overlap_analysis.png')
            plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Saved overlap analysis visualization: {output_path}")

        except Exception as e:
            self.logger.error(f"Error creating overlap visualization: {str(e)}")

    def save_overlap_report(self, sample_id, output_dir, results):
        """
        Save detailed overlap analysis report.
        """
        try:
            report_path = os.path.join(output_dir, f'{sample_id}_overlap_report.txt')

            with open(report_path, 'w') as f:
                f.write(f"Overlap Analysis Report - {sample_id}\n")
                f.write("=" * 50 + "\n\n")

                f.write("Detection Method Results:\n")
                f.write(f"- Histogram Peaks: {results['histogram_peaks']}\n")
                f.write(f"- K-Means Clusters: {results['kmeans_clusters']}\n")
                f.write(f"- DBSCAN Clusters: {results['dbscan_clusters']}\n")
                f.write(f"- Cross-Image Matches: {results['cross_matches']}\n")
                f.write(f"- Density Regions: {results['density_regions']}\n\n")

                f.write("Summary:\n")
                f.write(f"- Final Overlap Count: {results['final_count']}\n")
                f.write(f"- Total Contours Detected: {results['total_contours']}\n")
                f.write(f"- Images Analyzed: {results['images_analyzed']}\n")
                f.write(
                    f"- Average Contours per Image: {results['total_contours'] / max(1, results['images_analyzed']):.2f}\n")

                f.write(f"\nAnalysis Parameters:\n")
                f.write(f"- Minimum Contour Area: {Config.MIN_CONTOUR_AREA}\n")
                f.write(f"- Maximum Contour Area: {Config.MAX_CONTOUR_AREA}\n")
                f.write(f"- Shape Match Threshold: {Config.MATCH_SHAPE_THRESHOLD}\n")
                f.write(f"- Minimum Cluster Size: {Config.MIN_CLUSTER_SIZE}\n")

            self.logger.info(f"Saved overlap report: {report_path}")

        except Exception as e:
            self.logger.error(f"Error saving overlap report: {str(e)}")


def count_overlaps(image_paths, visualization_dir, sample_id):
    """
    Wrapper function for backward compatibility.
    """
    analyzer = OverlapAnalyzer()
    return analyzer.count_overlaps(image_paths, sample_id, visualization_dir)