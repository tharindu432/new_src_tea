import cv2
import numpy as np
import pandas as pd
import os
import logging
from config import Config
from utils import load_image, get_sample_groups
from segmentation import ParticleSegmentation
from overlap_handling import OverlapAnalyzer
from shape_analysis import extract_shape_features


class AdvancedFeatureExtractor:
    """Advanced feature extraction system for tea particle analysis."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.segmenter = ParticleSegmentation()
        self.overlap_analyzer = OverlapAnalyzer()

    def extract_features(self, input_dir, label_file, visualization_dir):
        """
        Extract comprehensive features and labels for all samples in input_dir.
        """
        try:
            # Validate input
            if not os.path.exists(input_dir):
                raise FileNotFoundError(f"Input directory {input_dir} not found")

            if not os.path.exists(label_file):
                raise FileNotFoundError(f"Labels file {label_file} not found")

            # Load labels
            labels_df = pd.read_csv(label_file)
            if labels_df.empty:
                raise ValueError("Labels file is empty")

            self.logger.info(f"Loaded labels with {len(labels_df)} entries")

            # Get sample groups
            sample_groups = get_sample_groups(input_dir)
            if not sample_groups:
                raise ValueError(f"No valid samples found in {input_dir}")

            self.logger.info(f"Found {len(sample_groups)} sample groups in {input_dir}")

            # Initialize output lists
            features = []
            labels = []
            overlap_counts = []
            sample_metadata = []

            # Create output directories
            segmentation_output = os.path.join(visualization_dir, "segmentation_results")
            overlap_output = os.path.join(visualization_dir, "overlap_analysis")
            os.makedirs(segmentation_output, exist_ok=True)
            os.makedirs(overlap_output, exist_ok=True)

            # Process each sample
            processed_count = 0
            for sample_id, image_paths in sample_groups:
                try:
                    self.logger.info(f"Processing sample {sample_id} with {len(image_paths)} images")

                    # Load and process images
                    images = []
                    all_contours = []
                    segmentation_results = []

                    for i, path in enumerate(image_paths):
                        try:
                            img = load_image(path)
                            images.append(img)

                            # Perform segmentation with visualization
                            contours, binary = self.segmenter.segment_image(
                                img, f"{sample_id}_img_{i}", segmentation_output
                            )
                            all_contours.extend(contours)
                            segmentation_results.append((contours, binary))

                        except Exception as e:
                            self.logger.error(f"Error processing image {path}: {str(e)}")
                            continue

                    if not images:
                        self.logger.warning(f"No images successfully loaded for sample {sample_id}")
                        continue

                    # Extract overlap information
                    try:
                        overlap_count = self.overlap_analyzer.count_overlaps(
                            image_paths, sample_id, overlap_output
                        )
                        overlap_counts.append(overlap_count)
                    except Exception as e:
                        self.logger.error(f"Error analyzing overlaps for {sample_id}: {str(e)}")
                        overlap_count = 1
                        overlap_counts.append(overlap_count)

                    # Extract shape features from all contours
                    if all_contours:
                        try:
                            shape_features = extract_shape_features(all_contours)
                            if len(shape_features) > 0:
                                # Aggregate features (mean, std, min, max)
                                feature_mean = np.mean(shape_features, axis=0)
                                feature_std = np.std(shape_features, axis=0)
                                feature_min = np.min(shape_features, axis=0)
                                feature_max = np.max(shape_features, axis=0)

                                # Combine aggregated features
                                aggregated_features = np.concatenate([
                                    feature_mean, feature_std, feature_min, feature_max
                                ])
                            else:
                                # Default feature vector if no valid shape features
                                aggregated_features = np.zeros(76)  # 19 * 4 (mean, std, min, max)
                        except Exception as e:
                            self.logger.error(f"Error extracting shape features for {sample_id}: {str(e)}")
                            aggregated_features = np.zeros(76)
                    else:
                        aggregated_features = np.zeros(76)

                    # Add additional features
                    additional_features = self.extract_additional_features(
                        images, all_contours, overlap_count
                    )

                    # Combine all features
                    sample_features = np.concatenate([
                        aggregated_features,
                        additional_features,
                        [overlap_count]  # Add overlap count as feature
                    ])

                    # Get label for this sample
                    sample_base_id = sample_id.split('_')[0]
                    matching_labels = labels_df[
                        labels_df['filename'].str.contains(sample_base_id, case=False, na=False)
                    ]

                    if matching_labels.empty:
                        self.logger.warning(f"No labels found for sample {sample_id} (base: {sample_base_id})")
                        continue

                    # Use first matching label
                    label_row = matching_labels.iloc[0]
                    tea_variant = label_row['tea_variant']
                    elevation = label_row['elevation']
                    combined_label = f"{tea_variant}_{elevation}"

                    # Store results
                    features.append(sample_features)
                    labels.append(combined_label)

                    # Store metadata
                    sample_metadata.append({
                        'sample_id': sample_id,
                        'tea_variant': tea_variant,
                        'elevation': elevation,
                        'num_images': len(images),
                        'num_contours': len(all_contours),
                        'overlap_count': overlap_count,
                        'total_particle_area': sum(cv2.contourArea(cnt) for cnt in all_contours)
                    })

                    processed_count += 1
                    self.logger.info(f"Successfully processed sample {sample_id} "
                                     f"({processed_count}/{len(sample_groups)})")

                    # Limit processing for visualization samples
                    if processed_count >= Config.VISUALIZE_SAMPLES:
                        self.logger.info(f"Reached visualization limit of {Config.VISUALIZE_SAMPLES} samples")
                        break

                except Exception as e:
                    self.logger.error(f"Error processing sample {sample_id}: {str(e)}")
                    continue

            if not features:
                raise ValueError("No valid features extracted from any samples")

            # Convert to numpy arrays
            features_array = np.array(features)

            # Save feature extraction report
            self.save_feature_report(sample_metadata, visualization_dir)

            # Create feature visualization
            self.create_feature_visualization(features_array, labels, sample_metadata, visualization_dir)

            self.logger.info(f"Feature extraction completed: {len(features)} samples, "
                             f"{features_array.shape[1]} features per sample")

            return features_array, labels, overlap_counts

        except Exception as e:
            self.logger.error(f"Error extracting features from {input_dir}: {str(e)}")
            raise

    def extract_additional_features(self, images, contours, overlap_count):
        """
        Extract additional features beyond shape analysis.
        """
        try:
            additional_features = []

            if not images:
                return np.zeros(15)  # Return default features

            # Image-level features
            combined_image = images[0] if len(images) == 1 else np.mean(images, axis=0).astype(np.uint8)

            # Texture features using Local Binary Pattern
            try:
                from skimage.feature import local_binary_pattern
                lbp = local_binary_pattern(combined_image, P=8, R=1, method='uniform')
                lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 9))
                lbp_features = lbp_hist / np.sum(lbp_hist)  # Normalize
                additional_features.extend(lbp_features[:5])  # Take first 5 bins
            except ImportError:
                # Fallback: simple texture measures
                additional_features.extend([
                    np.std(combined_image),  # Standard deviation as texture measure
                    np.mean(np.gradient(combined_image.astype(float))),  # Average gradient
                    0, 0, 0  # Padding to maintain feature count
                ])

            # Contour-based features
            if contours:
                # Density features
                total_area = sum(cv2.contourArea(cnt) for cnt in contours)
                image_area = combined_image.shape[0] * combined_image.shape[1]
                density_ratio = total_area / image_area

                # Size distribution features
                areas = [cv2.contourArea(cnt) for cnt in contours]
                size_std = np.std(areas) if len(areas) > 1 else 0
                size_range = np.max(areas) - np.min(areas) if areas else 0

                # Shape complexity features
                perimeters = [cv2.arcLength(cnt, True) for cnt in contours]
                avg_complexity = np.mean([p / np.sqrt(a) if a > 0 else 0
                                          for p, a in zip(perimeters, areas)]) if areas else 0

                # Spatial distribution features
                if len(contours) > 1:
                    centroids = []
                    for cnt in contours:
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            cx = M["m10"] / M["m00"]
                            cy = M["m01"] / M["m00"]
                            centroids.append([cx, cy])

                    if len(centroids) > 1:
                        centroids = np.array(centroids)
                        spatial_std = np.mean(np.std(centroids, axis=0))
                    else:
                        spatial_std = 0
                else:
                    spatial_std = 0

                additional_features.extend([
                    density_ratio,
                    size_std,
                    size_range,
                    avg_complexity,
                    spatial_std,
                    len(contours),  # Number of particles
                    total_area,  # Total particle area
                    np.mean(areas) if areas else 0,  # Average particle size
                    overlap_count / len(contours) if contours else 0,  # Overlap ratio
                    len(images)  # Number of images in sample
                ])
            else:
                # Default values when no contours
                additional_features.extend([0] * 10)

            return np.array(additional_features[:15])  # Ensure consistent feature count

        except Exception as e:
            self.logger.error(f"Error extracting additional features: {str(e)}")
            return np.zeros(15)

    def save_feature_report(self, metadata, output_dir):
        """
        Save detailed feature extraction report.
        """
        try:
            if not metadata:
                self.logger.warning("No metadata available for feature report")
                return

            # Convert metadata to DataFrame
            metadata_df = pd.DataFrame(metadata)

            # Save detailed CSV report
            csv_path = os.path.join(output_dir, 'feature_extraction_report.csv')
            metadata_df.to_csv(csv_path, index=False)

            # Create summary report
            report_path = os.path.join(output_dir, 'feature_extraction_summary.txt')

            with open(report_path, 'w') as f:
                f.write("Feature Extraction Summary Report\n")
                f.write("=" * 50 + "\n\n")

                f.write(f"Total Samples Processed: {len(metadata)}\n")
                f.write(f"Total Images Analyzed: {metadata_df['num_images'].sum()}\n")
                f.write(f"Total Contours Detected: {metadata_df['num_contours'].sum()}\n\n")

                f.write("Tea Variant Distribution:\n")
                variant_counts = metadata_df['tea_variant'].value_counts()
                for variant, count in variant_counts.items():
                    f.write(f"  {variant}: {count} samples\n")
                f.write("\n")

                f.write("Elevation Distribution:\n")
                elevation_counts = metadata_df['elevation'].value_counts()
                for elevation, count in elevation_counts.items():
                    f.write(f"  {elevation}: {count} samples\n")
                f.write("\n")

                f.write("Statistics:\n")
                f.write(f"  Average images per sample: {metadata_df['num_images'].mean():.2f}\n")
                f.write(f"  Average contours per sample: {metadata_df['num_contours'].mean():.2f}\n")
                f.write(f"  Average overlap count: {metadata_df['overlap_count'].mean():.2f}\n")
                f.write(f"  Average total particle area: {metadata_df['total_particle_area'].mean():.2f}\n")

                f.write(f"\nFeature Extraction Parameters:\n")
                f.write(f"  Min contour area: {Config.MIN_CONTOUR_AREA}\n")
                f.write(f"  Max contour area: {Config.MAX_CONTOUR_AREA}\n")
                f.write(f"  Visualization samples limit: {Config.VISUALIZE_SAMPLES}\n")

            self.logger.info(f"Saved feature extraction reports: {csv_path}, {report_path}")

        except Exception as e:
            self.logger.error(f"Error saving feature report: {str(e)}")

    def create_feature_visualization(self, features, labels, metadata, output_dir):
        """
        Create comprehensive feature visualization.
        """
        try:
            import matplotlib.pyplot as plt

            # Create feature analysis plots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Feature Extraction Analysis', fontsize=16, fontweight='bold')

            # 1. Feature distribution histogram
            feature_means = np.mean(features, axis=0)
            axes[0, 0].hist(feature_means, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_xlabel('Feature Value')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Distribution of Feature Means')
            axes[0, 0].grid(True, alpha=0.3)

            # 2. Sample distribution by tea variant
            if metadata:
                metadata_df = pd.DataFrame(metadata)
                variant_counts = metadata_df['tea_variant'].value_counts()

                axes[0, 1].pie(variant_counts.values, labels=variant_counts.index,
                               autopct='%1.1f%%', startangle=90)
                axes[0, 1].set_title('Sample Distribution by Tea Variant')

            # 3. Overlap count distribution
            if metadata:
                overlap_counts = metadata_df['overlap_count']
                axes[0, 2].hist(overlap_counts, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
                axes[0, 2].set_xlabel('Overlap Count')
                axes[0, 2].set_ylabel('Frequency')
                axes[0, 2].set_title('Distribution of Overlap Counts')
                axes[0, 2].grid(True, alpha=0.3)

            # 4. Feature correlation heatmap (first 20 features)
            if features.shape[1] > 1:
                n_features_viz = min(20, features.shape[1])
                correlation_matrix = np.corrcoef(features[:, :n_features_viz].T)
                im = axes[1, 0].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
                axes[1, 0].set_title(f'Feature Correlation Matrix (First {n_features_viz})')
                axes[1, 0].set_xlabel('Feature Index')
                axes[1, 0].set_ylabel('Feature Index')
                plt.colorbar(im, ax=axes[1, 0])

            # 5. Contour count vs overlap count scatter
            if metadata:
                contour_counts = metadata_df['num_contours']
                overlap_counts = metadata_df['overlap_count']

                scatter = axes[1, 1].scatter(contour_counts, overlap_counts,
                                             alpha=0.6, c=range(len(contour_counts)), cmap='viridis')
                axes[1, 1].set_xlabel('Number of Contours')
                axes[1, 1].set_ylabel('Overlap Count')
                axes[1, 1].set_title('Contours vs Overlaps Relationship')
                axes[1, 1].grid(True, alpha=0.3)

                # Add trend line
                if len(contour_counts) > 1:
                    z = np.polyfit(contour_counts, overlap_counts, 1)
                    p = np.poly1d(z)
                    axes[1, 1].plot(contour_counts, p(contour_counts), "r--", alpha=0.8)

            # 6. Feature importance (variance-based)
            feature_variances = np.var(features, axis=0)
            top_indices = np.argsort(feature_variances)[-10:]  # Top 10 most variable features

            axes[1, 2].bar(range(len(top_indices)), feature_variances[top_indices],
                           alpha=0.7, color='lightgreen')
            axes[1, 2].set_xlabel('Feature Index (Top 10)')
            axes[1, 2].set_ylabel('Variance')
            axes[1, 2].set_title('Most Variable Features')
            axes[1, 2].set_xticks(range(len(top_indices)))
            axes[1, 2].set_xticklabels([f'F{i}' for i in top_indices], rotation=45)
            axes[1, 2].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save feature visualization
            output_path = os.path.join(output_dir, 'feature_extraction_analysis.png')
            plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
            plt.close()

            # Create additional PCA visualization if possible
            try:
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler

                # Standardize features
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)

                # Apply PCA
                pca = PCA(n_components=2)
                features_pca = pca.fit_transform(features_scaled)

                # Create PCA plot
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                fig.suptitle('Principal Component Analysis', fontsize=16, fontweight='bold')

                # PCA scatter plot
                unique_labels = list(set(labels))
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

                for i, label in enumerate(unique_labels):
                    mask = np.array(labels) == label
                    axes[0].scatter(features_pca[mask, 0], features_pca[mask, 1],
                                    c=[colors[i]], label=label, alpha=0.7)

                axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                axes[0].set_title('PCA Scatter Plot by Tea Class')
                axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                axes[0].grid(True, alpha=0.3)

                # Explained variance plot
                if features.shape[1] > 2:
                    pca_full = PCA()
                    pca_full.fit(features_scaled)
                    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)

                    axes[1].plot(range(1, min(21, len(cumsum_var) + 1)),
                                 cumsum_var[:20], marker='o')
                    axes[1].set_xlabel('Number of Components')
                    axes[1].set_ylabel('Cumulative Explained Variance')
                    axes[1].set_title('PCA Explained Variance')
                    axes[1].grid(True, alpha=0.3)
                    axes[1].axhline(y=0.95, color='r', linestyle='--',
                                    label='95% variance')
                    axes[1].legend()

                plt.tight_layout()

                # Save PCA visualization
                pca_output_path = os.path.join(output_dir, 'feature_pca_analysis.png')
                plt.savefig(pca_output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
                plt.close()

                self.logger.info(f"Saved PCA analysis: {pca_output_path}")

            except ImportError:
                self.logger.warning("sklearn not available for PCA analysis")
            except Exception as e:
                self.logger.error(f"Error creating PCA visualization: {str(e)}")

            self.logger.info(f"Saved feature visualization: {output_path}")

        except Exception as e:
            self.logger.error(f"Error creating feature visualization: {str(e)}")


def extract_features(input_dir, label_file, visualization_dir):
    """
    Wrapper function for backward compatibility.
    """
    extractor = AdvancedFeatureExtractor()
    return extractor.extract_features(input_dir, label_file, visualization_dir)