import cv2
import numpy as np
import os
import logging
from config import Config
import matplotlib.pyplot as plt


class ParticleSegmentation:
    """Advanced tea particle segmentation with multiple techniques."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def enhance_image(self, image):
        """
        Enhance image for better particle visibility without whitescale conversion.
        Maintains original color information while improving contrast.
        """
        try:
            # Apply bilateral filter to reduce noise while preserving edges
            enhanced = cv2.bilateralFilter(
                image,
                Config.BILATERAL_D,
                Config.BILATERAL_SIGMA_COLOR,
                Config.BILATERAL_SIGMA_SPACE
            )

            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(enhanced)

            # Gamma correction for better contrast
            gamma_corrected = np.power(enhanced / 255.0, 1.0 / Config.GAMMA_CORRECTION)
            enhanced = np.uint8(gamma_corrected * 255)

            # Additional contrast stretching
            enhanced = cv2.convertScaleAbs(enhanced, alpha=Config.CONTRAST_ALPHA, beta=Config.CONTRAST_BETA)

            return enhanced

        except Exception as e:
            self.logger.error(f"Error enhancing image: {str(e)}")
            return image

    def remove_background_otsu(self, image):
        """
        Remove background using Otsu's thresholding method.
        """
        try:
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(image, Config.GAUSSIAN_KERNEL, 0)

            # Otsu's thresholding
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return binary

        except Exception as e:
            self.logger.error(f"Error in Otsu thresholding: {str(e)}")
            return image

    def remove_background_adaptive(self, image):
        """
        Remove background using adaptive thresholding.
        """
        try:
            # Apply median filter to reduce salt-and-pepper noise
            filtered = cv2.medianBlur(image, Config.MEDIAN_FILTER_SIZE)

            # Adaptive thresholding
            binary = cv2.adaptiveThreshold(
                filtered,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                Config.ADAPTIVE_BLOCK_SIZE,
                Config.ADAPTIVE_C
            )

            return binary

        except Exception as e:
            self.logger.error(f"Error in adaptive thresholding: {str(e)}")
            return image

    def morphological_operations(self, binary_image):
        """
        Apply morphological operations to refine particle boundaries.
        """
        try:
            # Define kernels for different operations
            kernel_small = np.ones(Config.MORPH_KERNEL_SIZE, np.uint8)
            kernel_large = np.ones((5, 5), np.uint8)

            # Remove noise with opening
            opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_small, iterations=1)

            # Fill gaps with closing
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_large, iterations=2)

            # Remove small objects
            cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_large, iterations=1)

            return cleaned

        except Exception as e:
            self.logger.error(f"Error in morphological operations: {str(e)}")
            return binary_image

    def detect_particles(self, binary_image):
        """
        Detect individual particles using contour detection.
        """
        try:
            # Find contours
            contours, hierarchy = cv2.findContours(
                binary_image,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # Filter contours by area
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if Config.MIN_CONTOUR_AREA <= area <= Config.MAX_CONTOUR_AREA:
                    valid_contours.append(contour)

            self.logger.info(f"Detected {len(valid_contours)} valid particles from {len(contours)} total contours")
            return valid_contours

        except Exception as e:
            self.logger.error(f"Error detecting particles: {str(e)}")
            return []

    def segment_image_fast(self, image):
        """
        Fast segmentation without visualization for efficiency.
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Quick enhancement
            enhanced = self.enhance_image(gray)

            # Quick background removal
            binary = self.remove_background_otsu(enhanced)

            # Quick morphological operations
            refined = self.morphological_operations(binary)

            # Detect particles
            contours = self.detect_particles(refined)

            return contours

        except Exception as e:
            self.logger.error(f"Error in fast segmentation: {str(e)}")
            return []

    def segment_image(self, image, sample_id, output_dir):
        """
        Complete segmentation pipeline for a single image with full visualization.
        """
        try:
            # Convert to grayscale if needed (but don't whitescale)
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Step 1: Enhance image
            enhanced = self.enhance_image(gray)

            # Step 2: Background removal using both methods
            otsu_binary = self.remove_background_otsu(enhanced)
            adaptive_binary = self.remove_background_adaptive(enhanced)

            # Combine both methods
            combined_binary = cv2.bitwise_and(otsu_binary, adaptive_binary)

            # Step 3: Morphological operations
            refined_binary = self.morphological_operations(combined_binary)

            # Step 4: Detect particles
            contours = self.detect_particles(refined_binary)

            # Create visualization
            self.visualize_segmentation_steps(
                gray, enhanced, otsu_binary, adaptive_binary,
                combined_binary, refined_binary, contours,
                sample_id, output_dir
            )

            # Save individual segmented particles
            self.save_individual_particles(gray, contours, sample_id, output_dir)

            # Create detailed segmentation statistics
            self.save_segmentation_statistics(contours, sample_id, output_dir)

            return contours, refined_binary

        except Exception as e:
            self.logger.error(f"Error segmenting image {sample_id}: {str(e)}")
            return [], image

    def visualize_segmentation_steps(self, original, enhanced, otsu, adaptive,
                                     combined, refined, contours, sample_id, output_dir):
        """
        Create comprehensive visualization of segmentation steps.
        """
        try:
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle(f'Segmentation Pipeline - {sample_id}', fontsize=16)

            # Original image
            axes[0, 0].imshow(original, cmap='gray')
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')

            # Enhanced image
            axes[0, 1].imshow(enhanced, cmap='gray')
            axes[0, 1].set_title('Enhanced Image')
            axes[0, 1].axis('off')

            # Otsu thresholding
            axes[0, 2].imshow(otsu, cmap='gray')
            axes[0, 2].set_title('Otsu Thresholding')
            axes[0, 2].axis('off')

            # Adaptive thresholding
            axes[0, 3].imshow(adaptive, cmap='gray')
            axes[0, 3].set_title('Adaptive Thresholding')
            axes[0, 3].axis('off')

            # Combined binary
            axes[1, 0].imshow(combined, cmap='gray')
            axes[1, 0].set_title('Combined Binary')
            axes[1, 0].axis('off')

            # Refined binary
            axes[1, 1].imshow(refined, cmap='gray')
            axes[1, 1].set_title('Morphologically Refined')
            axes[1, 1].axis('off')

            # Detected contours
            contour_image = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_image, contours, -1, Config.COLORS['particles'], 2)
            axes[1, 2].imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
            axes[1, 2].set_title(f'Detected Particles ({len(contours)})')
            axes[1, 2].axis('off')

            # Particle analysis
            self.plot_particle_analysis(axes[1, 3], contours)

            plt.tight_layout()
            output_path = os.path.join(output_dir, f'{sample_id}_segmentation_pipeline.png')
            plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Saved segmentation visualization: {output_path}")

        except Exception as e:
            self.logger.error(f"Error creating segmentation visualization: {str(e)}")

    def plot_particle_analysis(self, ax, contours):
        """
        Plot particle size distribution analysis.
        """
        try:
            if not contours:
                ax.text(0.5, 0.5, 'No particles detected',
                        horizontalalignment='center', verticalalignment='center')
                ax.set_title('Particle Analysis')
                return

            # Calculate particle areas
            areas = [cv2.contourArea(cnt) for cnt in contours]

            # Create histogram
            ax.hist(areas, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('Particle Area (pixels)')
            ax.set_ylabel('Frequency')
            ax.set_title('Particle Size Distribution')
            ax.grid(True, alpha=0.3)

            # Add statistics
            mean_area = np.mean(areas)
            std_area = np.std(areas)
            ax.axvline(mean_area, color='red', linestyle='--',
                       label=f'Mean: {mean_area:.1f}')
            ax.axvline(mean_area + std_area, color='orange', linestyle='--',
                       label=f'+1σ: {mean_area + std_area:.1f}')
            ax.axvline(mean_area - std_area, color='orange', linestyle='--',
                       label=f'-1σ: {mean_area - std_area:.1f}')
            ax.legend(fontsize='small')

        except Exception as e:
            self.logger.error(f"Error plotting particle analysis: {str(e)}")

    def save_individual_particles(self, original_image, contours, sample_id, output_dir):
        """
        Save individual segmented particles as separate images.
        """
        try:
            particles_dir = os.path.join(output_dir, f'{sample_id}_particles')
            os.makedirs(particles_dir, exist_ok=True)

            particle_info = []

            for i, contour in enumerate(contours):
                # Create bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Add padding
                padding = 10
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = min(original_image.shape[1], x + w + padding)
                y_end = min(original_image.shape[0], y + h + padding)

                # Extract particle region
                particle_roi = original_image[y_start:y_end, x_start:x_end]

                # Create mask for the particle
                mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                particle_mask = mask[y_start:y_end, x_start:x_end]

                # Apply mask to get clean particle
                if len(particle_roi.shape) == 2:
                    masked_particle = cv2.bitwise_and(particle_roi, particle_roi, mask=particle_mask)
                else:
                    masked_particle = cv2.bitwise_and(particle_roi, particle_roi, mask=particle_mask)

                # Save particle
                particle_path = os.path.join(particles_dir, f'particle_{i:03d}.png')
                cv2.imwrite(particle_path, masked_particle)

                # Store particle information
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                particle_info.append({
                    'particle_id': i,
                    'area': area,
                    'perimeter': perimeter,
                    'bounding_box': (x, y, w, h),
                    'aspect_ratio': w / h if h > 0 else 0,
                    'file_path': particle_path
                })

            # Save particle information to CSV
            import pandas as pd
            particle_df = pd.DataFrame(particle_info)
            info_path = os.path.join(particles_dir, f'{sample_id}_particle_info.csv')
            particle_df.to_csv(info_path, index=False)

            self.logger.info(f"Saved {len(contours)} individual particles to {particles_dir}")

        except Exception as e:
            self.logger.error(f"Error saving individual particles: {str(e)}")

    def save_segmentation_statistics(self, contours, sample_id, output_dir):
        """
        Save detailed segmentation statistics.
        """
        try:
            if not contours:
                self.logger.warning(f"No contours to analyze for {sample_id}")
                return

            # Calculate comprehensive statistics
            areas = [cv2.contourArea(cnt) for cnt in contours]
            perimeters = [cv2.arcLength(cnt, True) for cnt in contours]

            # Calculate additional metrics
            aspect_ratios = []
            circularities = []

            for contour in contours:
                # Bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                aspect_ratios.append(aspect_ratio)

                # Circularity
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                circularities.append(circularity)

            # Create statistics dictionary
            stats = {
                'sample_id': sample_id,
                'total_particles': len(contours),
                'area_mean': np.mean(areas),
                'area_std': np.std(areas),
                'area_min': np.min(areas),
                'area_max': np.max(areas),
                'perimeter_mean': np.mean(perimeters),
                'perimeter_std': np.std(perimeters),
                'aspect_ratio_mean': np.mean(aspect_ratios),
                'aspect_ratio_std': np.std(aspect_ratios),
                'circularity_mean': np.mean(circularities),
                'circularity_std': np.std(circularities),
                'total_area': np.sum(areas),
                'area_coverage_ratio': np.sum(areas) / (areas[0] if areas else 1)  # Approximate
            }

            # Save statistics to CSV
            import pandas as pd
            stats_df = pd.DataFrame([stats])
            stats_path = os.path.join(output_dir, f'{sample_id}_segmentation_stats.csv')
            stats_df.to_csv(stats_path, index=False)

            # Create detailed visualization
            self.create_detailed_particle_visualization(contours, stats, sample_id, output_dir)

            self.logger.info(f"Saved segmentation statistics: {stats_path}")

        except Exception as e:
            self.logger.error(f"Error saving segmentation statistics: {str(e)}")

    def create_detailed_particle_visualization(self, contours, stats, sample_id, output_dir):
        """
        Create detailed particle analysis visualization.
        """
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Detailed Particle Analysis - {sample_id}', fontsize=16, fontweight='bold')

            # Calculate metrics
            areas = [cv2.contourArea(cnt) for cnt in contours]
            perimeters = [cv2.arcLength(cnt, True) for cnt in contours]
            aspect_ratios = []
            circularities = []

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratios.append(w / h if h > 0 else 0)

                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                circularities.append((4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0)

            # 1. Area distribution
            axes[0, 0].hist(areas, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
            axes[0, 0].set_xlabel('Area (pixels²)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Particle Area Distribution')
            axes[0, 0].axvline(np.mean(areas), color='red', linestyle='--', label=f'Mean: {np.mean(areas):.1f}')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # 2. Perimeter distribution
            axes[0, 1].hist(perimeters, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0, 1].set_xlabel('Perimeter (pixels)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Particle Perimeter Distribution')
            axes[0, 1].axvline(np.mean(perimeters), color='red', linestyle='--',
                               label=f'Mean: {np.mean(perimeters):.1f}')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # 3. Aspect ratio distribution
            axes[0, 2].hist(aspect_ratios, bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[0, 2].set_xlabel('Aspect Ratio')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].set_title('Particle Aspect Ratio Distribution')
            axes[0, 2].axvline(np.mean(aspect_ratios), color='red', linestyle='--',
                               label=f'Mean: {np.mean(aspect_ratios):.2f}')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

            # 4. Circularity distribution
            axes[1, 0].hist(circularities, bins=15, alpha=0.7, color='gold', edgecolor='black')
            axes[1, 0].set_xlabel('Circularity')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Particle Circularity Distribution')
            axes[1, 0].axvline(np.mean(circularities), color='red', linestyle='--',
                               label=f'Mean: {np.mean(circularities):.3f}')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # 5. Area vs Perimeter scatter
            axes[1, 1].scatter(areas, perimeters, alpha=0.6, c='purple')
            axes[1, 1].set_xlabel('Area (pixels²)')
            axes[1, 1].set_ylabel('Perimeter (pixels)')
            axes[1, 1].set_title('Area vs Perimeter Relationship')
            axes[1, 1].grid(True, alpha=0.3)

            # Add trend line
            if len(areas) > 1:
                z = np.polyfit(areas, perimeters, 1)
                p = np.poly1d(z)
                axes[1, 1].plot(areas, p(areas), "r--", alpha=0.8, label=f'Trend: y={z[0]:.3f}x+{z[1]:.1f}')
                axes[1, 1].legend()

            # 6. Statistics summary
            axes[1, 2].axis('off')
            stats_text = f"""Segmentation Statistics:

Total Particles: {stats['total_particles']}

Area Statistics:
  Mean: {stats['area_mean']:.1f} px²
  Std: {stats['area_std']:.1f} px²
  Min: {stats['area_min']:.1f} px²
  Max: {stats['area_max']:.1f} px²

Perimeter Statistics:
  Mean: {stats['perimeter_mean']:.1f} px
  Std: {stats['perimeter_std']:.1f} px

Shape Statistics:
  Aspect Ratio: {stats['aspect_ratio_mean']:.2f} ± {stats['aspect_ratio_std']:.2f}
  Circularity: {stats['circularity_mean']:.3f} ± {stats['circularity_std']:.3f}

Total Area: {stats['total_area']:.1f} px²
"""
            axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes,
                            fontsize=10, verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

            plt.tight_layout()

            # Save detailed visualization
            output_path = os.path.join(output_dir, f'{sample_id}_detailed_particle_analysis.png')
            plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Saved detailed particle visualization: {output_path}")

        except Exception as e:
            self.logger.error(f"Error creating detailed particle visualization: {str(e)}")


def segment_image(image, sample_id=None, output_dir=None):
    """
    Wrapper function for backward compatibility.
    """
    segmenter = ParticleSegmentation()
    if sample_id and output_dir:
        contours, binary = segmenter.segment_image(image, sample_id, output_dir)
        return contours
    else:
        # Simple segmentation without visualization
        return segmenter.segment_image_fast(image)