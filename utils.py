import cv2
import numpy as np
import os
import logging
from pathlib import Path


def load_image(image_path, color_mode='grayscale'):
    """
    Load an image from the specified path.

    Args:
        image_path (str): Path to the image file
        color_mode (str): 'grayscale' or 'color'

    Returns:
        numpy.ndarray: Loaded image array
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if color_mode == 'grayscale':
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        return image

    except Exception as e:
        logging.error(f"Error loading image {image_path}: {str(e)}")
        raise


def ensure_dir(directory):
    """
    Ensure that a directory exists, create it if it doesn't.

    Args:
        directory (str): Path to the directory
    """
    try:
        os.makedirs(directory, exist_ok=True)
        logging.debug(f"Directory ensured: {directory}")
    except Exception as e:
        logging.error(f"Error creating directory {directory}: {str(e)}")
        raise


def get_sample_groups(input_dir):
    """
    Group image files by sample ID for multi-image samples.

    Args:
        input_dir (str): Directory containing image files

    Returns:
        list: List of tuples (sample_id, [image_paths])
    """
    try:
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        sample_groups = {}
        supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

        # Walk through directory and collect all image files
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(supported_extensions):
                    file_path = os.path.join(root, file)

                    # Extract sample ID from filename
                    # Assume format like: sample_001_img_1.png, tea_sample_A_1.jpg, etc.
                    base_name = os.path.splitext(file)[0]

                    # Try different patterns to extract sample ID
                    sample_id = extract_sample_id(base_name)

                    if sample_id not in sample_groups:
                        sample_groups[sample_id] = []

                    sample_groups[sample_id].append(file_path)

        # Convert to list of tuples and sort
        result = [(sample_id, sorted(paths)) for sample_id, paths in sample_groups.items()]
        result.sort(key=lambda x: x[0])

        logging.info(f"Found {len(result)} sample groups with {sum(len(paths) for _, paths in result)} total images")

        return result

    except Exception as e:
        logging.error(f"Error grouping samples from {input_dir}: {str(e)}")
        return []


def extract_sample_id(filename):
    """
    Extract sample ID from filename using various patterns.

    Args:
        filename (str): Base filename without extension

    Returns:
        str: Extracted sample ID
    """
    import re

    # Pattern 1: sample_001_img_1 -> sample_001
    match = re.match(r'(.+?)(?:_img_\d+|_image_\d+|_\d+)$', filename)
    if match:
        return match.group(1)

    # Pattern 2: tea_A_high_1 -> tea_A_high
    match = re.match(r'(.+?)_\d+$', filename)
    if match:
        return match.group(1)

    # Pattern 3: IMG_001_sample_A -> sample_A
    match = re.search(r'sample_([A-Za-z0-9_]+)', filename)
    if match:
        return f"sample_{match.group(1)}"

    # Pattern 4: If no pattern matches, use first part before numbers
    match = re.match(r'([A-Za-z_]+)', filename)
    if match:
        return match.group(1)

    # Fallback: use entire filename
    return filename


def save_image(image, output_path, create_dirs=True):
    """
    Save an image to the specified path.

    Args:
        image (numpy.ndarray): Image array to save
        output_path (str): Output file path
        create_dirs (bool): Whether to create directories if they don't exist
    """
    try:
        if create_dirs:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                ensure_dir(output_dir)

        success = cv2.imwrite(output_path, image)
        if not success:
            raise ValueError(f"Failed to save image to {output_path}")

        logging.debug(f"Image saved: {output_path}")

    except Exception as e:
        logging.error(f"Error saving image to {output_path}: {str(e)}")
        raise


def validate_image(image, min_size=(10, 10), max_size=(5000, 5000)):
    """
    Validate an image array.

    Args:
        image (numpy.ndarray): Image array to validate
        min_size (tuple): Minimum (width, height)
        max_size (tuple): Maximum (width, height)

    Returns:
        bool: True if image is valid
    """
    try:
        if image is None:
            return False

        if len(image.shape) not in [2, 3]:
            return False

        height, width = image.shape[:2]

        if width < min_size[0] or height < min_size[1]:
            return False

        if width > max_size[0] or height > max_size[1]:
            return False

        return True

    except Exception:
        return False


def resize_image(image, target_size=None, max_size=None, maintain_aspect=True):
    """
    Resize an image with various options.

    Args:
        image (numpy.ndarray): Input image
        target_size (tuple): Target (width, height), exact resize
        max_size (tuple): Maximum (width, height), resize if larger
        maintain_aspect (bool): Whether to maintain aspect ratio

    Returns:
        numpy.ndarray: Resized image
    """
    try:
        if image is None:
            raise ValueError("Input image is None")

        height, width = image.shape[:2]

        if target_size:
            target_width, target_height = target_size

            if maintain_aspect:
                # Calculate scaling factor
                scale = min(target_width / width, target_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
            else:
                new_width, new_height = target_width, target_height

            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        elif max_size:
            max_width, max_height = max_size

            if width <= max_width and height <= max_height:
                return image  # No resizing needed

            if maintain_aspect:
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
            else:
                new_width = min(width, max_width)
                new_height = min(height, max_height)

            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        else:
            return image  # No resizing

    except Exception as e:
        logging.error(f"Error resizing image: {str(e)}")
        return image


def create_output_filename(input_path, output_dir, suffix="", extension=None):
    """
    Create output filename based on input path.

    Args:
        input_path (str): Input file path
        output_dir (str): Output directory
        suffix (str): Suffix to add to filename
        extension (str): New extension (keep original if None)

    Returns:
        str: Output file path
    """
    try:
        input_file = os.path.basename(input_path)
        name, ext = os.path.splitext(input_file)

        if extension:
            ext = extension if extension.startswith('.') else f'.{extension}'

        output_filename = f"{name}{suffix}{ext}"
        return os.path.join(output_dir, output_filename)

    except Exception as e:
        logging.error(f"Error creating output filename: {str(e)}")
        return os.path.join(output_dir, "output.png")


def get_file_info(file_path):
    """
    Get information about a file.

    Args:
        file_path (str): Path to the file

    Returns:
        dict: File information
    """
    try:
        if not os.path.exists(file_path):
            return {'exists': False}

        stat = os.stat(file_path)
        return {
            'exists': True,
            'size': stat.st_size,
            'modified': stat.st_mtime,
            'basename': os.path.basename(file_path),
            'extension': os.path.splitext(file_path)[1],
            'directory': os.path.dirname(file_path)
        }

    except Exception as e:
        logging.error(f"Error getting file info for {file_path}: {str(e)}")
        return {'exists': False, 'error': str(e)}


def list_image_files(directory, recursive=True):
    """
    List all image files in a directory.

    Args:
        directory (str): Directory to search
        recursive (bool): Whether to search subdirectories

    Returns:
        list: List of image file paths
    """
    try:
        if not os.path.exists(directory):
            return []

        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
        image_files = []

        if recursive:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if os.path.splitext(file.lower())[1] in image_extensions:
                        image_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if (os.path.isfile(file_path) and
                        os.path.splitext(file.lower())[1] in image_extensions):
                    image_files.append(file_path)

        return sorted(image_files)

    except Exception as e:
        logging.error(f"Error listing image files in {directory}: {str(e)}")
        return []


def calculate_image_stats(image):
    """
    Calculate basic statistics for an image.

    Args:
        image (numpy.ndarray): Input image

    Returns:
        dict: Image statistics
    """
    try:
        if image is None:
            return {}

        stats = {
            'height': image.shape[0],
            'width': image.shape[1],
            'channels': len(image.shape),
            'dtype': str(image.dtype),
            'mean': float(np.mean(image)),
            'std': float(np.std(image)),
            'min': float(np.min(image)),
            'max': float(np.max(image))
        }

        if len(image.shape) == 3:
            stats['channels'] = image.shape[2]

        return stats

    except Exception as e:
        logging.error(f"Error calculating image stats: {str(e)}")
        return {}


def create_directory_structure(base_dir, subdirs):
    """
    Create a directory structure.

    Args:
        base_dir (str): Base directory path
        subdirs (list): List of subdirectory names
    """
    try:
        ensure_dir(base_dir)

        for subdir in subdirs:
            subdir_path = os.path.join(base_dir, subdir)
            ensure_dir(subdir_path)

        logging.info(f"Created directory structure in {base_dir}")

    except Exception as e:
        logging.error(f"Error creating directory structure: {str(e)}")
        raise


def cleanup_temp_files(temp_dir, pattern="*"):
    """
    Clean up temporary files.

    Args:
        temp_dir (str): Temporary directory
        pattern (str): File pattern to match
    """
    try:
        if not os.path.exists(temp_dir):
            return

        import glob
        files = glob.glob(os.path.join(temp_dir, pattern))

        for file in files:
            try:
                os.remove(file)
                logging.debug(f"Removed temp file: {file}")
            except Exception as e:
                logging.warning(f"Could not remove temp file {file}: {str(e)}")

        logging.info(f"Cleaned up {len(files)} temporary files")

    except Exception as e:
        logging.error(f"Error cleaning up temp files: {str(e)}")