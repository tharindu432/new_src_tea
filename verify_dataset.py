import os
import pandas as pd
import logging
from pathlib import Path
from config import Config
from utils import get_sample_groups


def setup_logging():
    """
    Configure logging for verify_dataset.py.
    """
    # Ensure logs directory exists
    log_dir = os.path.dirname("verify_dataset.log")
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("verify_dataset.log", mode='w'),  # Overwrite previous log
            logging.StreamHandler()
        ]
    )


def check_directory_structure(directory, dataset_name):
    """
    Check the structure of a dataset directory.
    """
    logging.info(f"=" * 60)
    logging.info(f"Checking {dataset_name}: {directory}")
    logging.info(f"=" * 60)

    if not os.path.exists(directory):
        logging.error(f"Directory does not exist: {directory}")
        return False, {}

    if not os.listdir(directory):
        logging.error(f"Directory is empty: {directory}")
        return False, {}

    structure_info = {
        'total_images': 0,
        'total_samples': 0,
        'elevations': set(),
        'tea_variants': set(),
        'samples_by_variant': {},
        'images_by_variant': {},
        'directory_structure': []
    }

    try:
        # Walk through directory structure
        for root, dirs, files in os.walk(directory):
            rel_path = os.path.relpath(root, directory)
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

            if image_files:
                structure_info['directory_structure'].append({
                    'path': rel_path,
                    'image_count': len(image_files),
                    'images': image_files[:3]  # Show first 3 images as sample
                })
                structure_info['total_images'] += len(image_files)

                # Extract elevation and tea variant from path
                path_parts = rel_path.replace('\\', '/').split('/')

                # Look for elevation indicators
                elevation = None
                tea_variant = None

                for part in path_parts:
                    if part in ['High-Medium', 'Low', 'High', 'Medium']:
                        elevation = part
                        structure_info['elevations'].add(part)
                    elif part in ['BOP', 'BOP01', 'BOPA', 'BOPF', 'BOPSP_NEW', 'FBOP', 'FBOPF01_NEW',
                                  'OP', 'OP_NEW', 'OP01', 'OP01_NEW', 'OPA', 'OPA_NEW', 'PEKOE', 'PEKOE01']:
                        tea_variant = part
                        structure_info['tea_variants'].add(part)

                if tea_variant:
                    if tea_variant not in structure_info['samples_by_variant']:
                        structure_info['samples_by_variant'][tea_variant] = 0
                        structure_info['images_by_variant'][tea_variant] = 0

                    if 'SAMPLE_' in rel_path or 'TAKE_' in rel_path:
                        structure_info['samples_by_variant'][tea_variant] += 1
                    structure_info['images_by_variant'][tea_variant] += len(image_files)

        # Get sample groups using utils function
        try:
            sample_groups = get_sample_groups(directory)
            structure_info['total_samples'] = len(sample_groups)

            logging.info(f"Sample groups found: {len(sample_groups)}")
            for i, (sample_id, image_paths) in enumerate(sample_groups[:5]):  # Show first 5 samples
                logging.info(f"  Sample {i + 1}: {sample_id} ({len(image_paths)} images)")

        except Exception as e:
            logging.warning(f"Error getting sample groups: {str(e)}")
            structure_info['total_samples'] = 0

        # Log summary
        logging.info(f"Summary for {dataset_name}:")
        logging.info(f"  Total images: {structure_info['total_images']}")
        logging.info(f"  Total samples: {structure_info['total_samples']}")
        logging.info(
            f"  Elevations found: {sorted(structure_info['elevations']) if structure_info['elevations'] else 'None'}")
        logging.info(
            f"  Tea variants found: {sorted(structure_info['tea_variants']) if structure_info['tea_variants'] else 'None'}")

        if structure_info['samples_by_variant']:
            logging.info("  Samples by tea variant:")
            for variant, count in sorted(structure_info['samples_by_variant'].items()):
                img_count = structure_info['images_by_variant'].get(variant, 0)
                logging.info(f"    {variant}: {count} samples, {img_count} images")

        # Show directory structure (first 10 entries)
        if structure_info['directory_structure']:
            logging.info("  Directory structure (sample):")
            for entry in structure_info['directory_structure'][:10]:
                logging.info(f"    {entry['path']}: {entry['image_count']} images")
            if len(structure_info['directory_structure']) > 10:
                logging.info(f"    ... and {len(structure_info['directory_structure']) - 10} more directories")

        return True, structure_info

    except Exception as e:
        logging.error(f"Error analyzing directory structure: {str(e)}")
        return False, {}


def verify_labels_file():
    """
    Verify the labels.csv file structure and content.
    """
    logging.info(f"=" * 60)
    logging.info(f"Verifying labels file: {Config.LABELS_FILE}")
    logging.info(f"=" * 60)

    if not os.path.exists(Config.LABELS_FILE):
        logging.error(f"Labels file not found: {Config.LABELS_FILE}")
        return False, None

    try:
        # Check file size
        file_size = os.path.getsize(Config.LABELS_FILE)
        if file_size == 0:
            logging.error(f"Labels file is empty: {Config.LABELS_FILE}")
            return False, None

        logging.info(f"Labels file size: {file_size} bytes")

        # Load and analyze labels
        labels_df = pd.read_csv(Config.LABELS_FILE)

        if labels_df.empty:
            logging.error("Labels DataFrame is empty")
            return False, None

        logging.info(f"Labels loaded successfully: {len(labels_df)} entries")

        # Check required columns
        required_columns = ['filename', 'tea_variant', 'elevation']
        missing_columns = [col for col in required_columns if col not in labels_df.columns]

        if missing_columns:
            logging.error(f"Missing required columns in labels.csv: {missing_columns}")
            logging.info(f"Available columns: {list(labels_df.columns)}")
            return False, None

        # Analyze content
        logging.info("Labels file analysis:")
        logging.info(f"  Columns: {list(labels_df.columns)}")
        logging.info(f"  Tea variants: {sorted(labels_df['tea_variant'].unique())}")
        logging.info(f"  Elevations: {sorted(labels_df['elevation'].unique())}")

        # Check for duplicates
        duplicates = labels_df[labels_df.duplicated(subset=['filename'])]
        if not duplicates.empty:
            logging.warning(f"Found {len(duplicates)} duplicate filenames in labels.csv")
            logging.info("Sample duplicates:")
            for _, row in duplicates.head().iterrows():
                logging.info(f"  {row['filename']}")

        # Check for missing values
        missing_values = labels_df.isnull().sum()
        if missing_values.any():
            logging.warning("Missing values found:")
            for col, count in missing_values.items():
                if count > 0:
                    logging.warning(f"  {col}: {count} missing values")

        # Show sample entries
        logging.info("Sample entries from labels.csv:")
        for i, (_, row) in enumerate(labels_df.head().iterrows()):
            logging.info(f"  {i + 1}: {row['filename']} -> {row['tea_variant']}_{row['elevation']}")

        return True, labels_df

    except Exception as e:
        logging.error(f"Error reading labels file: {str(e)}")
        return False, None


def cross_reference_labels_and_datasets(labels_df, dataset_structures):
    """
    Cross-reference labels with actual dataset files.
    """
    logging.info(f"=" * 60)
    logging.info("Cross-referencing labels with datasets")
    logging.info(f"=" * 60)

    if labels_df is None:
        logging.error("No labels data available for cross-reference")
        return

    # Collect all actual image files from datasets
    actual_files = set()
    dataset_file_mapping = {}  # Maps relative path to full path

    for dataset_name, directory in dataset_structures.items():
        logging.info(f"Scanning {dataset_name} dataset...")

        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, directory).replace('\\', '/')
                    actual_files.add(rel_path)
                    dataset_file_mapping[rel_path] = full_path

    logging.info(f"Found {len(actual_files)} actual image files across all datasets")

    # Get labeled files
    labeled_files = set(labels_df['filename'].str.replace('\\', '/'))
    logging.info(f"Found {len(labeled_files)} entries in labels.csv")

    # Find mismatches
    missing_in_labels = actual_files - labeled_files
    missing_in_datasets = labeled_files - actual_files
    matching_files = actual_files & labeled_files

    logging.info(f"Cross-reference results:")
    logging.info(f"  Files in both labels and datasets: {len(matching_files)}")
    logging.info(f"  Files in datasets but missing in labels: {len(missing_in_labels)}")
    logging.info(f"  Files in labels but missing in datasets: {len(missing_in_datasets)}")

    # Show some examples of missing files
    if missing_in_labels:
        logging.warning("Sample files missing from labels.csv:")
        for file in sorted(missing_in_labels)[:10]:
            logging.warning(f"  {file}")
        if len(missing_in_labels) > 10:
            logging.warning(f"  ... and {len(missing_in_labels) - 10} more")

    if missing_in_datasets:
        logging.warning("Sample files in labels.csv but missing from datasets:")
        for file in sorted(missing_in_datasets)[:10]:
            logging.warning(f"  {file}")
        if len(missing_in_datasets) > 10:
            logging.warning(f"  ... and {len(missing_in_datasets) - 10} more")

    # Calculate match percentage
    if len(labeled_files) > 0:
        match_percentage = (len(matching_files) / len(labeled_files)) * 100
        logging.info(f"Label-to-dataset match rate: {match_percentage:.1f}%")

    return {
        'matching_files': len(matching_files),
        'missing_in_labels': len(missing_in_labels),
        'missing_in_datasets': len(missing_in_datasets),
        'match_percentage': match_percentage if len(labeled_files) > 0 else 0
    }


def verify_dataset():
    """
    Main function to verify dataset structure and labels.csv consistency.
    """
    setup_logging()
    logging.info("Starting comprehensive dataset verification")
    logging.info(f"Project root: {Config.PROJECT_ROOT}")

    # Check dataset directories
    datasets_info = {}
    directories = {
        'train': Config.TRAIN_DIR,
        'test': Config.TEST_DIR,
        'train_new': Config.TRAIN_NEW_DIR,
        'test_new': Config.TEST_NEW_DIR
    }

    available_datasets = {}
    for name, directory in directories.items():
        success, structure_info = check_directory_structure(directory, name)
        if success:
            available_datasets[name] = directory
            datasets_info[name] = structure_info

    if not available_datasets:
        logging.error("No valid datasets found! Please check your dataset directories.")
        return False

    logging.info(f"Found {len(available_datasets)} valid datasets: {list(available_datasets.keys())}")

    # Verify labels file
    labels_success, labels_df = verify_labels_file()

    # Cross-reference if both datasets and labels are available
    if labels_success and available_datasets:
        cross_reference_results = cross_reference_labels_and_datasets(labels_df, available_datasets)
    else:
        cross_reference_results = None

    # Final summary
    logging.info(f"=" * 60)
    logging.info("VERIFICATION SUMMARY")
    logging.info(f"=" * 60)

    logging.info(f"Datasets status:")
    for name, directory in directories.items():
        if name in available_datasets:
            info = datasets_info[name]
            logging.info(f"  ✓ {name}: {info['total_images']} images, {info['total_samples']} samples")
        else:
            logging.info(f"  ✗ {name}: Not available")

    logging.info(f"Labels file status:")
    if labels_success:
        logging.info(f"  ✓ labels.csv: {len(labels_df)} entries")
    else:
        logging.info(f"  ✗ labels.csv: Not available or invalid")

    if cross_reference_results:
        logging.info(f"Cross-reference status:")
        logging.info(f"  Match rate: {cross_reference_results['match_percentage']:.1f}%")
        logging.info(f"  Matching files: {cross_reference_results['matching_files']}")
        logging.info(f"  Missing in labels: {cross_reference_results['missing_in_labels']}")
        logging.info(f"  Missing in datasets: {cross_reference_results['missing_in_datasets']}")

    # Determine overall status
    overall_success = (len(available_datasets) > 0 and labels_success and
                       (cross_reference_results is None or cross_reference_results['match_percentage'] > 50))

    if overall_success:
        logging.info("✓ Dataset verification PASSED - Ready for processing")
    else:
        logging.warning("✗ Dataset verification FAILED - Issues need to be resolved")

    logging.info("Verification complete. Check verify_dataset.log for detailed results.")
    return overall_success


if __name__ == "__main__":
    success = verify_dataset()
    exit(0 if success else 1)