import os
import pandas as pd
import logging
from pathlib import Path


def setup_logging():
    """
    Configure logging for generate_labels.py.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("generate_labels.log"),
            logging.StreamHandler()
        ]
    )


def generate_labels_csv(train_dir, test_dir, train_new_dir, test_new_dir, output_file):
    """
    Generate a single labels.csv for tea particle images, handling old and new dataset structures.
    """
    # Define tea variants
    tea_variants = [
        'BOP', 'BOP01', 'BOPA', 'BOPF', 'BOPSP_NEW', 'FBOP', 'FBOPF01_NEW',
        'OP', 'OP_NEW', 'OP01', 'OP01_NEW', 'OPA', 'OPA_NEW', 'PEKOE', 'PEKOE01'
    ]
    elevations = ['High-Medium', 'Low']

    filenames = []
    tea_variant_list = []
    elevation_list = []

    def process_directory(dir_path, dataset_type):
        logging.info(f"Processing {dataset_type} directory: {dir_path}")
        if not os.path.exists(dir_path):
            logging.warning(f"Directory does not exist: {dir_path}")
            return
        for root, _, files in os.walk(dir_path):
            if "PARTICLE" not in root:
                continue
            # Extract tea variant and elevation from path
            parts = root.replace('\\', '/').split('/')
            try:
                elevation_idx = parts.index('High-Medium') if 'High-Medium' in parts else parts.index('Low')
                tea_variant = parts[elevation_idx + 1]
                elevation = parts[elevation_idx]
            except ValueError:
                logging.warning(f"Skipping {root}: Could not determine elevation or tea variant")
                continue
            if tea_variant not in tea_variants:
                logging.warning(f"Skipping {root}: Tea variant {tea_variant} not in {tea_variants}")
                continue
            # Old dataset: PARTICLE/SAMPLE_XX
            if os.path.basename(root).startswith("SAMPLE_"):
                image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
                if not image_files:
                    logging.warning(f"Sample {root} has no valid images")
                    continue
                for f in sorted(image_files):
                    relative_path = os.path.join(root, f).replace(dir_path + os.sep, '').replace('\\', '/')
                    filenames.append(relative_path)
                    tea_variant_list.append(tea_variant)
                    elevation_list.append(elevation)
                logging.info(f"Processed sample {root} with {len(image_files)} images")
            # New dataset: PARTICLE/SAMPLE_XX/TAKE_XX
            elif os.path.basename(root).startswith("TAKE_"):
                image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
                if not image_files:
                    logging.warning(f"Take {root} has no valid images")
                    continue
                for f in sorted(image_files):
                    relative_path = os.path.join(root, f).replace(dir_path + os.sep, '').replace('\\', '/')
                    filenames.append(relative_path)
                    tea_variant_list.append(tea_variant)
                    elevation_list.append(elevation)
                logging.info(f"Processed take {root} with {len(image_files)} images")

    # Setup logging
    setup_logging()
    logging.info("Starting labels.csv generation")

    # Check for existing labels.csv
    if os.path.exists(output_file):
        logging.warning(f"Existing labels.csv found at {output_file}. Overwriting.")

    # Process all directories
    process_directory(train_dir, 'train')
    process_directory(test_dir, 'test')
    process_directory(train_new_dir, 'train_new')
    process_directory(test_new_dir, 'test_new')

    if not filenames:
        logging.error("No valid images found. Check dataset structure and tea_variants list.")
        raise ValueError(
            "No valid images found. Ensure dataset and dataset_New contain images in PARTICLE directories.")

    # Create DataFrame
    df = pd.DataFrame({
        'filename': filenames,
        'tea_variant': tea_variant_list,
        'elevation': elevation_list
    })

    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    logging.info(f"Saved labels.csv to {output_file} with {len(filenames)} entries")


if __name__ == "__main__":
    train_dir = "dataset/train"
    test_dir = "dataset/test"
    train_new_dir = "dataset_New/train"
    test_new_dir = "dataset_New/test"
    output_file = "dataset/labels.csv"
    generate_labels_csv(train_dir, test_dir, train_new_dir, test_new_dir, output_file)