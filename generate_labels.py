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
    dataset_type_list = []

    def process_directory(dir_path, dataset_type):
        """Process a single directory and extract image information."""
        logging.info(f"Processing {dataset_type} directory: {dir_path}")

        if not os.path.exists(dir_path):
            logging.warning(f"Directory does not exist: {dir_path}")
            return 0

        processed_count = 0

        for root, _, files in os.walk(dir_path):
            # Skip non-PARTICLE directories
            if "PARTICLE" not in root:
                continue

            # Extract tea variant and elevation from path
            parts = root.replace('\\', '/').split('/')

            try:
                # Find elevation in path
                elevation = None
                tea_variant = None

                for i, part in enumerate(parts):
                    if part in elevations:
                        elevation = part
                        # Tea variant should be the next directory after elevation
                        if i + 1 < len(parts):
                            tea_variant = parts[i + 1]
                        break

                if not elevation or not tea_variant:
                    logging.warning(f"Skipping {root}: Could not determine elevation or tea variant from path")
                    continue

                if tea_variant not in tea_variants:
                    logging.warning(f"Skipping {root}: Tea variant '{tea_variant}' not in predefined list")
                    continue

            except Exception as e:
                logging.warning(f"Skipping {root}: Error parsing path - {str(e)}")
                continue

            # Process images in this directory
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if not image_files:
                logging.warning(f"No valid images found in {root}")
                continue

            # Determine if this is old dataset (PARTICLE/SAMPLE_XX) or new dataset (PARTICLE/SAMPLE_XX/TAKE_XX)
            is_sample_dir = os.path.basename(root).startswith("SAMPLE_")
            is_take_dir = os.path.basename(root).startswith("TAKE_")

            if is_sample_dir or is_take_dir:
                for filename in sorted(image_files):
                    try:
                        # Create relative path from dataset directory
                        full_path = os.path.join(root, filename)
                        relative_path = os.path.relpath(full_path, dir_path).replace('\\', '/')

                        filenames.append(relative_path)
                        tea_variant_list.append(tea_variant)
                        elevation_list.append(elevation)
                        dataset_type_list.append(dataset_type)
                        processed_count += 1

                    except Exception as e:
                        logging.warning(f"Error processing file {filename} in {root}: {str(e)}")
                        continue

                logging.info(f"Processed {'sample' if is_sample_dir else 'take'} {root}: {len(image_files)} images")
            else:
                logging.warning(f"Skipping {root}: Not recognized as SAMPLE_ or TAKE_ directory")

        return processed_count

    # Setup logging
    setup_logging()
    logging.info("Starting labels.csv generation")

    # Check for existing labels.csv
    if os.path.exists(output_file):
        logging.warning(f"Existing labels.csv found at {output_file}. Will overwrite.")

    # Process all directories
    total_processed = 0
    directories = [
        (train_dir, 'train'),
        (test_dir, 'test'),
        (train_new_dir, 'train_new'),
        (test_new_dir, 'test_new')
    ]

    for dir_path, dir_type in directories:
        count = process_directory(dir_path, dir_type)
        total_processed += count
        logging.info(f"Processed {count} images from {dir_type} directory")

    if total_processed == 0:
        error_msg = "No valid images found. Check dataset structure and tea_variants list."
        logging.error(error_msg)
        raise ValueError(error_msg)

    # Create DataFrame
    df = pd.DataFrame({
        'filename': filenames,
        'tea_variant': tea_variant_list,
        'elevation': elevation_list,
        'dataset_type': dataset_type_list
    })

    # Remove duplicates if any
    initial_count = len(df)
    df = df.drop_duplicates(subset=['filename'])
    if len(df) < initial_count:
        logging.warning(f"Removed {initial_count - len(df)} duplicate entries")

    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)

    # Log summary
    logging.info(f"Successfully created labels.csv with {len(df)} entries")
    logging.info(f"Tea variants found: {sorted(df['tea_variant'].unique())}")
    logging.info(f"Elevations found: {sorted(df['elevation'].unique())}")
    logging.info(f"Dataset types: {sorted(df['dataset_type'].unique())}")
    logging.info(f"Saved labels.csv to {output_file}")

    return df


if __name__ == "__main__":
    # Define paths
    train_dir = "../dataset/train"
    test_dir = "../dataset/test"
    train_new_dir = "../dataset_New/train"
    test_new_dir = "../dataset_New/test"
    output_file = "../dataset/labels.csv"

    try:
        df = generate_labels_csv(train_dir, test_dir, train_new_dir, test_new_dir, output_file)
        print(f"Successfully generated labels.csv with {len(df)} entries")
    except Exception as e:
        print(f"Error generating labels.csv: {str(e)}")
        logging.error(f"Failed to generate labels.csv: {str(e)}")