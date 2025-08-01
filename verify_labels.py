import os
import pandas as pd
import logging
from pathlib import Path
from config import Config


def setup_logging():
    """
    Configure logging for verify_labels.py.
    """
    # Ensure logs directory exists
    log_dir = os.path.dirname("verify_labels.log")
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("verify_labels.log", mode='w'),  # Overwrite previous log
            logging.StreamHandler()
        ]
    )


def find_all_labels_files():
    """
    Find all labels.csv files in the project directory.
    """
    labels_files = []

    # Search for labels.csv files
    for root, dirs, files in os.walk(Config.PROJECT_ROOT):
        for file in files:
            if file.lower() == "labels.csv":
                full_path = os.path.join(root, file)
                labels_files.append(full_path)

    return labels_files


def analyze_labels_file(file_path, is_primary=False):
    """
    Analyze a single labels.csv file.
    """
    file_status = "PRIMARY" if is_primary else "DUPLICATE"
    logging.info(f"Analyzing {file_status} labels file: {file_path}")

    try:
        # Check if file exists and is readable
        if not os.path.exists(file_path):
            logging.error(f"File does not exist: {file_path}")
            return None

        # Check file size
        file_size = os.path.getsize(file_path)
        logging.info(f"File size: {file_size} bytes")

        if file_size == 0:
            logging.error(f"File is empty: {file_path}")
            return None

        # Try to read the CSV
        try:
            labels_df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            logging.error(f"CSV file is empty or invalid: {file_path}")
            return None
        except Exception as e:
            logging.error(f"Error reading CSV file {file_path}: {str(e)}")
            return None

        if labels_df.empty:
            logging.warning(f"CSV file contains no data: {file_path}")
            return labels_df

        # Analyze structure
        logging.info(f"Successfully loaded: {len(labels_df)} rows, {len(labels_df.columns)} columns")
        logging.info(f"Columns: {list(labels_df.columns)}")

        # Check required columns
        required_columns = ['filename', 'tea_variant', 'elevation']
        available_columns = list(labels_df.columns)
        missing_columns = [col for col in required_columns if col not in available_columns]

        if missing_columns:
            logging.error(f"Missing required columns: {missing_columns}")
        else:
            logging.info("✓ All required columns present")

        # Analyze content
        if not labels_df.empty and all(col in labels_df.columns for col in required_columns):
            # Tea variants
            tea_variants = labels_df['tea_variant'].unique()
            logging.info(f"Tea variants ({len(tea_variants)}): {sorted(tea_variants)}")

            # Elevations
            elevations = labels_df['elevation'].unique()
            logging.info(f"Elevations ({len(elevations)}): {sorted(elevations)}")

            # Check for missing values
            missing_values = labels_df.isnull().sum()
            if missing_values.any():
                logging.warning("Missing values detected:")
                for col, count in missing_values.items():
                    if count > 0:
                        logging.warning(f"  {col}: {count} missing values")
            else:
                logging.info("✓ No missing values")

            # Check for duplicates
            duplicate_filenames = labels_df[labels_df.duplicated(subset=['filename'], keep=False)]
            if not duplicate_filenames.empty:
                logging.warning(f"Found {len(duplicate_filenames)} duplicate filenames:")
                for filename in duplicate_filenames['filename'].unique()[:5]:
                    logging.warning(f"  {filename}")
                if len(duplicate_filenames['filename'].unique()) > 5:
                    logging.warning(f"  ... and {len(duplicate_filenames['filename'].unique()) - 5} more")
            else:
                logging.info("✓ No duplicate filenames")

            # Distribution analysis
            logging.info("Distribution by tea variant:")
            variant_counts = labels_df['tea_variant'].value_counts()
            for variant, count in variant_counts.items():
                logging.info(f"  {variant}: {count} samples")

            logging.info("Distribution by elevation:")
            elevation_counts = labels_df['elevation'].value_counts()
            for elevation, count in elevation_counts.items():
                logging.info(f"  {elevation}: {count} samples")

            # Sample entries
            logging.info("Sample entries (first 5):")
            for i, (_, row) in enumerate(labels_df.head().iterrows()):
                logging.info(f"  {i + 1}: {row['filename']} -> {row['tea_variant']}_{row['elevation']}")

        return labels_df

    except Exception as e:
        logging.error(f"Unexpected error analyzing {file_path}: {str(e)}")
        return None


def validate_file_paths(labels_df, labels_file_path):
    """
    Validate that file paths in labels.csv are reasonable.
    """
    logging.info("Validating file paths in labels...")

    if labels_df is None or labels_df.empty:
        logging.warning("No data to validate")
        return

    if 'filename' not in labels_df.columns:
        logging.error("No 'filename' column found")
        return

    # Check path patterns
    path_patterns = {
        'windows_style': 0,
        'unix_style': 0,
        'mixed_style': 0,
        'absolute_paths': 0,
        'relative_paths': 0
    }

    invalid_paths = []
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

    for idx, filename in enumerate(labels_df['filename']):
        if pd.isna(filename):
            invalid_paths.append(f"Row {idx}: Missing filename")
            continue

        filename_str = str(filename)

        # Check path style
        if '\\' in filename_str and '/' in filename_str:
            path_patterns['mixed_style'] += 1
        elif '\\' in filename_str:
            path_patterns['windows_style'] += 1
        elif '/' in filename_str:
            path_patterns['unix_style'] += 1

        # Check if absolute or relative
        if os.path.isabs(filename_str):
            path_patterns['absolute_paths'] += 1
        else:
            path_patterns['relative_paths'] += 1

        # Check file extension
        _, ext = os.path.splitext(filename_str.lower())
        if ext not in valid_extensions:
            invalid_paths.append(f"Row {idx}: Invalid extension '{ext}' in '{filename_str}'")

        # Check for unusual characters
        if any(char in filename_str for char in ['<', '>', ':', '"', '|', '?', '*']):
            invalid_paths.append(f"Row {idx}: Invalid characters in '{filename_str}'")

    # Report path patterns
    logging.info("Path style analysis:")
    for pattern, count in path_patterns.items():
        if count > 0:
            logging.info(f"  {pattern}: {count} files")

    # Report invalid paths
    if invalid_paths:
        logging.warning(f"Found {len(invalid_paths)} invalid file paths:")
        for issue in invalid_paths[:10]:  # Show first 10 issues
            logging.warning(f"  {issue}")
        if len(invalid_paths) > 10:
            logging.warning(f"  ... and {len(invalid_paths) - 10} more issues")
    else:
        logging.info("✓ All file paths appear valid")


def compare_labels_files(primary_df, duplicate_files):
    """
    Compare primary labels file with duplicate files.
    """
    if not duplicate_files:
        return

    logging.info(f"Comparing primary labels file with {len(duplicate_files)} duplicate(s)...")

    for dup_file in duplicate_files:
        logging.info(f"Comparing with: {dup_file}")

        dup_df = analyze_labels_file(dup_file, is_primary=False)

        if dup_df is None:
            logging.warning(f"Could not analyze duplicate file: {dup_file}")
            continue

        if primary_df is None or primary_df.empty:
            logging.info(f"Primary file is empty, duplicate has {len(dup_df)} entries")
            continue

        # Compare basic stats
        logging.info(f"Comparison results:")
        logging.info(f"  Primary entries: {len(primary_df)}")
        logging.info(f"  Duplicate entries: {len(dup_df)}")

        # Compare columns
        primary_cols = set(primary_df.columns)
        dup_cols = set(dup_df.columns)

        if primary_cols == dup_cols:
            logging.info("  ✓ Columns match")
        else:
            logging.warning("  ✗ Columns differ")
            logging.warning(f"    Primary only: {primary_cols - dup_cols}")
            logging.warning(f"    Duplicate only: {dup_cols - primary_cols}")

        # Compare content if both have data and same columns
        if (not primary_df.empty and not dup_df.empty and
                primary_cols == dup_cols and 'filename' in primary_cols):

            primary_files = set(primary_df['filename'])
            dup_files_set = set(dup_df['filename'])

            common_files = primary_files & dup_files_set
            primary_only = primary_files - dup_files_set
            dup_only = dup_files_set - primary_files

            logging.info(f"  File overlap: {len(common_files)} common files")
            if primary_only:
                logging.info(f"  Primary only: {len(primary_only)} files")
            if dup_only:
                logging.info(f"  Duplicate only: {len(dup_only)} files")


def check_labels_consistency():
    """
    Check for consistency issues in labels data.
    """
    logging.info("Checking labels consistency...")

    try:
        labels_df = pd.read_csv(Config.LABELS_FILE)

        if labels_df.empty:
            logging.warning("Labels file is empty")
            return

        # Check for logical inconsistencies
        issues = []

        # Check if same tea variant appears with different elevations
        if 'tea_variant' in labels_df.columns and 'elevation' in labels_df.columns:
            variant_elevations = labels_df.groupby('tea_variant')['elevation'].nunique()
            multi_elevation_variants = variant_elevations[variant_elevations > 1]

            if not multi_elevation_variants.empty:
                issues.append(f"Tea variants with multiple elevations: {list(multi_elevation_variants.index)}")

        # Check filename patterns
        if 'filename' in labels_df.columns:
            # Look for inconsistent naming patterns
            filename_patterns = {}
            for filename in labels_df['filename'].dropna():
                # Extract pattern (e.g., directory structure depth)
                parts = str(filename).replace('\\', '/').split('/')
                depth = len(parts)
                if depth not in filename_patterns:
                    filename_patterns[depth] = 0
                filename_patterns[depth] += 1

            if len(filename_patterns) > 1:
                issues.append(f"Inconsistent filename path depths: {filename_patterns}")

        # Report issues
        if issues:
            logging.warning("Consistency issues found:")
            for issue in issues:
                logging.warning(f"  {issue}")
        else:
            logging.info("✓ No major consistency issues found")

    except Exception as e:
        logging.error(f"Error checking consistency: {str(e)}")


def verify_labels():
    """
    Main function to verify labels.csv files.
    """
    setup_logging()
    logging.info("Starting labels.csv verification")
    logging.info(f"Project root: {Config.PROJECT_ROOT}")
    logging.info(f"Expected primary labels file: {Config.LABELS_FILE}")

    # Find all labels files
    all_labels_files = find_all_labels_files()
    logging.info(f"Found {len(all_labels_files)} labels.csv files in project:")
    for file_path in all_labels_files:
        rel_path = os.path.relpath(file_path, Config.PROJECT_ROOT)
        logging.info(f"  {rel_path}")

    # Identify primary and duplicate files
    primary_file = Config.LABELS_FILE
    duplicate_files = [f for f in all_labels_files if f != primary_file]

    if duplicate_files:
        logging.warning(f"Found {len(duplicate_files)} duplicate labels.csv files")

    # Analyze primary labels file
    logging.info("=" * 60)
    logging.info("ANALYZING PRIMARY LABELS FILE")
    logging.info("=" * 60)

    primary_df = analyze_labels_file(primary_file, is_primary=True)

    if primary_df is not None:
        validate_file_paths(primary_df, primary_file)
        check_labels_consistency()

    # Analyze duplicate files
    if duplicate_files:
        logging.info("=" * 60)
        logging.info("ANALYZING DUPLICATE LABELS FILES")
        logging.info("=" * 60)
        compare_labels_files(primary_df, duplicate_files)

    # Generate recommendations
    logging.info("=" * 60)
    logging.info("RECOMMENDATIONS")
    logging.info("=" * 60)

    recommendations = []

    if primary_df is None:
        recommendations.append("❌ PRIMARY ISSUE: Main labels.csv file is missing or invalid")
        recommendations.append("   → Run generate_labels.py to create a proper labels.csv file")
    elif primary_df.empty:
        recommendations.append("❌ PRIMARY ISSUE: Main labels.csv file is empty")
        recommendations.append("   → Run generate_labels.py to populate the labels.csv file")
    else:
        recommendations.append("✓ PRIMARY ISSUE: Main labels.csv file is valid")