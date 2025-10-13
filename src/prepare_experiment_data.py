"""Prepare experiment data files.

This script prepares the patient_df, samples_df, and audio_data files
required for training and evaluation.

Usage:
    python src/prepare_experiment_data.py
"""
import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_experiment_config
from src.data_setup import prepare_data, get_samples

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_experiment_data(
    download_path: str,
    data_drive_path: str,
    clinical_data_path: str,
    data_dictionary_path: str,
    selected_stethos=None
):
    """Prepare all data files for the experiment.
    
    Args:
        download_path: Path to raw data directory
        data_drive_path: Path to processed data directory
        clinical_data_path: Path to clinical data CSV
        data_dictionary_path: Path to data dictionary CSV
        selected_stethos: List of stethoscope types to include (default: ["L", "E"])
    """
    if selected_stethos is None:
        selected_stethos = ["L", "E"]
    
    # Get experiment configuration
    config = get_experiment_config()
    
    # Set up paths
    recordings_dirpath = Path(data_drive_path).absolute()
    
    # Project configuration
    project_locations = {"Ap": ["GVA"]}
    
    clinical_data_config = {
        'Ap': {
            'clinical_data': clinical_data_path,
            'data_dictionary': data_dictionary_path,
        },
    }
    
    logger.info("=" * 60)
    logger.info("STEP 1: Preparing patient_df")
    logger.info("=" * 60)
    
    # Prepare patient dataframe
    patient_df = prepare_data(
        download_path,
        project_locations,
        clinical_data_and_data_dictionary_filepaths_dict=clinical_data_config,
        recordings_dirpath=str(recordings_dirpath),
        n_splits=5,
        verbose=False,
        copy_files=False
    )
    
    # Save initial patient_df
    patient_df.to_csv(config["patient_df_path"], index=False)
    logger.info(f"Saved patient_df with {len(patient_df)} patients to {config['patient_df_path']}")
    
    logger.info("=" * 60)
    logger.info("STEP 2: Preparing samples_df and audio_data")
    logger.info("=" * 60)
    
    # Generate samples dataframe and audio data array
    get_samples(
        patient_df,
        str(recordings_dirpath),
        config['samples_path'],
        config["samples_df_path"],
        selected_stethos=selected_stethos
    )
    
    # Load and verify
    samples_df = pd.read_csv(config["samples_df_path"])
    data_array = np.load(config['samples_path'])
    
    logger.info(f"Samples dataframe shape: {samples_df.shape}")
    logger.info(f"Audio data array shape: {data_array.shape}")
    
    # Filter patient_df to only include patients with samples
    selected_patients = samples_df['patient'].unique()
    filtered_patient_df = patient_df[patient_df['patient'].isin(selected_patients)].copy()
    
    # Save filtered patient_df
    filtered_patient_df.to_csv(config["patient_df_path"], index=False)
    logger.info(f"Saved filtered patient_df with {len(filtered_patient_df)} patients")
    
    logger.info("=" * 60)
    logger.info("STEP 3: Data summary")
    logger.info("=" * 60)
    
    # Print summary statistics
    logger.info(f"Total patients: {len(filtered_patient_df)}")
    logger.info(f"Total samples: {len(samples_df)}")
    logger.info(f"Samples per patient: {len(samples_df) / len(filtered_patient_df):.1f}")
    
    # Asthma distribution
    asthma_counts = samples_df['asthma'].value_counts()
    logger.info(f"Asthma distribution: {dict(asthma_counts)}")
    
    # Fold distribution
    fold_counts = samples_df.groupby('fold')['patient'].nunique()
    logger.info(f"Patients per fold: {dict(fold_counts)}")
    
    # Stethoscope distribution
    stetho_counts = samples_df['stethoscope'].value_counts()
    logger.info(f"Stethoscope distribution: {dict(stetho_counts)}")
    
    logger.info("=" * 60)
    logger.info("Data preparation complete!")
    logger.info("=" * 60)
    logger.info(f"Files created:")
    logger.info(f"  - {config['patient_df_path']}")
    logger.info(f"  - {config['samples_df_path']}")
    logger.info(f"  - {config['samples_path']}")
    
    return filtered_patient_df, samples_df, data_array


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare experiment data files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python src/prepare_experiment_data.py \\
    --download-path /path/to/raw/data \\
    --data-drive-path /path/to/processed/data \\
    --clinical-data /path/to/clinical_data.csv \\
    --data-dictionary /path/to/data_dictionary.csv
        """
    )
    
    parser.add_argument(
        "--download-path",
        required=True,
        help="Path to raw data directory"
    )
    parser.add_argument(
        "--data-drive-path",
        required=True,
        help="Path to processed data directory"
    )
    parser.add_argument(
        "--clinical-data",
        required=True,
        help="Path to clinical data CSV file"
    )
    parser.add_argument(
        "--data-dictionary",
        required=True,
        help="Path to data dictionary CSV file"
    )
    parser.add_argument(
        "--stethos",
        nargs="+",
        default=["L", "E"],
        help="Stethoscope types to include (default: L E)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.download_path).exists():
        logger.error(f"Download path does not exist: {args.download_path}")
        return 1
    
    if not Path(args.clinical_data).exists():
        logger.error(f"Clinical data file does not exist: {args.clinical_data}")
        return 1
    
    if not Path(args.data_dictionary).exists():
        logger.error(f"Data dictionary file does not exist: {args.data_dictionary}")
        return 1
    
    # Run data preparation
    try:
        prepare_experiment_data(
            download_path=args.download_path,
            data_drive_path=args.data_drive_path,
            clinical_data_path=args.clinical_data,
            data_dictionary_path=args.data_dictionary,
            selected_stethos=args.stethos
        )
        return 0
    except Exception as e:
        logger.error(f"Error preparing data: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
