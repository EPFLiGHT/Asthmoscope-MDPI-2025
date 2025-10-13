"""Main script for training and evaluating the Asthmoscope model.

Usage:
    python src/main.py          # Evaluate trained model
    python src/main.py -tm      # Train model and evaluate
"""
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader

from src.config import SEED, get_experiment_config
from src.prepare_data.datasets import AudioDatasetWithSplits
from src.train import pipeline, sample_fit
from src.train.mixup import Mixup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


def load_data(config):
    """Load samples and audio data."""
    logger.info("Loading data...")
    samples_df = pd.read_csv(config["samples_df_path"])
    data = np.load(config["samples_path"])
    logger.info(f"Loaded {len(samples_df)} samples")
    return samples_df, data


def output_aggregation(samples_df, data_loader, val_fold, test_fold, config, device):
    """Generate and save aggregated predictions."""
    logger.info(f"Generating predictions for validation fold {val_fold}, test fold {test_fold}")
    
    sample_outputs = pipeline.make_features(samples_df, data_loader, val_fold, test_fold, config, device)
    
    output_df = samples_df.copy()
    target_str = '+'.join([str(t) for t in config["target"]])
    output_df = output_df.assign(output=sample_outputs).rename(columns={"output": f"output_{target_str}"})
    
    out_dir = Path(config["out_folder"]) / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / config["aggregate_file"].format(
        config["network"]["feature_model"], target_str, val_fold, test_fold
    )
    output_df.to_csv(out_path, index=False)
    logger.info(f"Saved predictions to {out_path}")


def evaluate_and_save(model, loader_1, loader_2, fold_1, fold_2, criterion, best_score, config, device, val_first=True):
    """Evaluate model and save if it's the best."""
    if val_first:
        val_loader = loader_1
        val_fold = fold_1
        test_fold = fold_2
    else:
        val_loader = loader_2
        val_fold = fold_2
        test_fold = fold_1

    metrics, figures_dict, example_spect = sample_fit.evaluate(model, val_loader, criterion, device)
    f1_score = metrics['f1_score']

    if f1_score > best_score:
        best_score = f1_score
        logger.info(f"New best model (F1: {f1_score:.4f}) for folds {val_fold}-{test_fold}")
        
        target_str = '+'.join([str(t) for t in config["target"]])
        out_dir = Path(config["out_folder"]) / "models"
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / config["model_file"].format(
            config["network"]["feature_model"], target_str, val_fold, test_fold
        )
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")

    return metrics, figures_dict, best_score


def fit_samples(model, train_loader, loader_1, loader_2, fold_1, fold_2, optimizer, criterion, cv_index, config):
    """Train the model."""
    logger.info(f"Starting training for {config['epochs']} epochs")
    
    best_score_1 = 0
    best_score_2 = 0

    mixup_augmenter = None
    if config["mixup"]:
        mixup_augmenter = Mixup(mixup_alpha=config["mixup_alpha"], device=device)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=config["epochs"]
    )

    for epoch in range(config["epochs"]):
        step = cv_index * config["epochs"] + epoch
        
        train_loss = sample_fit.train_epoch(
            model, train_loader, optimizer, criterion, epoch, device,
            mixup_augmenter=mixup_augmenter, scheduler=scheduler, log_interval=10
        )
        
        wandb.log({"Train Loss": train_loss}, step=step)

        if epoch + 1 >= config["validation_start"]:
            # Evaluate on first fold
            metrics, figures_dict, best_score_1 = evaluate_and_save(
                model, loader_1, loader_2, fold_1, fold_2, criterion,
                best_score_1, config, device, val_first=True
            )
            wandb.log(metrics, step=step)
            wandb_figures_dict = {k: wandb.Image(fig) for k, fig in figures_dict.items()}
            wandb.log(wandb_figures_dict, step=step)

            # Evaluate on second fold
            metrics, figures_dict, best_score_2 = evaluate_and_save(
                model, loader_1, loader_2, fold_1, fold_2, criterion,
                best_score_2, config, device, val_first=False
            )


def model_pipeline(samples_df, data, fold_1, fold_2, cv_index, config, train_model):
    """Main pipeline for training and/or evaluation."""
    logger.info(f"Running pipeline for folds {fold_1} and {fold_2}")
    
    # Create model and data loaders
    model, train_loader, loader_1, loader_2, optimizer, criterion = pipeline.make_sample_model(
        samples_df, data, fold_1, fold_2, config, device
    )

    if train_model:
        fit_samples(model, train_loader, loader_1, loader_2, fold_1, fold_2, optimizer, criterion, cv_index, config)

    # Create dataset for predictions
    ds = AudioDatasetWithSplits(
        samples_df, data=data, target=config["target"], 
        preprocessing=config["preprocessing"],
        pre_config=config["pre_config"], 
        split_config=config["split_config"], 
        train=False
    )

    data_loader = DataLoader(ds, batch_size=1, shuffle=False)

    # Generate predictions
    output_aggregation(ds.samples_df, data_loader, val_fold=fold_1, test_fold=fold_2, config=config, device=device)
    output_aggregation(ds.samples_df, data_loader, val_fold=fold_2, test_fold=fold_1, config=config, device=device)


def run_experiment(train_model=False, online=True):
    """Run the experiment."""
    # Get configuration
    config = get_experiment_config()
    
    # Initialize wandb
    mode = "online" if online else "offline"
    logger.info(f"Initializing wandb in {mode} mode")
    
    with wandb.init(project="asthmoscope_polyclinique", config=config, mode=mode):
        config = wandb.config
        
        # Load data
        samples_df, data = load_data(config)
        
        # Run on first fold combination
        fold_1, fold_2 = config["cv_folds"][0]
        logger.info(f"Processing folds {fold_1} and {fold_2}")
        model_pipeline(samples_df, data, fold_1, fold_2, 0, config, train_model)
        
        logger.info("Experiment completed successfully!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train and/or evaluate Asthmoscope model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py          # Evaluate trained model
  python src/main.py -tm      # Train model and evaluate
        """
    )
    parser.add_argument(
        "-tm", "--train_model",
        action="store_true",
        help="Train the model (default: evaluate only)"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run wandb in offline mode"
    )
    
    args = parser.parse_args()
    
    if args.train_model:
        logger.info("Mode: Training + Evaluation")
    else:
        logger.info("Mode: Evaluation only")
    
    run_experiment(train_model=args.train_model, online=not args.offline)


if __name__ == "__main__":
    main()
