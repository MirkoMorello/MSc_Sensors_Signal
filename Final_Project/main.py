#!/usr/bin/env python
"""
run_experiments.py

This script runs multiple experiments defined in a dictionary.
Each experiment uses a different combination of model and loss function.
"""

import os
import gc
import torch
from torch.utils.data import DataLoader

# Import custom losses and training utilities
from custom_loss import hybrid_loss, simple_loss, hybrid_loss_freq
from utils import get_spectrogram_datasets, get_datasets, load_checkpoint  # adjust get_datasets if necessary
from train_util import run_training, train

# Import your models
from models import HybridDenoiser, ResAutoencoder, UNetSpec

# Free up GPU memory and release resources
torch.cuda.empty_cache()
device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)

# Set the dataset directory (adjust to your directory)
dataset_dir = "noisy_speech_dataset"

# ====================== Train settings =======================
num_workers = 8
batch_size = 24
resume_training = True

optimizer_class = torch.optim.AdamW
optimizer_params = {"lr": 3e-4, "weight_decay": 1e-5, "fused": True}
scheduler_class = torch.optim.lr_scheduler.ReduceLROnPlateau
scheduler_params = {"mode": "min", "factor": 0.5, "patience": 3}

# Define experiment configurations in a dictionary
experiments = {
    # "hybrid_autoencoder_v1": {
    #     "model_class": HybridDenoiser,
    #     "loss_fn": simple_loss,
    #     "num_epochs": 6,
    # },
    # "hybrid_autoencoder_v2": {
    #     "model_class": HybridDenoiser,
    #     "loss_fn": hybrid_loss,
    #     "num_epochs": 6,
    # },
    # "ResAutoencoder_autoencoder_v1": {
    #     "model_class": ResAutoencoder,
    #     "loss_fn": simple_loss,
    #     "num_epochs": 6,
    # },
    "ResAutoencoder_autoencoder_v2": {
        "model_class": ResAutoencoder,
        "loss_fn": hybrid_loss,
        "num_epochs": 6,
    },
    "UNetSpec_autoencoder_v2": {
        "model_class": UNetSpec,
        "loss_fn": hybrid_loss_freq,
        "num_epochs": 6,
    },
    "UNetSpec_autoencoder_v1": {
        "model_class": UNetSpec,
        "loss_fn": simple_loss,
        "num_epochs": 6,
    },
}


def main():
    for exp_name, config in experiments.items():
        print(f"\nStarting experiment: {exp_name}")
        
        # Instantiate the model (assumes model_class is callable)
        model_instance = config["model_class"]()  
        loss_fn = config["loss_fn"]
        num_epochs = config["num_epochs"]
        experiment_name = exp_name  # use experiment name from the dictionary key
        
        # Choose dataset based on model type
        if isinstance(model_instance, UNetSpec):
            train_dataset, val_dataset = get_spectrogram_datasets(dataset_dir)
        else:
            train_dataset, val_dataset = get_datasets(dataset_dir)
        
        # Run training using the run_training function
        train_losses, val_losses, stoi_scores = run_training(
            model=model_instance,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            optimizer_class=optimizer_class,
            optimizer_params=optimizer_params,
            scheduler_class=scheduler_class,
            scheduler_params=scheduler_params,
            loss_fn=loss_fn,
            num_epochs=num_epochs,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            resume_training=resume_training,
            experiment_name=experiment_name,
            train_fn=train,
            load_checkpoint_fn=load_checkpoint
        )
        
        # Print final validation loss for the experiment
        print(f"Experiment {exp_name} completed. Final validation loss: {val_losses[-1]:.4f}")
        
        # Clean up: delete model and other GPU objects
        del model_instance, train_dataset, val_dataset
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
