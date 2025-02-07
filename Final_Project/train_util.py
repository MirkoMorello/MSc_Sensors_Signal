from tqdm import tqdm
import torch
import os
from datetime import datetime
import json
import gc
from torch.utils.data import DataLoader


from utils import save_checkpoint, load_checkpoint, _process_audio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility


def train(model, experiment_name, train_loader, val_loader, train_dataset, val_dataset, optimizer, scheduler, 
          loss_fn, num_epochs, device, resume=False):
    train_losses = []
    val_losses = []
    stoi_scores = [] 
    start_epoch = 0
    step = 0
    best_val_loss = float('inf')
    resume_batch = 0  # to skip already processed batches if resuming

    if resume:
        state = load_checkpoint(experiment_name, model, optimizer, scheduler)
        start_epoch = state['epoch']
        step = state['step']
        resume_batch = state.get('batch_idx', 0)
        train_losses = state['train_losses']
        val_losses = state['val_losses']
        stoi_scores = state.get('stoi_scores', [])
        best_val_loss = state['best_val_loss']
        print(f"Resuming training from epoch {start_epoch}, step {step}, batch {resume_batch}")

    model = model.to(device)
    
    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            epoch_train_loss = 0
            train_bar = tqdm(enumerate(train_loader), desc=f'Epoch {epoch+1}/{num_epochs} [Train]', total=len(train_loader), leave=False)
            
            for batch_idx, (noisy_batch, clean_batch) in train_bar:
                # If resuming and we're in the middle of an epoch, skip already processed batches.
                if resume and epoch == start_epoch and batch_idx < resume_batch:
                    continue

                noisy_batch = noisy_batch.to(device)
                clean_batch = clean_batch.to(device)

                optimizer.zero_grad()
                outputs = model(noisy_batch)
                loss = loss_fn(outputs, clean_batch)
                loss.backward()
                optimizer.step()

                step_loss = loss.item()
                epoch_train_loss += step_loss * noisy_batch.size(0)
                train_losses.append(step_loss)
                step += 1

                train_bar.set_postfix({
                    'loss': f'{step_loss:.4f}',
                    'lr': optimizer.param_groups[0]['lr']
                })

                # Save checkpoint every 100 steps (store current batch index)
                if step % 100 == 0:
                    save_checkpoint(experiment_name, model, optimizer, scheduler, 
                                    epoch, step, batch_idx, train_losses, val_losses, stoi_scores)
            
            # ------------ Validation ----------------
            model.eval()
            epoch_val_loss = 0
            epoch_stoi_total = 0.0
            num_batches = 0
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', total=len(val_loader), leave=False)
            
            with torch.no_grad():
                for noisy_batch, clean_batch in val_bar:
                    noisy_batch = noisy_batch.to(device)
                    clean_batch = clean_batch.to(device)
                    
                    outputs = model(noisy_batch)
                    loss = loss_fn(outputs, clean_batch)
                    epoch_val_loss += loss.item() * noisy_batch.size(0)
                    
                    # Process audio to proper shape/device for STOI metric
                    clean_proc = _process_audio(clean_batch)
                    denoised_proc = _process_audio(outputs)
                    batch_stoi = ShortTimeObjectiveIntelligibility(16000).to(device)(denoised_proc, clean_proc)
                    # batch_stoi is a tensor (averaged over the batch), so convert to float.
                    epoch_stoi_total += batch_stoi.item()
                    num_batches += 1
                    
                    val_bar.set_postfix({
                        'val_loss': f'{loss.item():.4f}',
                        'stoi': f'{batch_stoi.item():.4f}'
                    })

            avg_val_loss = epoch_val_loss / len(val_dataset)
            avg_stoi = epoch_stoi_total / num_batches if num_batches > 0 else 0.0
            val_losses.append(avg_val_loss)
            stoi_scores.append(avg_stoi)
            scheduler.step(avg_val_loss)

            # Save best model checkpoint if validation loss improves.
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(experiment_name, model, optimizer, scheduler,
                                epoch, step, batch_idx, train_losses, val_losses, stoi_scores, best=True)
                print(f"New best model saved with val loss: {best_val_loss:.4f} and avg STOI: {avg_stoi:.4f}")

            # Save checkpoint at the end of the epoch (set batch_idx to 0 as epoch is finished)
            save_checkpoint(experiment_name, model, optimizer, scheduler,
                            epoch, step, 0, train_losses, val_losses, stoi_scores)

            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {epoch_train_loss/len(train_dataset):.4f} | Val Loss: {avg_val_loss:.4f} | Avg STOI: {avg_stoi:.4f}')
            print(f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
            print('-'*50)

            # After finishing the resumed epoch, disable further skipping.
            if resume and epoch == start_epoch:
                resume = False

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current state...")
        save_checkpoint(experiment_name, model, optimizer, scheduler,
                        epoch, step, batch_idx, train_losses, val_losses, stoi_scores)
        print("Checkpoint saved. You can resume later with --resume flag")

    return train_losses, val_losses, stoi_scores


def run_training(
    model,
    train_dataset,
    val_dataset,
    optimizer_class,
    optimizer_params,
    scheduler_class,
    scheduler_params,
    loss_fn,
    num_epochs,
    device,
    batch_size=24,
    num_workers=8,
    resume_training=False,
    experiment_name="experiment",
    checkpoint_dir=None,
    train_fn=None,
    load_checkpoint_fn=None
):
    """
    Run training for a given model with provided datasets and training parameters.
            
    Returns:
        train_losses, val_losses, stoi_scores: The training statistics returned by train_fn.
    """
    
    # Set checkpoint directory if not provided
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join("checkpoints", experiment_name)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Initialize optimizer and scheduler
    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    scheduler = scheduler_class(optimizer, **scheduler_params)
    
    # Resume from checkpoint if requested and a loader function is provided
    if resume_training and load_checkpoint_fn is not None:
        if not os.path.exists(checkpoint_dir) or not any(fname.startswith("checkpoint_") for fname in os.listdir(checkpoint_dir)):
            print("No checkpoints found. Starting training from scratch.")
            resume_training = False
            start_epoch = 0
            best_val_loss = float('inf')
        else:
            state = load_checkpoint_fn(experiment_name, model, optimizer, scheduler)
            start_epoch = state.get('epoch', 0)
            best_val_loss = state.get('best_val_loss', float('inf'))
            print(f"Resuming from epoch {start_epoch}, best val loss {best_val_loss:.4f}")
    else:
        start_epoch = 0
        best_val_loss = float('inf')
    
    # Move model to the designated device
    model.to(device)
    
    # Ensure a training function is provided
    if train_fn is None:
        raise ValueError("A training function must be provided via the 'train_fn' parameter.")
    
    # Run the training loop
    train_losses, val_losses, stoi_scores = train_fn(
        model,
        experiment_name,
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
        optimizer,
        scheduler,
        loss_fn,
        num_epochs,
        device,
        resume=resume_training
    )
    
    # Clean up: delete DataLoaders and optimizer/scheduler objects and free GPU memory
    del train_loader, val_loader, optimizer, scheduler, model
    gc.collect()
    torch.cuda.empty_cache()

    
    return train_losses, val_losses, stoi_scores