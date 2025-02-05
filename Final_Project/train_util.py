from tqdm import tqdm
import torch
import os
from datetime import datetime
import json

from utils import save_checkpoint, load_checkpoint, _process_audio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility


def train(model, experiment_name, train_loader, val_loader, train_dataset, val_dataset, optimizer, scheduler, 
          loss_fn, num_epochs, device, resume=False):
    train_losses = []
    val_losses = []
    stoi_scores = []  # List to store per-epoch STOI scores
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