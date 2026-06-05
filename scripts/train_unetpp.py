"""
U-Net++ Training Script — Research Edition
Adds:
  - Curriculum Learning via CurriculumPatchDataset
  - Topological Loss (activated at TOPO_START_EPOCH)
  - Stage-aware DataLoader rebuilding
  - Per-epoch curriculum stats logging
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet_plus_plus import UNetPlusPlus, count_parameters
from scripts.dataloader_unetpp import create_data_loaders, PatchDataset, CurriculumPatchDataset
from models.losses_unetpp import DeepSupervisionLoss, BCEDiceLoss, dice_coefficient, calculate_metrics, MetricsTracker


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Training configuration."""

    BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_ROOT      = os.path.join(BASE_DIR, 'Retina')
    TRAIN_IMG_DIR  = os.path.join(DATA_ROOT, 'train', 'image')
    TRAIN_MASK_DIR = os.path.join(DATA_ROOT, 'train', 'mask')
    TEST_IMG_DIR   = os.path.join(DATA_ROOT, 'test',  'image')
    TEST_MASK_DIR  = os.path.join(DATA_ROOT, 'test',  'mask')
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'results', 'checkpoints_unetpp')

    # Model
    IN_CHANNELS      = 3
    OUT_CHANNELS     = 1
    DEEP_SUPERVISION = True

    # Training
    EPOCHS        = 60
    BATCH_SIZE    = 8
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY  = 1e-5

    # Data
    PATCH_SIZE   = 128
    STRIDE       = 64
    NUM_WORKERS  = 2
    AUGMENT      = True

    # Mixed precision + gradient tricks
    MIXED_PRECISION    = True
    GRADIENT_CLIP      = 1.0
    ACCUMULATION_STEPS = 2

    # Callbacks
    EARLY_STOPPING_PATIENCE = 10
    LR_PATIENCE             = 5
    LR_FACTOR               = 0.1
    MIN_LR                  = 1e-6

    # -------------------------------------------------------------------------
    # [NEW] Curriculum Learning
    # -------------------------------------------------------------------------
    USE_CURRICULUM         = True   # Set False to reproduce baseline (hard filter)
    # Epoch boundaries for curriculum stages:
    #   Stage 1: epochs 1  .. CURRICULUM_STAGE1_END  (vessel_ratio >= 0.05)
    #   Stage 2: epochs S1+1.. CURRICULUM_STAGE2_END  (vessel_ratio >= 0.02)
    #   Stage 3: epochs S2+1.. EPOCHS                 (vessel_ratio >= 0.001)
    CURRICULUM_STAGE1_END  = 20
    CURRICULUM_STAGE2_END  = 40

    # -------------------------------------------------------------------------
    # [NEW] Topological Loss
    # -------------------------------------------------------------------------
    USE_TOPO_LOSS   = True   # Set False to reproduce baseline without topo
    TOPO_WEIGHT     = 0.1    # Lambda — ablate over {0.05, 0.1, 0.2}
    TOPO_START_EPOCH = 21    # Activate after Stage 1 warmup

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_config():
    print('\n' + '='*80)
    print('U-NET++ RESEARCH TRAINING — Curriculum + Topological Loss')
    print('='*80)
    print(f'Device:          {Config.DEVICE}')
    if torch.cuda.is_available():
        print(f'GPU:             {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory:      {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
    print(f'Epochs:          {Config.EPOCHS}')
    print(f'Batch size:      {Config.BATCH_SIZE} (effective {Config.BATCH_SIZE*Config.ACCUMULATION_STEPS})')
    print(f'Curriculum:      {Config.USE_CURRICULUM}')
    print(f'  Stage 1 end:   epoch {Config.CURRICULUM_STAGE1_END}')
    print(f'  Stage 2 end:   epoch {Config.CURRICULUM_STAGE2_END}')
    print(f'Topo Loss:       {Config.USE_TOPO_LOSS} (lambda={Config.TOPO_WEIGHT}, start epoch={Config.TOPO_START_EPOCH})')
    print('='*80 + '\n')


# =============================================================================
# CURRICULUM STAGE HELPER
# =============================================================================

def get_curriculum_stage(epoch: int) -> int:
    """Map epoch number to curriculum stage (1, 2, or 3)."""
    if epoch <= Config.CURRICULUM_STAGE1_END:
        return 1
    elif epoch <= Config.CURRICULUM_STAGE2_END:
        return 2
    else:
        return 3


# =============================================================================
# TRAIN / VALIDATE
# =============================================================================

def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    metrics = MetricsTracker()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{Config.EPOCHS} [TRAIN]')
    optimizer.zero_grad()

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks  = masks.to(device)

        if Config.MIXED_PRECISION:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss    = criterion(outputs, masks) / Config.ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            if (batch_idx + 1) % Config.ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            loss    = criterion(outputs, masks) / Config.ACCUMULATION_STEPS
            loss.backward()
            if (batch_idx + 1) % Config.ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
                optimizer.step()
                optimizer.zero_grad()

        final_out = outputs[-1] if isinstance(outputs, tuple) else outputs
        dice = dice_coefficient(final_out, masks)
        metrics.update(loss.item() * Config.ACCUMULATION_STEPS, dice)
        pbar.set_postfix({'loss': f'{loss.item()*Config.ACCUMULATION_STEPS:.4f}', 'dice': f'{dice:.4f}'})

    return metrics.get_average()


def validate_epoch(model, val_loader, criterion, device, epoch):
    model.eval()
    metrics     = MetricsTracker()
    all_metrics = []
    pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{Config.EPOCHS} [VAL]  ')

    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks  = masks.to(device)
            outputs = model(images)
            loss    = criterion(outputs, masks)
            final_out = outputs[-1] if isinstance(outputs, tuple) else outputs
            dice      = dice_coefficient(final_out, masks)
            m         = calculate_metrics(final_out, masks)
            metrics.update(loss.item(), dice)
            all_metrics.append(m)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})

    result = metrics.get_average()
    result.update({
        'dice':        np.mean([m['dice']        for m in all_metrics]),
        'accuracy':    np.mean([m['accuracy']    for m in all_metrics]),
        'sensitivity': np.mean([m['sensitivity'] for m in all_metrics]),
        'specificity': np.mean([m['specificity'] for m in all_metrics]),
        'auc':         np.mean([m['auc']         for m in all_metrics]),
    })
    return result


def save_checkpoint(model, optimizer, epoch, metrics, is_best, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt = {'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'metrics': metrics}
    torch.save(ckpt, os.path.join(checkpoint_dir, 'latest.pth'))
    if is_best:
        torch.save(ckpt, os.path.join(checkpoint_dir, 'best.pth'))
        print(f"[SAVE] Best model saved — Dice: {metrics['dice']:.4f}")


def plot_training_history(history, checkpoint_dir):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(history['train_loss'], label='Train', marker='o')
    axes[0].plot(history['val_loss'],   label='Val',   marker='s')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history['train_dice'], label='Train', marker='o')
    axes[1].plot(history['val_dice'],   label='Val',   marker='s')
    axes[1].set_title('Dice'); axes[1].legend(); axes[1].grid(True)

    # Mark curriculum stage transitions
    for ax in axes:
        ax.axvline(x=Config.CURRICULUM_STAGE1_END-1, color='orange',
                   linestyle='--', alpha=0.7, label='Stage 2 start')
        ax.axvline(x=Config.CURRICULUM_STAGE2_END-1, color='red',
                   linestyle='--', alpha=0.7, label='Stage 3 start')
        if Config.USE_TOPO_LOSS:
            ax.axvline(x=Config.TOPO_START_EPOCH-1, color='purple',
                       linestyle=':', alpha=0.7, label='Topo loss ON')

    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'training_history.png'), dpi=150)
    print(f"[SAVE] Training curves saved.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print_config()
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data loaders
    # -------------------------------------------------------------------------
    print('[LOAD] Creating data loaders...')
    train_loader, val_loader, train_dataset = create_data_loaders(
        Config.TRAIN_IMG_DIR, Config.TRAIN_MASK_DIR,
        Config.TEST_IMG_DIR,  Config.TEST_MASK_DIR,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        patch_size=Config.PATCH_SIZE,
        stride=Config.STRIDE,
        use_curriculum=Config.USE_CURRICULUM
    )
    print(f'[OK] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    print('\n[LOAD] Creating U-Net++ model...')
    model  = UNetPlusPlus(Config.IN_CHANNELS, Config.OUT_CHANNELS,
                          Config.DEEP_SUPERVISION).to(Config.DEVICE)
    params = count_parameters(model)
    print(f'[OK] {params:,} parameters ({params/1e6:.1f}M)')

    # -------------------------------------------------------------------------
    # Loss / Optimizer / Scheduler
    # -------------------------------------------------------------------------
    criterion = DeepSupervisionLoss(
        weights=[0.25, 0.25, 0.25, 1.0],
        use_topo=False,                    # starts OFF — activated mid-training
        topo_weight=Config.TOPO_WEIGHT
    )
    optimizer = Adam(model.parameters(), lr=Config.LEARNING_RATE,
                     weight_decay=Config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=Config.LR_FACTOR,
                                  patience=Config.LR_PATIENCE, min_lr=Config.MIN_LR,
                                  verbose=True)
    scaler = torch.amp.GradScaler('cuda') if Config.MIXED_PRECISION else None

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    history = {k: [] for k in ['train_loss','train_dice','val_loss','val_dice',
                                'val_accuracy','val_sensitivity','val_specificity','val_auc',
                                'curriculum_stage','n_active_patches']}
    best_val_dice    = 0.0
    patience_counter = 0
    start_time       = time.time()

    print('\n' + '='*80)
    print('STARTING TRAINING')
    print('='*80 + '\n')

    for epoch in range(1, Config.EPOCHS + 1):

        # --- Curriculum stage update ---
        if Config.USE_CURRICULUM and isinstance(train_dataset, CurriculumPatchDataset):
            stage = get_curriculum_stage(epoch)
            train_dataset.set_curriculum_stage(stage)
            # Rebuild loader so shuffle sees updated active_patches
            train_loader = DataLoader(
                train_dataset, batch_size=Config.BATCH_SIZE,
                shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True
            )
            stats = train_dataset.get_stage_stats()
            history['curriculum_stage'].append(stage)
            history['n_active_patches'].append(stats['n_patches'])

        # --- Activate topo loss at TOPO_START_EPOCH ---
        if Config.USE_TOPO_LOSS and epoch == Config.TOPO_START_EPOCH:
            criterion.enable_topo(Config.TOPO_WEIGHT)

        # --- Train / Validate ---
        train_m = train_epoch(model, train_loader, criterion, optimizer, scaler, Config.DEVICE, epoch)
        val_m   = validate_epoch(model, val_loader, criterion, Config.DEVICE, epoch)
        scheduler.step(val_m['loss'])

        # --- Record history ---
        history['train_loss'].append(train_m['loss'])
        history['train_dice'].append(train_m['dice'])
        history['val_loss'].append(val_m['loss'])
        history['val_dice'].append(val_m['dice'])
        history['val_accuracy'].append(val_m['accuracy'])
        history['val_sensitivity'].append(val_m['sensitivity'])
        history['val_specificity'].append(val_m['specificity'])
        history['val_auc'].append(val_m['auc'])

        # --- Console summary ---
        print(f"\nEpoch {epoch}/{Config.EPOCHS}")
        print(f"  Train  loss={train_m['loss']:.4f}  dice={train_m['dice']:.4f}")
        print(f"  Val    loss={val_m['loss']:.4f}  dice={val_m['dice']:.4f}  "
              f"sens={val_m['sensitivity']:.4f}  spec={val_m['specificity']:.4f}  auc={val_m['auc']:.4f}")
        print(f"  LR={optimizer.param_groups[0]['lr']:.2e}  "
              f"Topo={'ON' if criterion.use_topo else 'OFF'}")
        if Config.USE_CURRICULUM and isinstance(train_dataset, CurriculumPatchDataset):
            print(f"  Curriculum stage={stage}  active_patches={stats['n_patches']}  "
                  f"mean_ratio={stats['mean_ratio']:.4f}")
        print('-'*80)

        # --- Checkpointing / early stopping ---
        is_best = val_m['dice'] > best_val_dice
        if is_best:
            best_val_dice = val_m['dice']
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(model, optimizer, epoch, val_m, is_best, Config.CHECKPOINT_DIR)

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f'\n[STOP] Early stopping after epoch {epoch}. Best Dice: {best_val_dice:.4f}')
            break

    # -------------------------------------------------------------------------
    # Post-training
    # -------------------------------------------------------------------------
    elapsed = time.time() - start_time
    print(f'\nTotal training time: {elapsed/60:.1f} min')
    print(f'Best Val Dice:       {best_val_dice:.4f}')

    plot_training_history(history, Config.CHECKPOINT_DIR)

    with open(os.path.join(Config.CHECKPOINT_DIR, 'metrics.json'), 'w') as f:
        json.dump({'best_val_dice': best_val_dice, 'total_epochs': epoch,
                   'training_time': elapsed, 'history': history}, f, indent=4)

    print(f'[OK] Done. Checkpoints → {Config.CHECKPOINT_DIR}')


if __name__ == '__main__':
    main()
