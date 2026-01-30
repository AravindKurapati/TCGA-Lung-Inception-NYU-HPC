"""Main training script"""

import argparse
import os
import sys
import tensorflow as tf
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import (
    load_hdf5_data, load_patient_labels, 
    get_labels_for_slides, map_patient_labels
)
from data.preprocessing import reconstruct_images, get_data_augmentation
from data.tfrecord_utils import write_tfrecords, create_tfrecord_dataset
from utils.dataset_split import split_by_patient, apply_patient_split
from models.inception import create_inception_model
from training.ssl_trainer import train_barlow_twins, extract_representations
from training.supervised_trainer import train_supervised_model
from utils.metrics import compute_auc
from utils.visualization import plot_training_curves

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Inception model on TCGA lung data')
    
    # Data arguments
    parser.add_argument('--test_hdf5', type=str, required=True, help='Path to test HDF5 file')
    parser.add_argument('--labels_txt', type=str, required=True, help='Path to labels file')
    parser.add_argument('--hpc_ids_csv', type=str, help='Path to HPC IDs CSV (optional)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=90, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    
    # Directory arguments
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Checkpoint directory')
    parser.add_argument('--model_dir', type=str, required=True, help='Model save directory')
    
    # SSL arguments
    parser.add_argument('--use_ssl', action='store_true', help='Use Barlow Twins SSL pretraining')
    parser.add_argument('--ssl_epochs', type=int, default=10, help='SSL pretraining epochs')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup
    tf.keras.backend.set_image_data_format('channels_last')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("TCGA Lung Cancer Classification Training")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"PyTorch device: {device}")
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 1. Load data
    print("\n[1/7] Loading data...")
    images, slides, tiles = load_hdf5_data(args.test_hdf5)
    images = images / 255.0
    patient_to_label = load_patient_labels(args.labels_txt)
    
    # 2. Create label mappings
    print("\n[2/7] Creating label mappings...")
    slide_labels = get_labels_for_slides(slides, patient_to_label)
    patient_to_label_map = map_patient_labels(slides, slide_labels)
    
    # 3. Split data by patient
    print("\n[3/7] Splitting dataset by patient...")
    patient_splits = split_by_patient(slides, patient_to_label_map)
    data_splits = apply_patient_split(images, slides, tiles, patient_splits)
    
    print(f"  Train patients: {len(patient_splits['train'][0])}")
    print(f"  Valid patients: {len(patient_splits['valid'][0])}")
    print(f"  Test patients: {len(patient_splits['test'][0])}")
    
    # 4. Reconstruct images
    print("\n[4/7] Reconstructing slide images...")
    train_images = reconstruct_images(*data_splits['train'], args.img_size, 128)
    valid_images = reconstruct_images(*data_splits['valid'], args.img_size, 128)
    test_images = reconstruct_images(*data_splits['test'], args.img_size, 128)
    
    print(f"  Train slides: {len(train_images)}")
    print(f"  Valid slides: {len(valid_images)}")
    print(f"  Test slides: {len(test_images)}")

# Get labels for reconstructed images
train_labels = patient_splits['train'][1][:len(train_images)]
valid_labels = patient_splits['valid'][1][:len(valid_images)]
test_labels = patient_splits['test'][1][:len(test_images)]

# 5. Optional: SSL pretraining
if args.use_ssl:
    print("\n[5/7] SSL pretraining with Barlow Twins...")
    backbone = train_barlow_twins(train_images, epochs=args.ssl_epochs, device=device)
    train_embeddings = extract_representations(backbone, train_images, device=device)
    print(f"  Extracted embeddings shape: {train_embeddings.shape}")
else:
    print("\n[5/7] Skipping SSL pretraining")

# 6. Create TFRecords
print("\n[6/7] Creating TFRecords...")
data_aug = get_data_augmentation()

train_tfrecord = os.path.join(args.model_dir, 'train_data.tfrecord')
valid_tfrecord = os.path.join(args.model_dir, 'valid_data.tfrecord')

write_tfrecords(train_tfrecord, train_images, train_labels, data_aug)
write_tfrecords(valid_tfrecord, valid_images, valid_labels, data_aug)

train_dataset = create_tfrecord_dataset(train_tfrecord, args.batch_size, args.img_size, is_training=True)
valid_dataset = create_tfrecord_dataset(valid_tfrecord, args.batch_size, args.img_size, is_training=False)

# 7. Train supervised model
print("\n[7/7] Training supervised model...")
model = create_inception_model((args.img_size, args.img_size, 128))
model.summary()

model, history = train_supervised_model(
    model, train_dataset, valid_dataset,
    len(train_images), len(valid_images), args
)

# Evaluate
print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)
compute_auc(model, valid_images, valid_labels, "Validation")
compute_auc(model, test_images, test_labels, "Test")

# Save model
model_path = os.path.join(args.model_dir, 'trained_model.h5')
model.save(model_path)
print(f"\nModel saved: {model_path}")

# Plot training curves
history_csv = os.path.join(args.model_dir, 'training_validation_losses.csv')
curves_path = os.path.join(args.model_dir, 'training_curves.png')
plot_training_curves(history_csv, curves_path)

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
