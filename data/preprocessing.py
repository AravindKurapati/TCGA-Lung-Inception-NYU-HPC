"""Image preprocessing and reconstruction utilities"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def extract_coordinates(tile_str):
    """Extract x, y coordinates from tile filename"""
    tile_str = tile_str.decode("utf-8")
    tile_str = tile_str.replace('.jpeg', '').replace('.jpg', '')
    x, y = map(int, tile_str.split('_'))
    return x, y

def reconstruct_images(images, slides, tiles, img_size, channels):
    """Reconstruct full slide images from tiles"""
    unique_slides = np.unique(slides)
    slide_to_idx = {slide.decode("utf-8"): idx for idx, slide in enumerate(unique_slides)}
    
    reconstructed = np.zeros((len(unique_slides), img_size, img_size, channels), dtype=np.uint8)
    
    for idx, tile in enumerate(tiles):
        x, y = extract_coordinates(tile)
        slide_id = slides[idx].decode("utf-8")
        slide_idx = slide_to_idx[slide_id]
        reconstructed[slide_idx, x, y] = images[idx]
    
    return reconstructed

def get_data_augmentation():
    """Get data augmentation pipeline"""
    return tf.keras.Sequential([
        layers.RandomFlip('horizontal_and_vertical'),
        layers.RandomRotation(0.2),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.2),
    ])

def extract_patch_embeddings(images, embedding_dim=128):
    """Extract patch embeddings (placeholder - replace with actual model)"""
    return np.random.rand(images.shape[0], embedding_dim).astype(np.float32)
