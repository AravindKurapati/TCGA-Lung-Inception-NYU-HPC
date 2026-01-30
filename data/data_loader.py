"""Data loading utilities for HDF5 files and patient labels"""

import numpy as np
import h5py

def load_hdf5_data(hdf5_path, img_key='img_z_latent', slide_key='slides', tile_key='tiles'):
    """Load images, slides, and tiles from HDF5 file"""
    with h5py.File(hdf5_path, 'r') as h5_file:
        images = np.array(h5_file[img_key])
        slides = np.array(h5_file[slide_key])
        tiles = np.array(h5_file[tile_key])
    return images, slides, tiles

def load_patient_labels(label_file):
    """Load patient labels from text file"""
    patient_to_label = {}
    with open(label_file, 'r') as file:
        for line in file:
            patient_id, label = line.strip().split()
            patient_to_label[patient_id[:12]] = 1 if 'LUAD' in label else 0
    print(f"Loaded {len(patient_to_label)} patient labels")
    return patient_to_label

def get_patient_id(slide_id):
    """Extract patient ID from slide ID (first 12 characters)"""
    return slide_id[:12]

def get_labels_for_slides(slides, patient_to_label):
    """Map slide IDs to patient labels"""
    labels = []
    unmapped_ids = []
    
    for slide in slides:
        slide_id = slide.decode("utf-8")[:12]
        label = patient_to_label.get(slide_id, None)
        
        if label is None:
            unmapped_ids.append(slide_id)
        labels.append(label)
    
    if unmapped_ids:
        print(f"Warning: {len(unmapped_ids)} slides without labels")
    
    return np.array(labels, dtype=object)

def map_patient_labels(slides, labels):
    """Create mapping of patient IDs to labels"""
    patient_to_label_map = {}
    
    for slide, label in zip(slides, labels):
        slide_id = slide.decode("utf-8") if isinstance(slide, bytes) else slide
        patient_id = slide_id[:12]
        
        if patient_id not in patient_to_label_map:
            patient_to_label_map[patient_id] = label
        elif patient_to_label_map[patient_id] != label:
            print(f"Warning: Inconsistent labels for {patient_id}")
    
    return patient_to_label_map
