"""Train/validation/test splitting utilities"""

import numpy as np
from sklearn.model_selection import train_test_split
from data.data_loader import get_patient_id

def split_by_patient(slides, patient_to_label_map, test_size=0.4, val_size=0.5, random_state=42):
    """Split dataset by patient to avoid data leakage"""
    
    # Get unique patient IDs and labels
    unique_patient_ids = np.array(list(patient_to_label_map.keys()))
    patient_labels = np.array(list(patient_to_label_map.values()))
    
    # Train/temp split
    train_patient_ids, temp_patient_ids, train_labels, temp_labels = train_test_split(
        unique_patient_ids, patient_labels, 
        test_size=test_size, random_state=random_state, stratify=patient_labels
    )
    
    # Validation/test split
    valid_patient_ids, test_patient_ids, valid_labels, test_labels = train_test_split(
        temp_patient_ids, temp_labels,
        test_size=val_size, random_state=random_state, stratify=temp_labels
    )
    
    return {
        'train': (train_patient_ids, train_labels),
        'valid': (valid_patient_ids, valid_labels),
        'test': (test_patient_ids, test_labels)
    }

def apply_patient_split(images, slides, tiles, patient_splits):
    """Apply patient-based split to image data"""
    train_ids, valid_ids, test_ids = (
        patient_splits['train'][0],
        patient_splits['valid'][0],
        patient_splits['test'][0]
    )
    
    # Create masks
    train_mask = np.isin([get_patient_id(s.decode("utf-8")) for s in slides], train_ids)
    valid_mask = np.isin([get_patient_id(s.decode("utf-8")) for s in slides], valid_ids)
    test_mask = np.isin([get_patient_id(s.decode("utf-8")) for s in slides], test_ids)
    
    return {
        'train': (images[train_mask], slides[train_mask], tiles[train_mask]),
        'valid': (images[valid_mask], slides[valid_mask], tiles[valid_mask]),
        'test': (images[test_mask], slides[test_mask], tiles[test_mask])
    }
