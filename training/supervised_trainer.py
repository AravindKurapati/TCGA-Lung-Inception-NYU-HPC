"""Supervised training pipeline"""

import os
import pandas as pd
from tensorflow.keras import optimizers, losses, callbacks

def train_supervised_model(model, train_dataset, valid_dataset, 
                          train_size, valid_size, args):
    """Train supervised classification model"""
    
    # Compile model
    model.compile(
        optimizer=optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
        loss=losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    # Calculate steps
    steps_per_epoch = train_size // args.batch_size
    validation_steps = valid_size // args.batch_size
    
    # Train
    history = model.fit(
        train_dataset,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_dataset,
        validation_steps=validation_steps,
        callbacks=[early_stopping]
    )
    
    # Save losses
    save_training_history(history, args.model_dir)
    
    return model, history

def save_training_history(history, output_dir):
    """Save training and validation losses to CSV"""
    
    losses_df = pd.DataFrame({
        'epoch': range(1, len(history.history['loss']) + 1),
        'train_loss': history.history['loss'],
        'valid_loss': history.history['val_loss']
    })
    
    csv_path = os.path.join(output_dir, 'training_validation_losses.csv')
    losses_df.to_csv(csv_path, index=False)
    print(f"Training history saved: {csv_path}")
