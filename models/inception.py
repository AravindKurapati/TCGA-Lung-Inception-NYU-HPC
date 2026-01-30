"""InceptionV3 model for slide classification"""

from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3

def create_inception_model(input_shape=(299, 299, 128)):
    """
    Create InceptionV3 model for binary classification
    
    Args:
        input_shape: Input tensor shape (height, width, channels)
        
    Returns:
        Compiled Keras model
    """
    base_model = InceptionV3(include_top=False, weights=None, input_shape=input_shape)
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=base_model.input, outputs=outputs)
    
    return model
