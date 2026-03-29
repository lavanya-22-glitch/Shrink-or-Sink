import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small

def build_model(input_shape=(96, 96, 3), num_classes=11):
    """
    Builds the MobileNetV3-Small architecture for STL-10.
    We use 11 classes: 10 STL-10 classes + 1 Background/Noise class 
    used for semi-supervised Noisy Student distillation.
    """
    # alpha=1.25 provides higher capacity while remaining within size limits
    base_model = MobileNetV3Small(
        input_shape=input_shape, 
        include_top=False, 
        weights=None, # SCRATCH training (ImageNet NOT allowed)
        alpha=1.25
    )
    
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    return model
