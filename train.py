import tensorflow as tf
import os
import numpy as np
import argparse
from tqdm import tqdm
from model import build_model

# Need `image-classifiers` (pip install image-classifiers)
try:
    from classification_models.tfkeras import Classifiers
except ImportError:
    print("Please install image-classifiers: pip install image-classifiers")
    exit(1)

# --- GLOBAL CONFIG AND SEEDS ---

def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# --- DATA LOADER UTILS ---

def read_stl10_images(filepath, count):
    RECORD_BYTES = 96 * 96 * 3
    dataset = tf.data.FixedLengthRecordDataset(filepath, RECORD_BYTES)
    def decode_image(record):
        decoded = tf.io.decode_raw(record, tf.uint8)
        reshaped = tf.reshape(decoded, (3, 96, 96))
        transposed = tf.transpose(reshaped, [2, 1, 0])
        return tf.cast(transposed, tf.float32)
    return dataset.take(count).map(decode_image)

def read_stl10_labels_hard(filepath, count, num_classes=11):
    dataset = tf.data.FixedLengthRecordDataset(filepath, 1)
    def decode_label(record):
        decoded = tf.io.decode_raw(record, tf.uint8)
        label_0_9 = tf.cast(decoded[0], tf.int32) - 1
        return tf.one_hot(label_0_9, num_classes)
    return dataset.take(count).map(decode_label)

def read_stl10_labels_soft(filepath, count, num_classes=11):
    RECORD_BYTES = num_classes * 4
    dataset = tf.data.FixedLengthRecordDataset(filepath, RECORD_BYTES)
    def decode_soft_label(record):
        return tf.io.decode_raw(record, tf.float32)
    return dataset.take(count).map(decode_soft_label)

def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize_with_crop_or_pad(image, 104, 104)
    image = tf.image.random_crop(image, size=[96, 96, 3])
    image = tf.image.random_brightness(image, max_delta=0.2)
    return image, label

# --- 6-PHASE TRAINING LOGIC ---

def phase1_teacher_init(data_dir, output_path, epochs=50):
    """PHASE 1: Train initial ResNet-18 on 5k labeled images (10 classes)."""
    print("\n--- PHASE 1: Teacher Initialization (ResNet-18 10-class) ---")
    ResNet18, preprocess_input = Classifiers.get('resnet18')
    base_model = ResNet18(input_shape=(96, 96, 3), weights=None, include_top=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    train_x = read_stl10_images(os.path.join(data_dir, 'train_X.bin'), 5000)
    # Note: 10 classes for initial pass
    train_y = read_stl10_labels_hard(os.path.join(data_dir, 'train_y.bin'), 5000, num_classes=10)
    
    ds = tf.data.Dataset.zip((train_x, train_y)).map(augment_image).map(lambda x,y: (preprocess_input(x),y)).batch(64).shuffle(1000)
    model.fit(ds, epochs=epochs)
    model.save_weights(output_path)
    print(f"Phase 1 weights saved to {output_path}")

def phase2_pseudo_labeling(data_dir, teacher_weights, output_path):
    """PHASE 2: Map 100k unlabeled images to 11 classes (10 STL + 1 Other)."""
    print("\n--- PHASE 2: Pseudo-labeling (100k unlabeled -> 11 classes) ---")
    ResNet18, preprocess_input = Classifiers.get('resnet18')
    # Build 10-class structure to load Phase 1 weights
    base_model = ResNet18(input_shape=(96, 96, 3), weights=None, include_top=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    model.load_weights(teacher_weights)
    
    unlabeled_x = read_stl10_images(os.path.join(data_dir, 'unlabeled_X.bin'), 100000)
    ds = unlabeled_x.map(preprocess_input).batch(128)
    
    print(f"Generating 11-class soft labels into {output_path}...")
    with open(output_path, 'wb') as f:
        for batch in tqdm(ds, total=100000//128 + 1):
            preds_10 = model.predict_on_batch(batch)
            confidences = np.max(preds_10, axis=1, keepdims=True)
            other_prob = 1.0 - confidences
            soft_labels_11 = np.concatenate([preds_10 * confidences, other_prob], axis=1)
            f.write(soft_labels_11.astype(np.float32).tobytes())

def phase3_noisy_student(data_dir, soft_labels_path, output_path, epochs=50):
    """PHASE 3: Train ResNet-18 on 105k images (pseudo+labeled)."""
    print("\n--- PHASE 3: Noisy Student Training (105k images) ---")
    ResNet18, preprocess_input = Classifiers.get('resnet18')
    base_model = ResNet18(input_shape=(96, 96, 3), weights=None, include_top=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(11, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    real_x = read_stl10_images(os.path.join(data_dir, 'train_X.bin'), 5000)
    real_y = read_stl10_labels_hard(os.path.join(data_dir, 'train_y.bin'), 5000, num_classes=11)
    unlabeled_x = read_stl10_images(os.path.join(data_dir, 'unlabeled_X.bin'), 100000)
    unlabeled_y = read_stl10_labels_soft(soft_labels_path, 100000)
    
    ds = tf.data.Dataset.zip((real_x, real_y)).concatenate(tf.data.Dataset.zip((unlabeled_x, unlabeled_y)))
    ds = ds.map(augment_image).map(lambda x,y: (preprocess_input(x),y)).shuffle(10000).batch(64)
    
    model.fit(ds, epochs=epochs)
    model.save_weights(output_path)

def phase4_clean_recalibration(data_dir, noisy_weights, output_path, epochs=50):
    """PHASE 4: Final Fine-tuning on 5k labeled images with VERY low learning rate."""
    print("\n--- PHASE 4: Clean Re-calibration (Fine-tuning on 5k) ---")
    ResNet18, preprocess_input = Classifiers.get('resnet18')
    base_model = ResNet18(input_shape=(96, 96, 3), weights=None, include_top=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(11, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    model.load_weights(noisy_weights)
    
    # Very low learning rate for re-calibration
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    train_x = read_stl10_images(os.path.join(data_dir, 'train_X.bin'), 5000)
    train_y = read_stl10_labels_hard(os.path.join(data_dir, 'train_y.bin'), 5000, num_classes=11)
    ds = tf.data.Dataset.zip((train_x, train_y)).map(augment_image).map(lambda x,y: (preprocess_input(x),y)).batch(32)
    
    model.fit(ds, epochs=epochs)
    model.save_weights(output_path)

class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temp = tf.Variable(5.0, trainable=False)

    def train_step(self, data):
        x, y = data
        teacher_preds = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            student_preds = self.student(x, training=True)
            student_loss = tf.keras.losses.categorical_crossentropy(y, student_preds)
            distill_loss = tf.keras.losses.kl_divergence(
                tf.nn.softmax(teacher_preds / self.temp, axis=1),
                tf.nn.softmax(student_preds / self.temp, axis=1)
            )
            loss = 0.1 * student_loss + 0.9 * (self.temp**2) * distill_loss
        
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        self.compiled_metrics.update_state(y, student_preds)
        return {m.name: m.result() for m in self.metrics}

def phase5_boosted_distillation(data_dir, teacher_weights, output_path, epochs=50):
    """PHASE 5: Boosted Distillation into MobileNetV3 (Alpha=1.25)."""
    print("\n--- PHASE 5: Boosted Distillation (Dynamic Temperature) ---")
    ResNet18, preprocess_resnet = Classifiers.get('resnet18')
    t_base = ResNet18(input_shape=(96, 96, 3), weights=None, include_top=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(t_base.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    t_outputs = tf.keras.layers.Dense(11, activation='softmax')(x)
    teacher = tf.keras.Model(inputs=t_base.input, outputs=t_outputs)
    teacher.load_weights(teacher_weights)
    
    student = build_model(num_classes=11)
    distiller = Distiller(student, teacher)
    distiller.compile(optimizer='adam', metrics=['accuracy'])
    
    real_x = read_stl10_images(os.path.join(data_dir, 'train_X.bin'), 5000)
    real_y = read_stl10_labels_hard(os.path.join(data_dir, 'train_y.bin'), 5000, num_classes=11)
    unlabeled_x = read_stl10_images(os.path.join(data_dir, 'unlabeled_X.bin'), 100000)
    unlabeled_y = read_stl10_labels_soft(os.path.join(data_dir, 'unlabeled_soft_y.bin'), 100000)
    
    preprocess_mb = tf.keras.applications.mobilenet_v3.preprocess_input
    ds = tf.data.Dataset.zip((real_x, real_y)).concatenate(tf.data.Dataset.zip((unlabeled_x, unlabeled_y)))
    ds = ds.map(augment_image).map(lambda x,y: (preprocess_mb(x),y)).shuffle(10000).batch(128)
    
    # Simple temperature scheduler
    for epoch in range(epochs):
        distiller.temp.assign(5.0 - (4.0 * epoch / epochs))
        print(f"Epoch {epoch+1}/{epochs} - Temp: {distiller.temp.numpy():.2f}")
        distiller.fit(ds, epochs=1)
    
    student.save_weights(output_path)

def phase6_quantization(keras_weights_path, data_dir, output_path):
    print("\n--- PHASE 6: INT8 Quantization ---")
    model = build_model(num_classes=11)
    model.load_weights(keras_weights_path)
    
    def representative_data_gen():
        train_x = read_stl10_images(os.path.join(data_dir, 'train_X.bin'), 100)
        for img in train_x:
            img = tf.expand_dims(img, 0)
            img = tf.keras.applications.mobilenet_v3.preprocess_input(img)
            yield [img]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Quantized TFLite model saved to {output_path}")

# --- MAIN RUNNER ---

def main():
    parser = argparse.ArgumentParser(description='Train STL-10 end-to-end')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to stl10_binary folder')
    args = parser.parse_args()
    set_seeds(42)

    # Sequential phase execution
    phase1_teacher_init(args.data_dir, 'teacher_init.weights.h5', epochs=50)
    phase2_pseudo_labeling(args.data_dir, 'teacher_init.weights.h5', 'unlabeled_soft_y.bin')
    phase3_noisy_student(args.data_dir, 'unlabeled_soft_y.bin', 'teacher_noisy.weights.h5', epochs=50)
    phase4_clean_recalibration(args.data_dir, 'teacher_noisy.weights.h5', 'teacher_calibrated.weights.h5', epochs=50)
    phase5_boosted_distillation(args.data_dir, 'teacher_calibrated.weights.h5', 'student_boosted.weights.h5', epochs=50)
    phase6_quantization('student_boosted.weights.h5', args.data_dir, 'model.tflite')

if __name__ == '__main__':
    main()
