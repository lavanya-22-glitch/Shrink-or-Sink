import tensorflow as tf
import os
import numpy as np
import argparse
from model import build_model

def read_stl10_images(filepath):
    """Reads STL-10 binary image format."""
    RECORD_BYTES = 96 * 96 * 3
    dataset = tf.data.FixedLengthRecordDataset(filepath, RECORD_BYTES)
    def decode_image(record):
        decoded = tf.io.decode_raw(record, tf.uint8)
        reshaped = tf.reshape(decoded, (3, 96, 96))
        transposed = tf.transpose(reshaped, [2, 1, 0])
        return tf.cast(transposed, tf.float32)
    return dataset.map(decode_image)

def read_stl10_labels(filepath):
    """Reads STL-10 binary label format."""
    dataset = tf.data.FixedLengthRecordDataset(filepath, 1)
    def decode_label(record):
        decoded = tf.io.decode_raw(record, tf.uint8)
        return tf.cast(decoded[0], tf.int32) - 1
    return dataset.map(decode_label)

def evaluate_keras(model_path, data_dir):
    model = build_model()
    model.load_weights(model_path)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    test_x = read_stl10_images(os.path.join(data_dir, 'test_X.bin'))
    test_y = read_stl10_labels(os.path.join(data_dir, 'test_y.bin'))
    
    preprocess = tf.keras.applications.mobilenet_v3.preprocess_input
    test_ds = tf.data.Dataset.zip((test_x, test_y))
    test_ds = test_ds.map(lambda x,y: (preprocess(x), y)).batch(64)
    
    print("Evaluating Keras model...")
    _, accuracy = model.evaluate(test_ds)
    return accuracy

def evaluate_tflite(model_path, data_dir):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    is_quantized = input_details['dtype'] == np.uint8
    
    test_x = read_stl10_images(os.path.join(data_dir, 'test_X.bin'))
    test_y = read_stl10_labels(os.path.join(data_dir, 'test_y.bin'))
    
    def preprocess_tflite(img):
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.mobilenet_v3.preprocess_input(img)
        if is_quantized:
            scale, zero_point = input_details['quantization']
            img = img / scale + zero_point
            img = tf.cast(img, tf.uint8)
        return img

    test_ds = tf.data.Dataset.zip((test_x, test_y)).batch(1)
    
    correct = 0
    total = 0
    print("Evaluating TFLite model...")
    for img, label in test_ds:
        # Preprocess manually (faster than mapping if batching 1 by 1)
        processed_img = preprocess_tflite(img[0])
        processed_img = tf.expand_dims(processed_img, 0)
        
        interpreter.set_tensor(input_details['index'], processed_img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details['index'])
        
        prediction = np.argmax(output[0])
        if prediction == label.numpy()[0]:
            correct += 1
        total += 1
    
    accuracy = correct / total
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Evaluate STL-10 Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to stl10_binary folder')
    parser.add_argument('--model_path', type=str, required=True, help='Path to .weights.h5 or .tflite file')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return

    if args.model_path.endswith('.tflite'):
        accuracy = evaluate_tflite(args.model_path, args.data_dir)
    else:
        accuracy = evaluate_keras(args.model_path, args.data_dir)

    print("\n" + "="*40)
    print(f"FINAL ACCURACY: {accuracy * 100:.2f}%")
    print("="*40)

if __name__ == '__main__':
    main()
