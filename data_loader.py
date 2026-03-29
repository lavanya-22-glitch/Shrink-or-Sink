"""
data_loader.py — Zero-RAM Binary Data Loader for STL-10
========================================================
Uses tf.data.FixedLengthRecordDataset to stream raw binary files
directly from disk, chunk-by-chunk — no materialization in RAM.

STL-10 binary format:
  • Images are stored as flat uint8 arrays in CHW order (3 × 96 × 96 = 27,648 bytes per image)
  • Labels are stored as single uint8 bytes, 1-indexed (1 = airplane, …, 10 = truck)
  • Unlabeled images have no label file

Usage:
    from data_loader import load_stl10_binary

    train_ds, test_ds, unlbl_ds = load_stl10_binary("./data")
"""

import os
import glob
import tensorflow as tf

# ========================= Constants ========================================
IMG_H, IMG_W, IMG_C = 96, 96, 3
RECORD_BYTES = IMG_H * IMG_W * IMG_C   # 27,648 bytes per image
LABEL_BYTES  = 1                       # 1 byte per label
NUM_CLASSES  = 10


# ========================= File Discovery ===================================

def _find_stl10_binary_dir(data_dir: str) -> str:
    """
    Locate the 'stl10_binary' folder inside the TFDS download cache.

    TFDS extracts STL-10 into a path like:
        <data_dir>/downloads/extracted/<hash>/stl10_binary/

    This helper searches for it automatically so the user only needs to pass
    the top-level data directory used with `--data_dir`.
    """
    pattern = os.path.join(data_dir, "**", "stl10_binary")
    matches = glob.glob(pattern, recursive=True)

    # Filter to directories only (glob can match files too)
    matches = [m for m in matches if os.path.isdir(m)]

    if not matches:
        raise FileNotFoundError(
            f"Could not find 'stl10_binary' directory under {data_dir}.\n"
            f"Make sure the STL-10 dataset is downloaded. You can do this by "
            f"running: python -c \"import tensorflow_datasets as tfds; "
            f"tfds.load('stl10', data_dir='{data_dir}')\""
        )

    # If there are multiple matches (e.g. complete + incomplete), prefer the
    # one that does NOT contain "incomplete" in its path.
    for m in matches:
        if "incomplete" not in m.lower():
            return m

    return matches[0]


# ========================= Parsing Functions ================================

def _parse_image(raw_record):
    """
    Decode a single raw binary record into a float32 image tensor.

    STL-10 stores each image as 27,648 bytes in CHW order (channel-first):
        [R_0, R_1, ..., R_9215, G_0, ..., G_9215, B_0, ..., B_9215]

    We decode → reshape (C, H, W) → transpose to (H, W, C) → normalize to [0, 1].
    """
    # Decode raw bytes → uint8 flat tensor
    image = tf.io.decode_raw(raw_record, tf.uint8)

    # Reshape from flat (27648,) → (3, 96, 96) — channel-first
    image = tf.reshape(image, [IMG_C, IMG_H, IMG_W])

    # Transpose to channel-last (96, 96, 3) for TensorFlow/Keras
    image = tf.transpose(image, perm=[1, 2, 0])

    # Normalize to [0, 1] float32
    image = tf.cast(image, tf.float32) / 255.0

    return image


def _parse_label(raw_record):
    """
    Decode a single label byte.

    STL-10 labels are 1-indexed (1–10), so we subtract 1 to get 0-indexed
    labels (0–9) matching standard classification conventions.
    """
    label = tf.io.decode_raw(raw_record, tf.uint8)
    label = tf.cast(label[0], tf.int64) - 1   # 1-indexed → 0-indexed
    return label


def _parse_label_onehot(raw_record):
    """
    Decode a single label byte into a one-hot float32 vector.
    Used for training (matches the format expected by KD loss).
    """
    label = _parse_label(raw_record)
    return tf.one_hot(label, NUM_CLASSES)       # → float32 (10,)


# ========================= Dataset Builders =================================

def _build_image_dataset(bin_path: str) -> tf.data.Dataset:
    """
    Build a FixedLengthRecordDataset that streams images from a .bin file.
    Each record is exactly RECORD_BYTES (27,648) bytes.
    """
    ds = tf.data.FixedLengthRecordDataset(bin_path, RECORD_BYTES)
    ds = ds.map(_parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def _build_label_dataset(bin_path: str, one_hot: bool = False) -> tf.data.Dataset:
    """
    Build a FixedLengthRecordDataset that streams labels from a .bin file.
    Each record is exactly 1 byte.
    """
    parse_fn = _parse_label_onehot if one_hot else _parse_label
    ds = tf.data.FixedLengthRecordDataset(bin_path, LABEL_BYTES)
    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


# ========================= Public API =======================================

def load_stl10_binary(data_dir: str):
    """
    Load the entire STL-10 dataset using FixedLengthRecordDataset — zero RAM.

    Parameters
    ----------
    data_dir : str
        Top-level data directory (the same one passed to `--data_dir`).
        Must contain the TFDS-extracted `stl10_binary/` folder somewhere inside.

    Returns
    -------
    train_ds : tf.data.Dataset
        5,000 labeled training images.  Yields (image, one_hot_label).
        image: float32 (96, 96, 3), range [0, 1]
        label: float32 (10,), one-hot encoded

    test_ds : tf.data.Dataset
        8,000 test images.  Yields (image, sparse_label).
        image: float32 (96, 96, 3), range [0, 1]
        label: int64 scalar, 0-indexed

    unlbl_ds : tf.data.Dataset
        100,000 unlabeled images.  Yields image only.
        image: float32 (96, 96, 3), range [0, 1]
    """
    stl_dir = _find_stl10_binary_dir(data_dir)

    train_x_path = os.path.join(stl_dir, "train_X.bin")
    train_y_path = os.path.join(stl_dir, "train_y.bin")
    test_x_path  = os.path.join(stl_dir, "test_X.bin")
    test_y_path  = os.path.join(stl_dir, "test_y.bin")
    unlbl_x_path = os.path.join(stl_dir, "unlabeled_X.bin")

    # Validate all files exist
    for path in [train_x_path, train_y_path, test_x_path,
                 test_y_path, unlbl_x_path]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing STL-10 binary file: {path}")

    # --- Train: (image, one_hot_label) ---
    train_images = _build_image_dataset(train_x_path)
    train_labels = _build_label_dataset(train_y_path, one_hot=True)
    train_ds = tf.data.Dataset.zip((train_images, train_labels))

    # --- Test: (image, sparse_label) ---
    test_images = _build_image_dataset(test_x_path)
    test_labels = _build_label_dataset(test_y_path, one_hot=False)
    test_ds = tf.data.Dataset.zip((test_images, test_labels))

    # --- Unlabeled: image only ---
    unlbl_ds = _build_image_dataset(unlbl_x_path)

    print(f"  ✅ Loaded STL-10 from binary files (zero-RAM streaming)")
    print(f"     Directory: {stl_dir}")
    print(f"     Train: 5,000 images  │  Test: 8,000 images  │  Unlabeled: 100,000 images")

    return train_ds, test_ds, unlbl_ds


# ========================= Standalone Test ==================================

if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"

    print("Testing FixedLengthRecordDataset loader …\n")
    train_ds, test_ds, unlbl_ds = load_stl10_binary(data_dir)

    # Peek at one sample from each split
    print("\n--- Train sample ---")
    for img, lbl in train_ds.take(1):
        print(f"  Image shape: {img.shape}, dtype: {img.dtype}, "
              f"range: [{img.numpy().min():.3f}, {img.numpy().max():.3f}]")
        print(f"  Label (one-hot): {lbl.numpy()}")

    print("\n--- Test sample ---")
    for img, lbl in test_ds.take(1):
        print(f"  Image shape: {img.shape}, dtype: {img.dtype}")
        print(f"  Label (sparse): {lbl.numpy()}")

    print("\n--- Unlabeled sample ---")
    for img in unlbl_ds.take(1):
        print(f"  Image shape: {img.shape}, dtype: {img.dtype}, "
              f"range: [{img.numpy().min():.3f}, {img.numpy().max():.3f}]")

    print("\n✅ All good!")
