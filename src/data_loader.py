import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Sign Language MNIST maps 0-24 excluding 9 (J) and 25 (Z) because they require motion
RAW_LABEL_TO_CHAR = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H",
    8: "I", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P",
    16: "Q", 17: "R", 18: "S", 19: "T", 20: "U", 21: "V", 22: "W",
    23: "X", 24: "Y"
}

SORTED_RAW_LABELS = sorted(RAW_LABEL_TO_CHAR.keys())
RAW_TO_IDX = {raw: idx for idx, raw in enumerate(SORTED_RAW_LABELS)}
IDX_TO_CHAR = {idx: RAW_LABEL_TO_CHAR[raw] for idx, raw in enumerate(SORTED_RAW_LABELS)}


def load_csv_dataset(csv_path: str):
    """Reads Sign MNIST CSV and returns normalized (N, 28, 28, 1) tensors and contiguous labels (0..23)."""
    df = pd.read_csv(csv_path)
    raw_labels = df["label"].values
    y = np.array([RAW_TO_IDX[val] for val in raw_labels], dtype=np.int32)

    pixels = df.drop(columns=["label"]).values.astype("float32") / 255.0
    X = pixels.reshape(-1, 28, 28, 1)
    return X, y


def get_train_val_data(train_csv_path: str, val_split: float = 0.15, seed: int = 42):
    """Loads training CSV and creates a stratified train/validation split."""
    X, y = load_csv_dataset(train_csv_path)
    return train_test_split(X, y, test_size=val_split, stratify=y, random_state=seed)


def build_augmentation_layer():
    """
    Domain-safe data augmentation for hand signs.
    NOTE: Horizontal flip is intentionally omitted because flipping alters ASL handedness/meaning.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(factor=0.04, fill_mode="nearest"),
            tf.keras.layers.RandomZoom(height_factor=0.10, width_factor=0.10, fill_mode="nearest"),
            tf.keras.layers.RandomTranslation(height_factor=0.08, width_factor=0.08, fill_mode="nearest"),
        ],
        name="asl_augmentation",
    )