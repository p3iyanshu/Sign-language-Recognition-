import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split

CSV_SEARCH_DIRS = [
    "data",
    "archive (3)",
]

EXTRA_DATASETS = [
    r"D:\Univeristy\Sign Language Recognition\SignAlphaSet",
    r"D:\Univeristy\Sign Language Recognition\ASL_dynamic",
]

MODEL_SAVE_PATH = "models/best_asl_cnn_multi.keras"
LABELS_SAVE_PATH = "models/class_names.txt"
IMG_SIZE = 28
BATCH_SIZE = 64
EPOCHS = 35
SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def preprocess_frame_28x28(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    resized = cv2.resize(enhanced, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    tensor = resized.astype("float32") / 255.0
    return np.expand_dims(tensor, axis=-1)


def extract_video_frames(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total <= 0:
        cap.release()
        return frames
    indices = np.linspace(0, max(total - 1, 0), num=min(num_frames, total), dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(preprocess_frame_28x28(frame))
    cap.release()
    return frames


def load_recursive_dataset(root_dir):
    X_list, y_list = [], []
    if not os.path.exists(root_dir):
        print(f"[WARNING] Path not found: {root_dir}")
        return np.empty((0, IMG_SIZE, IMG_SIZE, 1), dtype="float32"), []

    img_count, vid_count = 0, 0
    for dirpath, _, filenames in os.walk(root_dir):
        folder_name = os.path.basename(dirpath).strip().upper()
        # Match single alphabet folder names A-Z (e.g., ...\A\img1.jpg or ...\J\clip1.mp4)
        if len(folder_name) != 1 or not folder_name.isalpha():
            continue

        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            fpath = os.path.join(dirpath, fname)

            if ext in IMG_EXTS:
                img = cv2.imread(fpath)
                if img is not None:
                    X_list.append(preprocess_frame_28x28(img))
                    y_list.append(folder_name)
                    img_count += 1
            elif ext in VID_EXTS:
                extracted = extract_video_frames(fpath, num_frames=5)
                for fr in extracted:
                    X_list.append(fr)
                    y_list.append(folder_name)
                if extracted:
                    vid_count += 1

    print(f"[INFO] {os.path.basename(root_dir)} -> Loaded {img_count} images + {vid_count} videos ({len(y_list)} total samples)")
    if len(y_list) == 0:
        return np.empty((0, IMG_SIZE, IMG_SIZE, 1), dtype="float32"), []
    return np.array(X_list, dtype="float32"), y_list


def load_all_sign_mnist_csvs():
    X_parts, y_parts = [], []
    for folder in CSV_SEARCH_DIRS:
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith(".csv") and "sign_mnist" in fname.lower():
                csv_path = os.path.join(folder, fname)
                df = pd.read_csv(csv_path)
                if "label" not in df.columns:
                    continue
                raw_labels = df["label"].values
                pixels = df.drop(columns=["label"]).values.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0
                str_labels = [chr(ord("A") + int(lbl)) for lbl in raw_labels]
                print(f"[INFO] Loaded {len(str_labels)} samples from: {csv_path}")
                X_parts.append(pixels)
                y_parts.extend(str_labels)
    return X_parts, y_parts


def get_augmenter():
    return tf.keras.Sequential([
        layers.RandomRotation(0.04, fill_mode="nearest"),
        layers.RandomZoom(0.10, fill_mode="nearest"),
        layers.RandomTranslation(0.08, 0.08, fill_mode="nearest"),
    ], name="handedness_safe_augmenter")


def build_asl_cnn(input_shape=(28, 28, 1), num_classes=26, learning_rate=1e-3, augmenter=None):
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = augmenter(inputs) if augmenter is not None else inputs

    for filters, drop in [(32, 0.20), (64, 0.25)]:
        for _ in range(2):
            x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(drop)(x)

    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu", name="last_conv_activation")(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.40)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="ASL_MultiDataset_CNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    os.makedirs("models", exist_ok=True)
    X_all, y_str_all = load_all_sign_mnist_csvs()

    for folder_path in EXTRA_DATASETS:
        X_fld, y_fld = load_recursive_dataset(folder_path)
        if len(y_fld) > 0:
            X_all.append(X_fld)
            y_str_all.extend(y_fld)

    if len(y_str_all) == 0:
        raise RuntimeError("No images/videos found! Run Get-ChildItem -Recurse -Depth 3 to check folder names.")

    X = np.concatenate(X_all, axis=0)
    class_names = sorted(list(set(y_str_all)))
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    y = np.array([label_to_idx[c] for c in y_str_all], dtype="int32")

    print(f"\n[SUMMARY] Total Combined Samples: {len(y)} | Total Classes ({len(class_names)}): {class_names}")

    with open(LABELS_SAVE_PATH, "w") as f:
        for c in class_names:
            f.write(f"{c}\n")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    print(f"[SPLIT] Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}\n")

    model = build_asl_cnn(
        input_shape=(IMG_SIZE, IMG_SIZE, 1),
        num_classes=len(class_names),
        learning_rate=1e-3,
        augmenter=get_augmenter(),
    )

    cb_list = [
        callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True, verbose=1
        ),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb_list,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[FINAL TEST RESULT] Test Accuracy: {test_acc * 100:.2f}% | Test Loss: {test_loss:.4f}")
    print(f"[SAVED] Best model weights saved to: {MODEL_SAVE_PATH}")
    print(f"[SAVED] Class label order saved to:  {LABELS_SAVE_PATH}")


if __name__ == "__main__":
    main()
