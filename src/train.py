import os
import yaml
import matplotlib.pyplot as plt
import tensorflow as tf

from src.data_loader import get_train_val_data, build_augmentation_layer
from src.model import build_asl_cnn


def save_training_curves(history, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Accuracy subplot
    axes[0].plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss subplot
    axes[1].plot(history.history["loss"], label="Train Loss", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    tf.keras.utils.set_random_seed(cfg["data"]["random_seed"])

    print("[INFO] Loading and splitting Sign Language MNIST dataset...")
    X_train, X_val, y_train, y_val = get_train_val_data(
        train_csv_path=cfg["data"]["train_path"],
        val_split=cfg["data"]["val_split"],
        seed=cfg["data"]["random_seed"],
    )
    print(f"[INFO] Train samples: {X_train.shape[0]} | Val samples: {X_val.shape[0]}")

    augmenter = build_augmentation_layer()
    model = build_asl_cnn(
        input_shape=(cfg["data"]["img_height"], cfg["data"]["img_width"], cfg["data"]["channels"]),
        num_classes=cfg["data"]["num_classes"],
        learning_rate=cfg["training"]["learning_rate"],
        augmenter=augmenter,
    )
    model.summary()

    os.makedirs(os.path.dirname(cfg["paths"]["model_save_path"]), exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=cfg["paths"]["model_save_path"],
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=cfg["training"]["reduce_lr_patience"],
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=cfg["training"]["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print("[INFO] Starting model training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=cfg["training"]["batch_size"],
        epochs=cfg["training"]["epochs"],
        callbacks=callbacks,
        verbose=1,
    )

    save_training_curves(history, cfg["paths"]["history_plot_path"])
    print(f"[INFO] Best model saved to {cfg['paths']['model_save_path']}")
    print(f"[INFO] Training curves saved to {cfg['paths']['history_plot_path']}")


if __name__ == "__main__":
    main()