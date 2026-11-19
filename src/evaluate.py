import json
import os
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from src.data_loader import load_csv_dataset, IDX_TO_CHAR


def compute_gradcam(model, img_tensor, layer_name="last_conv_activation"):
    """Generates a Grad-CAM saliency heatmap for a single (1, 28, 28, 1) image."""
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def plot_gradcam_grid(model, X_test, y_test, y_pred, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    indices = np.linspace(0, len(X_test) - 1, 8, dtype=int)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    for ax, idx in zip(axes.flat, indices):
        img = X_test[idx : idx + 1]
        heatmap = compute_gradcam(model, img)
        heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (28, 28)).numpy().squeeze()

        ax.imshow(img.squeeze(), cmap="gray")
        ax.imshow(heatmap_resized, cmap="jet", alpha=0.45)
        true_char = IDX_TO_CHAR[int(y_test[idx])]
        pred_char = IDX_TO_CHAR[int(y_pred[idx])]
        ax.set_title(f"True: {true_char} | Pred: {pred_char}", fontsize=10)
        ax.axis("off")

    plt.suptitle("Grad-CAM Model Explainability (Finger Articulation Focus)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    print("[INFO] Loading unseen test dataset and trained model...")
    X_test, y_test = load_csv_dataset(cfg["data"]["test_path"])
    model = tf.keras.models.load_model(cfg["paths"]["model_save_path"])

    # Benchmark inference latency
    _ = model.predict(X_test[:10], verbose=0)  # Warmup
    start_time = time.perf_counter()
    probs = model.predict(X_test, batch_size=64, verbose=0)
    total_time_ms = (time.perf_counter() - start_time) * 1000.0
    latency_per_sample_ms = total_time_ms / len(X_test)

    y_pred = np.argmax(probs, axis=1)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    target_names = [IDX_TO_CHAR[i] for i in range(cfg["data"]["num_classes"])]
    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    print("\n=== Classification Report (Test Set) ===")
    print(report)

    # Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    os.makedirs(os.path.dirname(cfg["paths"]["cm_plot_path"]), exist_ok=True)
    plt.figure(figsize=(14, 11))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.title(f"ASL Sign MNIST Confusion Matrix (Test Accuracy: {acc * 100:.2f}%)")
    plt.xlabel("Predicted Sign")
    plt.ylabel("Actual Sign")
    plt.tight_layout()
    plt.savefig(cfg["paths"]["cm_plot_path"], dpi=300)
    plt.close()

    # Save Grad-CAM visualization
    plot_gradcam_grid(model, X_test, y_test, y_pred, cfg["paths"]["gradcam_plot_path"])

    # Save summary metrics JSON
    metrics = {
        "test_samples": int(len(X_test)),
        "test_accuracy": round(float(acc), 4),
        "macro_f1_score": round(float(macro_f1), 4),
        "avg_inference_latency_ms": round(float(latency_per_sample_ms), 4),
    }
    with open(cfg["paths"]["metrics_json_path"], "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[INFO] Metrics saved: {metrics}")


if __name__ == "__main__":
    main()