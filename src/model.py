import tensorflow as tf
from tensorflow.keras import layers, models


def build_asl_cnn(input_shape=(28, 28, 1), num_classes=24, learning_rate=1e-3, augmenter=None):
    """
    Lightweight, high-accuracy Regularized CNN tailored for 28x28 grayscale hand gestures.
    Uses BatchNorm + Spatial Dropout + GlobalAveragePooling2D to prevent overfitting.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = augmenter(inputs) if augmenter is not None else inputs

    # Convolutional Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.20)(x)

    # Convolutional Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Convolutional Block 3 (Target layer for Grad-CAM)
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu", name="last_conv_activation")(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Classification Head
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.40)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ASL_SignMNIST_CNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model