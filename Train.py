#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import os

# ================= CONFIG ================= #
DATASET_PATH = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_STAGE1 = 15
EPOCHS_STAGE2 = 25
os.makedirs("saved", exist_ok=True)
# ========================================== #


# ============= LOAD DATASET =============== #
def load_dataset():
    train_aug = ImageDataGenerator(
        rescale=1/255,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.25,
        horizontal_flip=True
    )

    base_aug = ImageDataGenerator(rescale=1/255)

    train = train_aug.flow_from_directory(
        f"{DATASET_PATH}/train", target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode='categorical'
    )

    val = base_aug.flow_from_directory(
        f"{DATASET_PATH}/validation", target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode='categorical'
    )

    test = base_aug.flow_from_directory(
        f"{DATASET_PATH}/test", target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
    )

    return train, val, test, len(train.class_indices)


# ============ BUILD MODEL ================= #
def build_model(n_class):
    base = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False,
        input_shape=(224,224,3)
    )
    base.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(n_class, activation="softmax")(x)

    model = tf.keras.Model(base.input, out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"]
    )
    return model, base


# =========== FINE TUNE ==================== #
def fine_tune(model, base, train, val):
    for layer in base.layers[:-70]: layer.trainable = False
    for layer in base.layers[-70:]: layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"]
    )

    cb2 = [
        ModelCheckpoint("saved/best_finetuned.keras", save_best_only=True, monitor="val_accuracy", verbose=1),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.4, min_lr=1e-6, verbose=1)
    ]

    return model.fit(train, validation_data=val, epochs=EPOCHS_STAGE2, callbacks=cb2)


# ============== PLOT ====================== #
def plot_graph(h1, h2):
    acc = np.concatenate([h1.history["accuracy"], h2.history["accuracy"]])
    val = np.concatenate([h1.history["val_accuracy"], h2.history["val_accuracy"]])
    loss = np.concatenate([h1.history["loss"], h2.history["loss"]])
    vlos = np.concatenate([h1.history["val_loss"], h2.history["val_loss"]])

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(acc); plt.plot(val)
    plt.title("Accuracy"); plt.legend(["Train","Val"])

    plt.subplot(1,2,2)
    plt.plot(loss); plt.plot(vlos)
    plt.title("Loss"); plt.legend(["Train","Val"])

    plt.tight_layout(); plt.show()


# ================ MAIN ==================== #
def main():
    train, val, test, num_class = load_dataset()
    print("\nClass mapping ->", train.class_indices)

    model, base = build_model(num_class)

    cb1 = [
        ModelCheckpoint("saved/best_stage1.keras", save_best_only=True, monitor="val_accuracy", verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]

    h1 = model.fit(train, validation_data=val, epochs=EPOCHS_STAGE1, callbacks=cb1)
    h2 = fine_tune(model, base, train, val)

    loss, acc = model.evaluate(test)
    print(f"\nFinal Test -> Acc: {acc:.4f} | Loss: {loss:.4f}")

    plot_graph(h1, h2)


if __name__ == "__main__":
    main()
