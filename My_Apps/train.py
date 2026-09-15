import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

PROCESSED_DIR = "data/processed"
MODEL_SAVE_PATH = "digit_model.keras"
IMG_SIZE = (32, 32)
BATCH_SIZE = 64
EPOCHS = 15


def load_custom_dataset():
    """Loads custom processed images from data/processed/<label>/*.png"""
    images = []
    labels = []

    if not os.path.exists(PROCESSED_DIR):
        print(f"ℹ️ Directory '{PROCESSED_DIR}' not found. Training on MNIST only.")
        return None, None

    for label in range(10):
        label_dir = os.path.join(PROCESSED_DIR, str(label))
        if os.path.exists(label_dir):
            image_paths = glob.glob(os.path.join(label_dir, "*.png"))
            for img_path in image_paths:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(label)

    if len(images) == 0:
        print("ℹ️ No custom images found in class subfolders. Training on MNIST only.")
        return None, None

    print(f"✅ Loaded {len(images)} custom samples from '{PROCESSED_DIR}'.")
    return np.array(images), np.array(labels)


def prepare_datasets():
    # 1. Load MNIST base dataset
    print("📥 Loading MNIST dataset...")
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()

    # 2. Resize MNIST images from 28x28 to 32x32 to match target size
    x_train_resized = np.array([cv2.resize(img, IMG_SIZE) for img in x_train_mnist])
    x_test_resized = np.array([cv2.resize(img, IMG_SIZE) for img in x_test_mnist])

    # 3. Load custom images if available
    x_custom, y_custom = load_custom_dataset()

    if x_custom is not None:
        # Merge MNIST with custom datasets
        x_train = np.concatenate((x_train_resized, x_custom), axis=0)
        y_train = np.concatenate((y_train_mnist, y_custom), axis=0)
    else:
        x_train, y_train = x_train_resized, y_train_mnist

    x_test, y_test = x_test_resized, y_test_mnist

    # 4. Normalize pixel values to range [0, 1] and add channel dimension
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return x_train, y_train, x_test, y_test


def build_cnn_model():
    """Builds a CNN architecture tailored for 32x32 digit recognition."""
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Dense Classifier Block
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    x_train, y_train, x_test, y_test = prepare_datasets()

    print(f"📊 Training shape: {x_train.shape}, Labels shape: {y_train.shape}")
    print(f"📊 Testing shape: {x_test.shape}, Labels shape: {y_test.shape}")

    # Data Augmentation to handle varying handwriting angles and camera zoom
    datagen = ImageDataGenerator(
        rotation_range=12,
        zoom_range=0.12,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    datagen.fit(x_train)

    # Build and summarize model
    model = build_cnn_model()
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True)
    ]

    print("\n🚀 Starting model training...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        callbacks=callbacks
    )

    # Final evaluation
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n✨ Training Complete!")
    print(f"🎯 Final Test Accuracy: {test_acc * 100:.2f}%")
    print(f"💾 Model saved to '{MODEL_SAVE_PATH}'")


if __name__ == "__main__":
    main()