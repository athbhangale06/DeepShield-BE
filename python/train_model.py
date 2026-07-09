"""
DEEPSHIELD Model Training Script
==================================
Trains a CNN on the CIFAKE dataset to detect AI-generated images.
Saves the model as deepshield_model.h5 in the project root.

Usage:
    python train_model.py

The CIFAKE dataset should be at ../../cifake/ with structure:
    cifake/train/FAKE/  (50,000 images)
    cifake/train/REAL/  (50,000 images)
    cifake/test/FAKE/   (10,000 images)
    cifake/test/REAL/   (10,000 images)
"""

import os
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# ============================================================
# Configuration
# ============================================================
IMG_SIZE = (128, 128)
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
TRAIN_DIR = os.path.join(PROJECT_ROOT, 'cifake', 'train')
TEST_DIR = os.path.join(PROJECT_ROOT, 'cifake', 'test')
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'deepshield_model.h5')


def check_gpu():
    """Check and report GPU availability."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  ✅ GPU detected: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"     - {gpu.name}")
        # Allow memory growth to prevent OOM
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        return True
    else:
        print("  ⚠️  No GPU detected. Training on CPU (will be slower).")
        return False


def build_model(input_shape=(128, 128, 3)):
    """
    Build a CNN model for binary classification (REAL vs FAKE).
    Uses a custom architecture with BatchNorm and Dropout for robustness.
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 4
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Classifier
        layers.Flatten(),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary: 0=REAL, 1=FAKE
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_data_generators():
    """Create training and validation data generators with augmentation."""

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.15,
        shear_range=0.1,
        fill_mode='nearest',
        validation_split=0.1  # Use 10% of training data for validation
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    print(f"\n  📂 Loading training data from: {TRAIN_DIR}")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['REAL', 'FAKE'],  # 0=REAL, 1=FAKE
        subset='training',
        shuffle=True
    )

    print(f"  📂 Loading validation data from: {TRAIN_DIR} (10% split)")
    val_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['REAL', 'FAKE'],
        subset='validation',
        shuffle=False
    )

    print(f"  📂 Loading test data from: {TEST_DIR}")
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['REAL', 'FAKE'],
        shuffle=False
    )

    print(f"\n  Class mapping: {train_generator.class_indices}")
    print(f"  Training samples:   {train_generator.samples}")
    print(f"  Validation samples: {val_generator.samples}")
    print(f"  Test samples:       {test_generator.samples}")

    return train_generator, val_generator, test_generator


def train():
    """Main training function."""
    print("\n" + "=" * 60)
    print("  🛡️  DEEPSHIELD Model Training")
    print("=" * 60)

    # Check GPU
    has_gpu = check_gpu()

    # Verify dataset
    if not os.path.exists(TRAIN_DIR):
        print(f"\n  ❌ Training directory not found: {TRAIN_DIR}")
        sys.exit(1)
    if not os.path.exists(TEST_DIR):
        print(f"\n  ❌ Test directory not found: {TEST_DIR}")
        sys.exit(1)

    # Create data generators
    train_gen, val_gen, test_gen = create_data_generators()

    # Build model
    print("\n  🏗️  Building CNN model...")
    model = build_model()
    model.summary()

    # Callbacks
    training_callbacks = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train
    print("\n  🚀 Starting training...")
    print(f"  Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE} | Image Size: {IMG_SIZE}")
    print(f"  Device: {'GPU' if has_gpu else 'CPU'}")
    print("-" * 60)

    start_time = time.time()

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=training_callbacks,
        verbose=1
    )

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    # Evaluate on test set
    print("\n" + "-" * 60)
    print("  📊 Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=0)

    # Print results
    print("\n" + "=" * 60)
    print("  📈 Training Results")
    print("=" * 60)
    print(f"  Training time:      {minutes}m {seconds}s")
    print(f"  Best val accuracy:  {max(history.history['val_accuracy']) * 100:.2f}%")
    print(f"  Test accuracy:      {test_accuracy * 100:.2f}%")
    print(f"  Test loss:          {test_loss:.4f}")
    print(f"  Model saved to:     {MODEL_SAVE_PATH}")
    print(f"  Model size:         {os.path.getsize(MODEL_SAVE_PATH) / (1024*1024):.1f} MB")
    print("=" * 60)

    # Quick prediction test
    print("\n  🔍 Quick prediction test...")
    test_batch = next(iter(test_gen))
    images, labels = test_batch
    preds = model.predict(images[:5], verbose=0)
    for i in range(5):
        actual = "FAKE" if labels[i] == 1 else "REAL"
        predicted = "FAKE" if preds[i][0] > 0.5 else "REAL"
        conf = preds[i][0] if preds[i][0] > 0.5 else 1 - preds[i][0]
        status = "✅" if actual == predicted else "❌"
        print(f"    {status} Actual: {actual:4s} | Predicted: {predicted:4s} | Confidence: {conf*100:.1f}%")

    print("\n  ✅ Training complete!")
    return model


if __name__ == '__main__':
    train()
