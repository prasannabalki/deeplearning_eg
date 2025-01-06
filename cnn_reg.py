import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt

def generate_data(num_samples=1000, img_height=64, img_width=64):
    """
    Generate synthetic grayscale images of shape (num_samples, 64, 64, 1)
    and random continuous targets in the range [0, 100]..
    """
    # Create random images
    X = np.random.rand(num_samples, img_height, img_width, 1).astype(np.float32)
    # Create random regression targets
    y = 100.0 * np.random.rand(num_samples).astype(np.float32)
    return X, y

def build_regression_cnn(input_shape=(64,64,1)):
    """
    Build a CNN for regression with:
    - 2 conv+pool blocks
    - L2 regularization in conv and dense layers
    - Dropout in the fully connected part
    - Single output for regression (MSE as loss)
    """
    model = models.Sequential()

    # 1) Convolutional block #1
    model.add(layers.Conv2D(
        filters=16, kernel_size=(3,3), activation='relu',
        kernel_regularizer=regularizers.l2(0.001),  # L2 on conv weights
        input_shape=input_shape
    ))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    # 2) Convolutional block #2
    model.add(layers.Conv2D(
        filters=32, kernel_size=(3,3), activation='relu',
        kernel_regularizer=regularizers.l2(0.001)   # L2 on conv weights
    ))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    # 3) Flatten
    model.add(layers.Flatten())

    # 4) Dense layer (with L2 + Dropout)
    model.add(layers.Dense(64, activation='relu',
                           kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.3))  # dropout for regularization

    # 5) Final output layer for regression: 1 unit
    model.add(layers.Dense(1, activation=None))

    # Compile the model (MSE loss, Adam optimizer, track MAE and MSE)
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    return model

def plot_history(history):
    """
    Plot training/validation Loss (MSE) and MAE over epochs.
    """
    plt.figure(figsize=(12,4))

    # Plot MSE (loss)
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Val Loss (MSE)')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

    # Plot MAE
    plt.subplot(1,2,2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('MAE Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    # 1) Generate synthetic data
    num_samples = 1000
    X, y = generate_data(num_samples=num_samples, img_height=64, img_width=64)

    # 2) Split into train/test (80:20)
    train_size = int(num_samples * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    print(f"Train set: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set:  X={X_test.shape}, y={y_test.shape}")

    # 3) Build the model
    model = build_regression_cnn(input_shape=(64,64,1))
    model.summary()

    # 4) Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32,
        verbose=1
    )

    # 5) Evaluate on the test set
    test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}")

    # 6) Plot training history
    plot_history(history)

if __name__ == '__main__':
    main()
