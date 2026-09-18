"""COMP3710 Lab 2 solution runner.

Run one section at a time, for example:
    python lab2_solution.py --part dft
    python lab2_solution.py --part lfw
    python lab2_solution.py --part unet --data-root "C:/path/to/keras_png_slices_data/keras_png_slices_data"

Training sections intentionally require an explicit command-line choice.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def square_wave(t: np.ndarray, f0: float = 1.0) -> np.ndarray:
    """Return a unit square wave sampled at times t."""
    return np.sign(np.sin(2.0 * np.pi * f0 * t)).astype(np.float32)


def square_wave_fourier(t: np.ndarray, f0: float, harmonics: int) -> np.ndarray:
    """Reconstruct a square wave from its first odd harmonics."""
    result = np.zeros_like(t, dtype=np.float32)
    for harmonic_index in range(harmonics):
        harmonic = 2 * harmonic_index + 1
        result += np.sin(2 * np.pi * harmonic * f0 * t) / harmonic
    return (4 / np.pi) * result


def naive_dft(signal: np.ndarray) -> np.ndarray:
    """Compute a direct O(N^2) DFT for a one-dimensional signal."""
    signal = np.asarray(signal, dtype=np.complex128)
    sample_count = signal.size
    result = np.zeros(sample_count, dtype=np.complex128)
    for frequency_bin in range(sample_count):
        for sample_index in range(sample_count):
            result[frequency_bin] += signal[sample_index] * np.exp(
                -2j * np.pi * frequency_bin * sample_index / sample_count
            )
    return result


def run_dft() -> None:
    import matplotlib.pyplot as plt

    sample_count = 2048
    duration = 1.0
    frequency = 1.0
    times = np.linspace(0.0, duration, sample_count, endpoint=False)
    original = square_wave(times, frequency)

    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].plot(times, original, color="black")
    axes[0].set_title("Original")
    for axis, harmonic_count in zip(axes[1:], (3, 20, 50)):
        axis.plot(times, square_wave_fourier(times, frequency, harmonic_count))
        axis.plot(times, original, "k--", alpha=0.45)
        axis.set_title(f"{harmonic_count} harmonics")
    for axis in axes:
        axis.set_ylim(-1.5, 1.5)
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    plt.show()

    signal = square_wave_fourier(times, frequency, 50)
    start = time.perf_counter()
    direct_result = naive_dft(signal)
    naive_seconds = time.perf_counter() - start
    start = time.perf_counter()
    fft_result = np.fft.fft(signal)
    fft_seconds = time.perf_counter() - start
    print(f"Naive DFT: {naive_seconds:.6f} s")
    print(f"NumPy FFT: {fft_seconds:.6f} s")
    print(f"Results agree: {np.allclose(direct_result, fft_result, atol=1e-5)}")

    frequencies = np.fft.fftfreq(sample_count, d=duration / sample_count)
    positive = frequencies[: sample_count // 2]
    magnitude = 2.0 / sample_count * np.abs(direct_result[: sample_count // 2])
    plt.figure(figsize=(12, 4))
    plt.stem(positive, magnitude, basefmt=" ")
    plt.xlim(0, 50)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Square-wave DFT magnitude")
    plt.grid(True, alpha=0.3)
    plt.show()


def run_lfw(epochs: int) -> None:
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_lfw_people
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    import tensorflow as tf

    faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    images = faces.images.astype("float32")
    labels = faces.target
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.25, random_state=42, stratify=labels
    )
    height, width = images.shape[1:]

    pca = PCA(n_components=min(150, x_train.shape[0] - 1), svd_solver="randomized", random_state=42)
    x_train_pca = pca.fit_transform(x_train.reshape(len(x_train), -1))
    x_test_pca = pca.transform(x_test.reshape(len(x_test), -1))
    forest = RandomForestClassifier(
        n_estimators=150, max_depth=15, max_features="sqrt", random_state=42, n_jobs=-1
    )
    forest.fit(x_train_pca, y_train)
    forest_predictions = forest.predict(x_test_pca)
    print(f"PCA + Random Forest accuracy: {accuracy_score(y_test, forest_predictions):.4f}")
    print(classification_report(y_test, forest_predictions, target_names=faces.target_names, zero_division=0))

    x_train_cnn = x_train[..., np.newaxis]
    x_test_cnn = x_test[..., np.newaxis]
    class_count = len(faces.target_names)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((height, width, 1)),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(class_count, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    history = model.fit(x_train_cnn, y_train, validation_split=0.15, epochs=epochs, batch_size=32)
    print("CNN test metrics:", model.evaluate(x_test_cnn, y_test, verbose=0))
    plt.plot(history.history["accuracy"], label="training")
    plt.plot(history.history["val_accuracy"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def load_png_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image

    image_dir = root / f"keras_png_slices_{split}"
    mask_dir = root / f"keras_png_slices_seg_{split}"
    image_paths = sorted(image_dir.glob("case_*.png"))
    images = []
    masks = []
    for image_path in image_paths:
        mask_path = mask_dir / image_path.name.replace("case_", "seg_", 1)
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}")
        images.append(np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0)
        masks.append(np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0)
    if not images:
        raise FileNotFoundError(f"No case_*.png images found in {image_dir}")
    return np.asarray(images)[..., np.newaxis], np.asarray(masks)[..., np.newaxis]


def dice_coefficient(y_true, y_pred):
    import tensorflow as tf

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    axes = (1, 2, 3)
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    denominator = tf.reduce_sum(y_true + y_pred, axis=axes)
    return tf.reduce_mean((2.0 * intersection + 1e-6) / (denominator + 1e-6))


def unet_model(input_shape=(256, 256, 1)):
    import tensorflow as tf

    inputs = tf.keras.Input(input_shape)

    def block(tensor, filters):
        tensor = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(tensor)
        tensor = tf.keras.layers.BatchNormalization()(tensor)
        tensor = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(tensor)
        return tensor

    encoder1 = block(inputs, 32)
    pooled1 = tf.keras.layers.MaxPooling2D()(encoder1)
    encoder2 = block(pooled1, 64)
    pooled2 = tf.keras.layers.MaxPooling2D()(encoder2)
    bottleneck = block(pooled2, 128)
    decoder2 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding="same")(bottleneck)
    decoder2 = tf.keras.layers.Concatenate()([decoder2, encoder2])
    decoder2 = block(decoder2, 64)
    decoder1 = tf.keras.layers.Conv2DTranspose(32, 2, strides=2, padding="same")(decoder2)
    decoder1 = tf.keras.layers.Concatenate()([decoder1, encoder1])
    decoder1 = block(decoder1, 32)
    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(decoder1)
    return tf.keras.Model(inputs, outputs)


def run_unet(data_root: Path, epochs: int) -> None:
    import matplotlib.pyplot as plt
    import tensorflow as tf

    x_train, y_train = load_png_split(data_root, "train")
    x_validate, y_validate = load_png_split(data_root, "validate")
    x_test, y_test = load_png_split(data_root, "test")
    print("Train/validate/test:", x_train.shape, x_validate.shape, x_test.shape)

    model = unet_model(x_train.shape[1:])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=[dice_coefficient])
    model.summary()
    history = model.fit(
        x_train, y_train, validation_data=(x_validate, y_validate), epochs=epochs, batch_size=16
    )
    print("Test metrics:", model.evaluate(x_test, y_test, verbose=0))
    predictions = model.predict(x_test[:4], verbose=0)
    figure, axes = plt.subplots(4, 3, figsize=(9, 12))
    for row in range(4):
        axes[row, 0].imshow(x_test[row, ..., 0], cmap="gray")
        axes[row, 1].imshow(y_test[row, ..., 0], cmap="gray")
        axes[row, 2].imshow(predictions[row, ..., 0] > 0.5, cmap="gray")
        for column in range(3):
            axes[row, column].axis("off")
    axes[0, 0].set_title("Image")
    axes[0, 1].set_title("Ground truth")
    axes[0, 2].set_title("Prediction")
    figure.tight_layout()
    plt.show()
    plt.plot(history.history["dice_coefficient"], label="training")
    plt.plot(history.history["val_dice_coefficient"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def naive_dft_tf(signal, device: str = "/CPU:0"):
    """O(N^2) DFT expressed as a TF tensor matmul (DFT matrix x signal), runnable on CPU or GPU."""
    import tensorflow as tf

    with tf.device(device):
        signal_c = tf.cast(tf.reshape(signal, (-1, 1)), tf.complex64)
        n = tf.shape(signal)[0]
        n_float = tf.cast(n, tf.float32)
        k = tf.reshape(tf.range(n, dtype=tf.float32), (-1, 1))
        sample_index = tf.reshape(tf.range(n, dtype=tf.float32), (1, -1))
        angle = -2.0 * np.pi * k * sample_index / n_float
        basis = tf.complex(tf.cos(angle), tf.sin(angle))
        result = tf.linalg.matmul(basis, signal_c)
        return tf.reshape(result, (-1,))


def run_dft_tensor() -> None:
    """Compare CPU-tensor, GPU-tensor, and built-in FFT DFT timings across signal sizes."""
    import tensorflow as tf

    gpu_available = bool(tf.config.list_physical_devices("GPU"))
    if not gpu_available:
        print("No GPU detected on this machine; GPU timings must be measured on Rangpur.")

    for size in (256, 512, 1024, 2048):
        times = np.linspace(0.0, 1.0, size, endpoint=False).astype(np.float32)
        signal = square_wave_fourier(times, 1.0, 50)
        signal_tf = tf.constant(signal, dtype=tf.float32)

        start = time.perf_counter()
        naive_dft_tf(signal_tf, device="/CPU:0").numpy()
        cpu_seconds = time.perf_counter() - start

        if gpu_available:
            start = time.perf_counter()
            naive_dft_tf(signal_tf, device="/GPU:0").numpy()
            gpu_seconds = time.perf_counter() - start
        else:
            gpu_seconds = float("nan")

        start = time.perf_counter()
        tf.signal.fft(tf.cast(signal_tf, tf.complex64)).numpy()
        fft_seconds = time.perf_counter() - start

        print(
            f"N={size}: CPU tensor naive DFT {cpu_seconds:.4f}s | "
            f"GPU tensor naive DFT {gpu_seconds:.4f}s | tf.signal.fft {fft_seconds:.4f}s"
        )


def run_eigenfaces_manual() -> None:
    """Task-sheet-exact pipeline: manual SVD PCA, eigenface gallery, compactness plot, RF(max_features=150)."""
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_lfw_people
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    n_samples, h, w = lfw_people.images.shape
    x = lfw_people.data
    y = lfw_people.target
    target_names = lfw_people.target_names

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    n_components = 150
    mean = np.mean(x_train, axis=0)
    x_train_centered = x_train - mean
    x_test_centered = x_test - mean

    _, s, v = np.linalg.svd(x_train_centered, full_matrices=False)
    components = v[:n_components]
    eigenfaces = components.reshape((n_components, h, w))

    x_train_transformed = np.dot(x_train_centered, components.T)
    x_test_transformed = np.dot(x_test_centered, components.T)

    def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
        plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
        for i in range(n_row * n_col):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            plt.title(titles[i], size=12)
            plt.xticks(())
            plt.yticks(())

    plot_gallery(eigenfaces, [f"eigenface {i}" for i in range(eigenfaces.shape[0])], h, w)
    plt.show()

    explained_variance = (s ** 2) / (n_samples - 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()
    ratio_cumsum = np.cumsum(explained_variance_ratio)
    plt.plot(np.arange(n_components), ratio_cumsum[:n_components])
    plt.title("Compactness")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.grid(True, alpha=0.3)
    plt.show()

    estimator = RandomForestClassifier(n_estimators=150, max_depth=15, max_features=150, random_state=42)
    estimator.fit(x_train_transformed, y_train)
    predictions = estimator.predict(x_test_transformed)
    correct = predictions == y_test
    print("Total Testing:", len(x_test_transformed))
    print("Total Correct:", np.sum(correct))
    print("Accuracy:", np.sum(correct) / len(x_test_transformed))
    print(classification_report(y_test, predictions, target_names=target_names, zero_division=0))


def build_resnet18(input_shape=(32, 32, 3), num_classes: int = 10):
    """CIFAR-style ResNet-18: 3x3 stem (no initial maxpool), 4 stages of 2 residual blocks."""
    import tensorflow as tf

    def block(x, filters, stride=1):
        shortcut = x
        y = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.Activation("relu")(y)
        y = tf.keras.layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False)(y)
        y = tf.keras.layers.BatchNormalization()(y)
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        y = tf.keras.layers.Add()([shortcut, y])
        return tf.keras.layers.Activation("relu")(y)

    inputs = tf.keras.Input(input_shape)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    filters = 64
    for stage_index, stride in enumerate((1, 2, 2, 2)):
        for block_index in range(2):
            x = block(x, filters, stride=stride if block_index == 0 else 1)
        filters *= 2
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return tf.keras.Model(inputs, outputs)


def run_dawnbench(epochs: int, mixed_precision: bool) -> None:
    """CIFAR-10 ResNet-18 training/timing for the DAWNBench-style requirement (final numbers must come from Rangpur)."""
    import tensorflow as tf

    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    model = build_resnet18((32, 32, 3), 10)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    start = time.perf_counter()
    model.fit(x_train, y_train, validation_split=0.1, epochs=epochs, batch_size=128)
    elapsed = time.perf_counter() - start
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Training time: {elapsed:.1f} s")
    print(f"Test accuracy: {test_accuracy:.4f}, test loss: {test_loss:.4f}")
    print("Note: the >90%/94% timing targets must be measured on Rangpur's A100 GPU, not this machine.")


def load_image_folder(root: Path, split: str) -> np.ndarray:
    """Load a folder of grayscale PNGs (no masks needed) for VAE/GAN training."""
    from PIL import Image

    image_dir = root / f"keras_png_slices_{split}"
    image_paths = sorted(image_dir.glob("case_*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No case_*.png images found in {image_dir}")
    images = [np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0 for path in image_paths]
    return np.asarray(images)[..., np.newaxis]


def image_paths_for_split(root: Path, split: str) -> list[Path]:
    image_dir = root / f"keras_png_slices_{split}"
    image_paths = sorted(image_dir.glob("case_*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No case_*.png images found in {image_dir}")
    return image_paths


def build_vae(input_shape=(128, 128, 1), latent_dim: int = 2):
    import tensorflow as tf

    encoder_inputs = tf.keras.Input(input_shape)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(encoder_inputs)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    shape_before_flatten = x.shape[1:]
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")

    latent_inputs = tf.keras.Input((latent_dim,))
    x = tf.keras.layers.Dense(int(np.prod(shape_before_flatten)), activation="relu")(latent_inputs)
    x = tf.keras.layers.Reshape(shape_before_flatten)(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    decoder_outputs = tf.keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
    decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return encoder, decoder


def run_vae(data_root: Path, epochs: int, latent_dim: int = 2, output_dir: Path = Path("demo_outputs/vae")) -> None:
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    train_paths = image_paths_for_split(data_root, "train")
    with Image.open(train_paths[0]) as first_image:
        input_shape = (first_image.height, first_image.width, 1)
    encoder, decoder = build_vae(input_shape, latent_dim)
    optimizer = tf.keras.optimizers.Adam(1e-3)

    def load_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=1)
        return tf.cast(image, tf.float32) / 255.0

    dataset = (
        tf.data.Dataset.from_tensor_slices([str(path) for path in train_paths])
        .shuffle(min(1024, len(train_paths)))
        .map(load_image, num_parallel_calls=1)
        .batch(8)
        .prefetch(1)
    )

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = encoder(batch, training=True)
            epsilon = tf.random.normal(tf.shape(z_mean))
            z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
            reconstruction = decoder(z, training=True)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.keras.losses.binary_crossentropy(batch, reconstruction), axis=(1, 2))
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            loss = reconstruction_loss + kl_loss
        trainable_vars = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))
        return loss

    loss_history = []
    for epoch in range(epochs):
        epoch_loss = tf.keras.metrics.Mean()
        for batch in dataset:
            epoch_loss.update_state(train_step(batch))
        loss_value = float(epoch_loss.result())
        loss_history.append(loss_value)
        print(f"Epoch {epoch + 1}/{epochs}: loss={loss_value:.4f}")

    np.savetxt(output_dir / "vae_loss.csv", np.asarray(loss_history), delimiter=",", header="loss", comments="")
    encoder.save(output_dir / "encoder.keras")
    decoder.save(output_dir / "decoder.keras")

    if latent_dim == 2:
        grid_size = 15
        canvas = np.zeros((grid_size * input_shape[0], grid_size * input_shape[1]))
        for i, yi in enumerate(np.linspace(-2.5, 2.5, grid_size)):
            for j, xi in enumerate(np.linspace(-2.5, 2.5, grid_size)):
                decoded = decoder.predict(np.array([[xi, yi]], dtype=np.float32), verbose=0)[0, ..., 0]
                canvas[
                    i * input_shape[0] : (i + 1) * input_shape[0],
                    j * input_shape[1] : (j + 1) * input_shape[1],
                ] = decoded
        plt.figure(figsize=(10, 10))
        plt.imshow(canvas, cmap="gray")
        plt.title("VAE latent manifold (2D grid sampling)")
        plt.axis("off")
        plt.savefig(output_dir / "vae_latent_manifold.png", dpi=160, bbox_inches="tight")
        plt.show()
    else:
        test_paths = image_paths_for_split(data_root, "test")
        test_dataset = tf.data.Dataset.from_tensor_slices([str(path) for path in test_paths]).map(
            load_image, num_parallel_calls=1
        ).batch(8)
        z_mean_test = encoder.predict(test_dataset, verbose=0)[0]
        try:
            import importlib

            umap = importlib.import_module("umap")

            projected = umap.UMAP(n_components=2, random_state=42).fit_transform(z_mean_test)
            method = "UMAP"
        except ImportError:
            from sklearn.decomposition import PCA

            projected = PCA(n_components=2, random_state=42).fit_transform(z_mean_test)
            method = "PCA (umap-learn not installed)"
        plt.figure(figsize=(8, 8))
        plt.scatter(projected[:, 0], projected[:, 1], s=5, alpha=0.6)
        plt.title(f"VAE latent space projected with {method}")
        plt.savefig(output_dir / "vae_latent_projection.png", dpi=160, bbox_inches="tight")
        plt.show()


def _remap_mask_to_labels(mask_array: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert an 8-bit grayscale mask (e.g. values 0/85/170/255) into integer class labels 0..num_classes-1."""
    step = 255.0 / (num_classes - 1)
    return np.round(mask_array / step).astype(np.int32)


def load_oasis_multiclass_split(root: Path, split: str, num_classes: int = 4) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image

    image_dir = root / f"keras_png_slices_{split}"
    mask_dir = root / f"keras_png_slices_seg_{split}"
    image_paths = sorted(image_dir.glob("case_*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No case_*.png images found in {image_dir}")
    images = []
    labels = []
    for image_path in image_paths:
        mask_path = mask_dir / image_path.name.replace("case_", "seg_", 1)
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}")
        images.append(np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0)
        mask_array = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32)
        labels.append(_remap_mask_to_labels(mask_array, num_classes))
    return np.asarray(images)[..., np.newaxis], np.asarray(labels)


def unet_model_multiclass(input_shape=(256, 256, 1), num_classes: int = 4):
    import tensorflow as tf

    inputs = tf.keras.Input(input_shape)

    def block(tensor, filters):
        tensor = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(tensor)
        tensor = tf.keras.layers.BatchNormalization()(tensor)
        tensor = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(tensor)
        return tensor

    encoder1 = block(inputs, 32)
    pooled1 = tf.keras.layers.MaxPooling2D()(encoder1)
    encoder2 = block(pooled1, 64)
    pooled2 = tf.keras.layers.MaxPooling2D()(encoder2)
    bottleneck = block(pooled2, 128)
    decoder2 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding="same")(bottleneck)
    decoder2 = tf.keras.layers.Concatenate()([decoder2, encoder2])
    decoder2 = block(decoder2, 64)
    decoder1 = tf.keras.layers.Conv2DTranspose(32, 2, strides=2, padding="same")(decoder2)
    decoder1 = tf.keras.layers.Concatenate()([decoder1, encoder1])
    decoder1 = block(decoder1, 32)
    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation="softmax")(decoder1)
    return tf.keras.Model(inputs, outputs)


def discrete_dice_per_label(y_true_labels: np.ndarray, y_pred_labels: np.ndarray, label: int) -> float:
    """Standard discrete DSC for one class: 2*|A∩B| / (|A|+|B|) on hard argmax labels."""
    true_mask = y_true_labels == label
    pred_mask = y_pred_labels == label
    denominator = true_mask.sum() + pred_mask.sum()
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(true_mask, pred_mask).sum() / denominator)


def multiclass_dice_loss(y_true, y_pred):
    import tensorflow as tf

    smooth = tf.constant(1e-6, dtype=tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2))
    dice_by_class = (2.0 * intersection + smooth) / (denominator + smooth)
    foreground_dice = dice_by_class[:, 1:]
    return 1.0 - tf.reduce_mean(foreground_dice)


def weighted_categorical_crossentropy(class_weights):
    import tensorflow as tf

    weights = tf.constant(class_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        pixel_weights = tf.reduce_sum(y_true * weights, axis=-1)
        cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        return tf.reduce_mean(cross_entropy * pixel_weights, axis=(1, 2))

    return loss


def run_oasis_unet(
    data_root: Path, epochs: int, num_classes: int = 4, output_dir: Path = Path("demo_outputs/oasis_unet")
) -> None:
    import matplotlib.pyplot as plt
    import tensorflow as tf

    output_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train = load_oasis_multiclass_split(data_root, "train", num_classes)
    x_validate, y_validate = load_oasis_multiclass_split(data_root, "validate", num_classes)
    x_test, y_test = load_oasis_multiclass_split(data_root, "test", num_classes)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(8).map(
        lambda images, labels: (images, tf.one_hot(labels, num_classes)), num_parallel_calls=1
    )
    validate_dataset = tf.data.Dataset.from_tensor_slices((x_validate, y_validate)).batch(8).map(
        lambda images, labels: (images, tf.one_hot(labels, num_classes)), num_parallel_calls=1
    )

    model = unet_model_multiclass(x_train.shape[1:], num_classes)
    class_pixel_counts = np.bincount(y_train.reshape(-1), minlength=num_classes).astype(np.float64)
    inverse_frequency = class_pixel_counts.sum() / (num_classes * np.maximum(class_pixel_counts, 1.0))
    class_weights = np.sqrt(inverse_frequency / inverse_frequency.mean())
    class_weights[0] = 0.25

    def combined_loss(y_true, y_pred):
        weighted_cross_entropy = weighted_categorical_crossentropy(class_weights)(y_true, y_pred)
        return 0.5 * weighted_cross_entropy + multiclass_dice_loss(y_true, y_pred)

    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss=combined_loss, metrics=["accuracy"])
    model.summary()
    checkpoint_path = output_dir / "best_oasis_unet.weights.h5"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, monitor="val_loss", save_best_only=True, save_weights_only=True
    )
    history = model.fit(
        train_dataset, validation_data=validate_dataset, epochs=epochs, callbacks=[checkpoint_callback]
    )

    model.load_weights(checkpoint_path)
    model.save(output_dir / "oasis_unet.keras")
    print("Discrete test DSC per foreground label:")
    dice_scores = {label: [] for label in range(1, num_classes)}
    preview_predictions = []
    for start in range(0, len(x_test), 2):
        batch_predictions = np.argmax(model.predict(x_test[start : start + 2], verbose=0), axis=-1)
        preview_predictions.extend(batch_predictions[: max(0, 4 - len(preview_predictions))])
        for offset, prediction in enumerate(batch_predictions):
            sample_index = start + offset
            for label in range(1, num_classes):
                dice_scores[label].append(discrete_dice_per_label(y_test[sample_index], prediction, label))

    mean_dice_scores = {}
    for label in range(1, num_classes):
        mean_dice_scores[label] = float(np.mean(dice_scores[label]))
        print(f"  Label {label}: mean DSC = {mean_dice_scores[label]:.4f}")

    (output_dir / "test_dsc.txt").write_text(
        "\n".join(f"Label {label}: mean DSC = {score:.4f}" for label, score in mean_dice_scores.items()),
        encoding="utf-8",
    )
    np.savetxt(output_dir / "unet_training_loss.csv", np.asarray(history.history["loss"]), delimiter=",", header="loss", comments="")

    figure, axes = plt.subplots(4, 3, figsize=(9, 12))
    for row in range(min(4, len(x_test))):
        axes[row, 0].imshow(x_test[row, ..., 0], cmap="gray")
        axes[row, 1].imshow(y_test[row], cmap="viridis", vmin=0, vmax=num_classes - 1)
        axes[row, 2].imshow(preview_predictions[row], cmap="viridis", vmin=0, vmax=num_classes - 1)
        for column in range(3):
            axes[row, column].axis("off")
    axes[0, 0].set_title("Image")
    axes[0, 1].set_title("Ground truth labels")
    axes[0, 2].set_title("Predicted labels")
    figure.tight_layout()
    figure.savefig(output_dir / "segmentation_predictions.png", dpi=160, bbox_inches="tight")
    plt.show()


def build_gan_generator(latent_dim: int = 128, output_shape=(128, 128, 1)):
    import tensorflow as tf

    start_size = output_shape[0] // 4
    inputs = tf.keras.Input((latent_dim,))
    x = tf.keras.layers.Dense(start_size * start_size * 128, use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Reshape((start_size, start_size, 128))(x)
    x = tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2DTranspose(32, 4, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    outputs = tf.keras.layers.Conv2D(output_shape[-1], 5, padding="same", activation="tanh")(x)
    return tf.keras.Model(inputs, outputs, name="generator")


def build_gan_discriminator(input_shape=(128, 128, 1)):
    import tensorflow as tf

    inputs = tf.keras.Input(input_shape)
    x = tf.keras.layers.Conv2D(32, 4, strides=2, padding="same")(inputs)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Conv2D(64, 4, strides=2, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="discriminator")


def run_gan(
    data_root: Path,
    epochs: int,
    latent_dim: int = 128,
    checkpoint_dir: str = "gan_checkpoints",
    output_dir: Path = Path("demo_outputs/gan"),
) -> None:
    import matplotlib.pyplot as plt
    import tensorflow as tf

    output_dir.mkdir(parents=True, exist_ok=True)
    x_train = load_image_folder(data_root, "train") * 2.0 - 1.0  # scale to [-1, 1] for tanh output
    input_shape = x_train.shape[1:]

    generator = build_gan_generator(latent_dim, input_shape)
    discriminator = build_gan_discriminator(input_shape)
    bce = tf.keras.losses.BinaryCrossentropy()
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)
    checkpoint = tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
    )
    fixed_noise = tf.random.normal((16, latent_dim), seed=42)
    dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1024).batch(32, drop_remainder=True)
    generator_losses = []
    discriminator_losses = []

    @tf.function
    def train_step(real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal((batch_size, latent_dim))
        with tf.GradientTape() as discriminator_tape, tf.GradientTape() as generator_tape:
            generated_images = generator(noise, training=True)
            real_output = discriminator(real_images, training=True)
            fake_output = discriminator(generated_images, training=True)
            discriminator_loss = bce(tf.ones_like(real_output), real_output) + bce(
                tf.zeros_like(fake_output), fake_output
            )
            generator_loss = bce(tf.ones_like(fake_output), fake_output)
        discriminator_gradients = discriminator_tape.gradient(discriminator_loss, discriminator.trainable_variables)
        generator_gradients = generator_tape.gradient(generator_loss, generator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        return generator_loss, discriminator_loss

    for epoch in range(epochs):
        generator_loss_metric = tf.keras.metrics.Mean()
        discriminator_loss_metric = tf.keras.metrics.Mean()
        for batch in dataset:
            g_loss, d_loss = train_step(batch)
            generator_loss_metric.update_state(g_loss)
            discriminator_loss_metric.update_state(d_loss)
        print(
            f"Epoch {epoch + 1}/{epochs}: generator_loss={generator_loss_metric.result():.4f}, "
            f"discriminator_loss={discriminator_loss_metric.result():.4f}"
        )
        generator_losses.append(float(generator_loss_metric.result()))
        discriminator_losses.append(float(discriminator_loss_metric.result()))
        epoch_samples = (generator(fixed_noise, training=False).numpy() + 1.0) / 2.0
        epoch_figure, epoch_axes = plt.subplots(4, 4, figsize=(8, 8))
        for index, axis in enumerate(epoch_axes.flat):
            axis.imshow(epoch_samples[index, ..., 0], cmap="gray", vmin=0.0, vmax=1.0)
            axis.axis("off")
        epoch_figure.suptitle(f"Generated brain slices, epoch {epoch + 1}")
        epoch_figure.savefig(output_dir / f"generated_epoch_{epoch + 1:03d}.png", dpi=140, bbox_inches="tight")
        plt.close(epoch_figure)
        checkpoint.save(str(checkpoint_path / "ckpt"))  # keep a checkpoint each epoch in case of later collapse

    np.savetxt(
        output_dir / "gan_losses.csv",
        np.column_stack((generator_losses, discriminator_losses)),
        delimiter=",",
        header="generator_loss,discriminator_loss",
        comments="",
    )
    loss_figure = plt.figure(figsize=(8, 4))
    plt.plot(generator_losses, label="generator")
    plt.plot(discriminator_losses, label="discriminator")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    loss_figure.tight_layout()
    loss_figure.savefig(output_dir / "gan_loss_curves.png", dpi=160, bbox_inches="tight")

    samples = (generator(fixed_noise, training=False).numpy() + 1.0) / 2.0
    figure, axes = plt.subplots(4, 4, figsize=(8, 8))
    for index, axis in enumerate(axes.flat):
        axis.imshow(samples[index, ..., 0], cmap="gray")
        axis.axis("off")
    figure.suptitle("Generated 2D brain slices")
    figure.savefig(output_dir / "generated_final.png", dpi=160, bbox_inches="tight")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--part",
        choices=(
            "dft",
            "dft-tensor",
            "lfw",
            "eigenfaces",
            "unet",
            "dawnbench",
            "vae",
            "oasis-unet",
            "gan",
        ),
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("demo_outputs"))
    args = parser.parse_args()

    if args.part == "dft":
        run_dft()
    elif args.part == "dft-tensor":
        run_dft_tensor()
    elif args.part == "lfw":
        run_lfw(args.epochs)
    elif args.part == "eigenfaces":
        run_eigenfaces_manual()
    elif args.part == "dawnbench":
        run_dawnbench(args.epochs, args.mixed_precision)
    elif args.part in ("unet", "vae", "oasis-unet", "gan") and args.data_root is None:
        parser.error(f"--data-root is required for --part {args.part}")
    elif args.part == "unet":
        run_unet(args.data_root, args.epochs)
    elif args.part == "vae":
        run_vae(args.data_root, args.epochs, args.latent_dim, args.output_dir / "vae")
    elif args.part == "oasis-unet":
        run_oasis_unet(args.data_root, args.epochs, args.num_classes, args.output_dir / "oasis_unet")
    else:
        run_gan(args.data_root, args.epochs, args.latent_dim, output_dir=args.output_dir / "gan")


if __name__ == "__main__":
    main()
