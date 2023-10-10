# See: https://github.com/keras-team/keras-io/blob/master/examples/audio/speaker_recognition_using_cnn.py
# https://keras.io/examples/audio/speaker_recognition_using_cnn/

# We train a model to recognize Strombergs voice within the episodes.
# Maybe later we will also recognize different voices, we will see.
import os
import shutil
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
from tensorflow import keras
from pathlib import Path
from IPython.display import display, Audio

# ========================= Create some parameters
DATASET_ROOT = "E:\\Python_Projects\\StrombergAI\\src\\speaker_recognition\\datasets"

# The folders in which we will put the audio samples and the noise samples
AUDIO_SUBFOLDER = "audio"
NOISE_SUBFOLDER = "noise"

DATASET_AUDIO_PATH = os.path.join(DATASET_ROOT, AUDIO_SUBFOLDER)
DATASET_NOISE_PATH = os.path.join(DATASET_ROOT, NOISE_SUBFOLDER)

# Percentage of samples to use for validation
VALID_SPLIT = 0.1

# Seed to use when shuffling the dataset and the noise
SHUFFLE_SEED = 43

# The sampling rate to use.
# This is the one used in all the audio samples.
# We will resample all the noise to this sampling rate.
# This will also be the output size of the audio wave samples
# (since all samples are of 1 second long)
SAMPLING_RATE = 16000

# The factor to multiply the noise with according to:
#   noisy_sample = sample + noise * prop * scale
#      where prop = sample_amplitude / noise_amplitude
SCALE = 0.5

BATCH_SIZE = 128
EPOCHS = 100
# The amount of mel frequency bands or bins.
MEL_BINS = 128
# ========================= End parameters

# ========================= Preprare audio files
# Split noise into chunks of 16000 each


def load_noise_sample(path):
    sample, sampling_rate = tf.audio.decode_wav(
        tf.io.read_file(path), desired_channels=1
    )
    if sampling_rate == SAMPLING_RATE:
        # Number of slices of 16000 each that can be generated from the noise sample
        slices = int(sample.shape[0] / SAMPLING_RATE)
        sample = tf.split(sample[: slices * SAMPLING_RATE], slices)
        return sample
    else:
        print("Sampling rate for {} is incorrect. Ignoring it".format(path))
        return None

# ========================= End Preprare audio files

# ========================= Start dataset generation


def paths_and_labels_to_dataset(audio_paths, labels):
    """Constructs a dataset of audios and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(
        lambda x: path_to_audio(x), num_parallel_calls=tf.data.AUTOTUNE
    )
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds)) 


def path_to_audio(path):
    """Reads and decodes an audio file."""
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
    return audio


def add_noise(audio, noises=None, scale=0.5):
    if noises is not None:
        # Create a random tensor of the same size as audio ranging from
        # 0 to the number of noise stream samples that we have.
        tf_rnd = tf.random.uniform(
            (tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32
        )
        noise = tf.gather(noises, tf_rnd, axis=0)

        # Get the amplitude proportion between the audio and the noise
        prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
        prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)

        # Adding the rescaled noise to audio
        audio = audio + noise * prop * scale

    return audio


def audio_to_logmel(audio, sample_rate=SAMPLING_RATE, n_mels=MEL_BINS):
    # Compute STFT
    audio = tf.squeeze(audio, axis=-1)
    stft = tf.signal.stft(audio,
                          frame_length=400,  # window length. The FFT will be done on windowed frames of this length.
                          frame_step=160,    # frame step. This is the distance to slide along the window.
                          fft_length=512)    # fft length. This parameter determines the number length of the FFT. 
    spectrogram = tf.abs(stft)
    # Compute mel spectrogram: Create mel filter and apply it
    num_spectrogram_bins = stft.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=0.0,
        upper_edge_hertz=sample_rate // 2)
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)

    # Compute log mel spectrogram
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    # Add a channel dimension
    log_mel_spectrogram = tf.expand_dims(log_mel_spectrogram, -1)
    return log_mel_spectrogram


def audio_to_fft(audio):
    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again
    # after FFT
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)

    # Return the absolute value of the first half of the FFT
    # which represents the positive frequencies
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])
# ========================= End dataset generation

# ========================= Start model definition


def residual_block(x, filters, conv_num=3, activation="relu"):
    # Shortcut
    s = keras.layers.Conv2D(filters, (1, 1), padding="same")(x)
    for i in range(conv_num - 1):
        x = keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
        x = keras.layers.Activation(activation)(x)
    x = keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = keras.layers.Add()([x, s])
    x = keras.layers.Activation(activation)(x)
    return keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)


def build_model(input_shape, num_classes):
    inputs = keras.layers.Input(shape=input_shape, name="input")

    x = residual_block(inputs, 16, 2)
    # x = residual_block(x, 32, 2)
    # x = residual_block(x, 64, 3)
    # x = residual_block(x, 128, 3)
    # x = residual_block(x, 128, 3)

    # Convert the 2D outputs (height x width x channels) from CNN to sequences for RNN
    x = keras.layers.Reshape((-1, x.shape[-1]))(x)

    # Recurrent layers for analyzing sequences of extracted features (RNN)
    # x = keras.layers.LSTM(128, return_sequences=True)(x)
    x = keras.layers.LSTM(128)(x)

    # x = keras.layers.AveragePooling2D(pool_size=(3, 3), strides=3)(x)
    # x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)

    # outputs = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    return keras.models.Model(inputs=inputs, outputs=outputs)

# ========================= End model definition


if __name__ == "__main__":
    try:
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        # Prepare the noise files
        noise_paths = []
        for subdir in os.listdir(DATASET_NOISE_PATH):
            subdir_path = Path(DATASET_NOISE_PATH) / subdir
            if os.path.isdir(subdir_path):
                noise_paths += [
                    os.path.join(subdir_path, filepath)
                    for filepath in os.listdir(subdir_path)
                    if filepath.endswith(".wav")
                ]

        print(
            "Found {} files belonging to {} directories".format(
                len(noise_paths), len(os.listdir(DATASET_NOISE_PATH))
            )
        )

        # Downsample the audio to 16khz. The old audio is overwritten!!
        for path in noise_paths:
            y, s = librosa.load(path, sr=SAMPLING_RATE)  # Downsample to 16kHz
            sf.write(path, y, s)

        # Now divide the 16khz audio into 1 seconds each
        noises = []
        for path in noise_paths:
            sample = load_noise_sample(path)
            if sample:
                noises.extend(sample)
        noises = tf.stack(noises)

        print(
            "{} noise files were split into {} noise samples where each is {} sec. long".format(
                len(noise_paths), noises.shape[0], noises.shape[1] // SAMPLING_RATE
            )
        )
        # ===========================================================
        # Now we are ready to create the datasets from the audio files
        # Get the list of audio file paths along with their corresponding labels
        class_names = os.listdir(DATASET_AUDIO_PATH)
        print("Our class names: {}".format(class_names,))

        audio_paths = []
        labels = []
        for label, name in enumerate(class_names):
            print("Processing speaker {}".format(name,))
            dir_path = Path(DATASET_AUDIO_PATH) / name

            speaker_sample_paths = [
                os.path.join(dir_path, filepath)
                for filepath in os.listdir(dir_path)
                if filepath.endswith(".wav")
            ]
            # Lets make sure that the audio is indeed 16k sampled
            # Overwrites the existing audio files!
            for sample_path in speaker_sample_paths:
                y, s = librosa.load(os.path.join(dir_path, sample_path), sr=SAMPLING_RATE)  # Downsample to 16kHz
                sf.write(os.path.join(dir_path, sample_path), y, s)

            audio_paths += speaker_sample_paths
            labels += [label] * len(speaker_sample_paths)

        print(
            "Found {} files belonging to {} classes.".format(len(audio_paths), len(class_names))
        )

        # Shuffle
        rng = np.random.RandomState(SHUFFLE_SEED)
        rng.shuffle(audio_paths)
        rng = np.random.RandomState(SHUFFLE_SEED)
        rng.shuffle(labels)

        # Split into training and validation
        num_val_samples = int(VALID_SPLIT * len(audio_paths))
        print("Using {} files for training.".format(len(audio_paths) - num_val_samples))
        train_audio_paths = audio_paths[:-num_val_samples]
        train_labels = labels[:-num_val_samples]

        print("Using {} files for validation.".format(num_val_samples))
        valid_audio_paths = audio_paths[-num_val_samples:]
        valid_labels = labels[-num_val_samples:]

        # Create 2 datasets, one for training and the other for validation
        train_ds = paths_and_labels_to_dataset(train_audio_paths, train_labels)
        train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(
            BATCH_SIZE
        )

        valid_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
        valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(32)

        # Add noise to the training set
        train_ds = train_ds.map(
            lambda x, y: (add_noise(x, noises, scale=SCALE), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Transform audio wave to the frequency domain using `audio_to_fft`
        train_ds = train_ds.map(
            lambda x, y: (audio_to_logmel(x, SAMPLING_RATE), y), num_parallel_calls=tf.data.AUTOTUNE
        )
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        valid_ds = valid_ds.map(
            lambda x, y: (audio_to_logmel(x, SAMPLING_RATE), y), num_parallel_calls=tf.data.AUTOTUNE
        )
        valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

        # ===========================================================
        # Now we define the model
        # This is the amount of frames our logmel function produces.
        # This is dervied from the Short-time Fourir transform and is constant 
        # for our specific parameters..
        time_frames = 98
        model = build_model((time_frames, MEL_BINS, 1), len(class_names))
        model.summary()

        # Compile the model using Adam's default learning rate
        # loss="sparse_categorical_crossentropy"
        model.compile(
            optimizer="Adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        # Add callbacks:
        # 'EarlyStopping' to stop training when the model is not enhancing anymore
        # 'ModelCheckPoint' to always keep the model that has the best val_accuracy
        model_save_filename = "stromberg-recognizer-model.h5"

        earlystopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
            model_save_filename, monitor="val_accuracy", save_best_only=True
        )

        # ===========================================================
        # Training
        # We want to give a bias to Not_Stromberg classes since that has to
        # detect multiple other speakers, while Stromberg only detects one.
        class_weights = {0: 2., 1: 1.}

        history = model.fit(
            train_ds,
            epochs=EPOCHS,
            # class_weight=class_weights,
            validation_data=valid_ds,
            callbacks=[earlystopping_cb, mdlcheckpoint_cb],
        )
        # Save the model
        model.save("stromberg-recognizer-model.keras")
        # Show the results
        print(model.evaluate(valid_ds))
    except Exception as e:
        print(str(e))
