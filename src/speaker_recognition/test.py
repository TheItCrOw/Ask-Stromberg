import tensorflow as tf
import numpy as np
import keras
import librosa
import os
# Indexes for the model:
# 0: Benjamin
# 1: Stromberg
# 2: Julia
# 3: Margaret
# 4: Nelson
model_path = "E:\\Python_Projects\\StrombergAI\\src\\speaker_recognition\\model\\stromberg-recognizer-model.keras"


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


def audio_to_logmel(audio, sample_rate=16000, n_mels=128):
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


if __name__ == "__main__":
    print("Loading the model...")
    model = keras.models.load_model(model_path)
    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    print("Model loaded")

    # We do the recognizition of the voices by folders and each folder is one episode
    episode_folder = "E:\\Python_Projects\\StrombergAI\\audio\\tests"
    audios = []
    directory = os.fsencode(episode_folder)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".wav"):
            # Here we load the audio file into a tf tensor
            full_path = os.path.join(episode_folder, filename)
            audio, sampling_rate = librosa.load(full_path, mono=True, sr=16000)
            audios.append((tf.convert_to_tensor(audio, dtype=tf.float32), filename))

    # Now let the model predict the speaker by its index.
    for audio_tensor, file_name in audios:
        # Add the batch and feature dimensions to the tensor
        audio_tensor = tf.expand_dims(audio_tensor, axis=0)  # Add batch dimension
        audio_tensor = tf.expand_dims(audio_tensor, axis=-1)  # Add feature dimension
        # Get the signal FFT
        ffts = audio_to_logmel(audio_tensor)
        # Predict
        y_pred = model.predict(ffts)
        print(file_name)
        print(np.argmax(y_pred))
        print(y_pred)
        print(y_pred.item((0, 0)))
        print("")
