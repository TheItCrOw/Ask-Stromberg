import tensorflow as tf
import numpy as np
import soundfile as sf
import keras
from pathlib import Path
import librosa
import os
import uuid

# The annotator only works with audio files with a sampling rate of this.
SAMPLING_RATE = 16000
MEL_BINS = 128
# Not sure if I will keep this...
DATASET_NOISE_PATH = 'E:\\Python_Projects\\StrombergAI\\src\\speaker_recognition\\datasets\\noise'

# This is the min value our model has to achieve in order to annotate an audio
# from stromberg.
MIN_VALUE = 0.5


# The class which wraps the model trained to recognize stromberg audios.
class StrombergAnnotator:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)
        # 0: Stromberg, 1: Others!
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

        # Now divide the 16khz audio into 1 seconds each
        self.noises = []
        for path in noise_paths:
            sample = self.load_noise_sample(path)
            if sample:
                self.noises.extend(sample)
        self.noises = tf.stack(self.noises)

    def load_noise_sample(self, path):
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

    def audio_to_fft(self, audio):
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

    def audio_to_logmel(self, audio, sample_rate=SAMPLING_RATE, n_mels=MEL_BINS):
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

    def is_stromberg(self, audio):
        '''This returns true if the given audio is stromberg'''
        # Lets check the length of the segment. The model is trained to idenfity
        # 1s clips, not longer!
        duration = librosa.get_duration(y=audio, sr=SAMPLING_RATE)
        total_clips = len(audio) // SAMPLING_RATE
        if(total_clips == 0):
            return False

        total = 0
        values = 0
        for i in range(total_clips):
            start = i * SAMPLING_RATE
            end = (i + 1) * SAMPLING_RATE

            # Extract the one sec clip
            clip = audio[start:end]
            # sf.write('test/' + str(uuid.uuid4()) + '_test_clippp.wav', clip, 16000)
            # Add a bit of noise like we did in the training. Maybe delete this.
            noise = np.random.normal(loc=0, scale=0.005, size=clip.shape)
            noisy_clip = clip + noise
            # Make sure the values are within [-1, 1] after adding the noise
            noisy_clip = np.clip(noisy_clip, -1, 1)

            audio_tensor = tf.convert_to_tensor(clip, dtype=tf.float32)
            # Add the batch and feature dimensions to the tensor
            audio_tensor = tf.expand_dims(audio_tensor, axis=0)  # Add batch dimension
            audio_tensor = tf.expand_dims(audio_tensor, axis=-1)  # Add feature dimension
            # Get the signal FFT
            ffts = self.audio_to_logmel(audio_tensor)
            # Predict
            y_pred = self.model.predict(ffts)
            print(y_pred)
            index = np.argmax(y_pred)
            val = y_pred.item((0, 0))
            print(val)
            if(val <= MIN_VALUE):
                total = total + 1
            values = values + val
            total = total + index

        return total == total_clips
