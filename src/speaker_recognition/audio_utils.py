from pydub import AudioSegment
import math
import uuid
import numpy as np


class SplitWavAudioMubin():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + '\\' + filename
        self.uuid = str(uuid.uuid4())
        # Else: from_wav or from_mp3
        self.audio = AudioSegment.from_wav(self.filepath)

    def get_duration(self):
        return self.audio.duration_seconds

    def single_split(self, from_min, to_min, split_filename):
        t1 = from_min * 60 * 1000
        t2 = to_min * 60 * 1000
        split_audio = self.audio[t1:t2]
        # Here either 'mp3' or 'wav'
        split_audio.export(self.folder + '\\' + split_filename, format="wav")

    def multiple_split(self, min_per_split):
        counter = 0
        total_mins = math.ceil(self.get_duration() / 60)
        for i in np.arange(0, total_mins, min_per_split):
            split_fn = str(i) + '_' + self.filename
            self.single_split(i, i+min_per_split,
                              self.uuid + '_' + str(counter) + ".wav")
            print(str(i) + ' Done')
            print(str(self.uuid))
            counter = counter + 1
            if i == total_mins - min_per_split:
                print('All splited successfully')


folder = "E:\\Python_Projects\\StrombergAI\\audio\\other"
file = 'Die Büttenrede - Stromberg.wav'
split_wav = SplitWavAudioMubin(folder, file)
split_wav.multiple_split(min_per_split=0.0166667)
