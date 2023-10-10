import sys
import os
import stable_whisper
import librosa
import json
import traceback
from pathlib import Path
from pydub import AudioSegment
from moviepy.editor import *
from stromberg_annotator import StrombergAnnotator

MODEL_PATH = "E:\\Python_Projects\\StrombergAI\\src\\speaker_recognition\\model\\stromberg-recognizer-model.keras"
ROOT_PATH = 'E:\\Python_Projects\\StrombergAI'
AUDIO_FOLDER = 'audio'
VIDEO_FOLDER = 'videos'
TRANSCRIPT_FOLDER = 'transcripts'
AUDIO_SEGMENTS_FOLDER = 'audio_segments'
SAMPLING_RATE = 16000


def transcribe(audio_path):
    '''
    Extracts the text from a given audio file
    https://github.com/jianfch/stable-ts
    https://wandb.ai/wandb_fc/gentle-intros/reports/OpenAI-Whisper-How-to-Transcribe-Your-Audio-to-Text-for-Free-with-SRTs-VTTs---VmlldzozNDczNTI0
    '''
    model = stable_whisper.load_model("large-v2")
    result = model.transcribe(str(audio_path), regroup="sp=./?/!+1_sl=80", language="de", verbose=False, fp16=False)
    print(result.to_dict())
    with open("movie.json", "w", encoding='utf-8') as outfile:
        json.dump(result.to_dict(), outfile)
    # result.save_as_json('audio.json')


def extract_audio_from_video(video_path, audio_output_path):
    '''
    Gets the audio from a video file
    https://github.com/steadylearner/Python-Blog/blob/master/posts/Python/How%20to%20extract%20audio%20from%20a%20video%20with%20Python.md
    '''
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_output_path)


def read_whisper_result_from_file_as_dict(path):
    '''Reads a whisper result as json into a dict'''
    with open(path, 'r', encoding='utf-8') as j:
        result = json.loads(j.read())
    return result


def extract_segments_as_wav(episode_name):
    '''Loops through each segment of the transcript and cuts the audio'''
    transcript = read_whisper_result_from_file_as_dict(
        os.path.join(*[ROOT_PATH, TRANSCRIPT_FOLDER, episode_name, episode_name + '.json']))

    # Also load in the audio to that episode
    audio = AudioSegment.from_file(
        os.path.join(*[ROOT_PATH, AUDIO_FOLDER, episode_name, episode_name + '.wav']), format='wav')

    counter = 0
    for segment in transcript['segments']:
        # Get the start and end of that segment, export to audio as one clip
        start_time = int(segment['start'] * 1000)  # convert to milliseconds
        end_time = int((segment['end'] + 0.25) * 1000)  # convert to milliseconds
        # Extract the segment
        audio_segment = audio[start_time:end_time]
        # Our model works with 16k sampling_rate, so store them as such.
        audio_segment_16k = audio_segment.set_frame_rate(16000)
        # Save the audio segment
        file_name = str(counter) + '_' + str(segment['start']) + '_' + str(segment['end']) + '.wav'
        full_path = os.path.join(*[
            ROOT_PATH, AUDIO_SEGMENTS_FOLDER, episode_name, file_name
        ])
        Path(os.path.join(*[ROOT_PATH, AUDIO_SEGMENTS_FOLDER, episode_name])).mkdir(parents=True, exist_ok=True)
        # Store the audio file
        audio_segment_16k.export(full_path, format='wav')
        counter = counter + 1


def annotate_stromberg_episode(model, episode_name):
    '''Annotes all the segments of one stromberg episode'''
    # We do the recognizition of the voices by folders and each folder is one episode
    episode_folder = os.path.join(*[ROOT_PATH, AUDIO_SEGMENTS_FOLDER, episode_name])
    audios = []
    total = 0
    hits = 0
    directory = os.fsencode(episode_folder)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".wav"):
            # Here we load the audio file into a tf tensor
            full_path = os.path.join(episode_folder, filename)
            print("Doing segment " + full_path)
            audio, sampling_rate = librosa.load(full_path, mono=True, sr=SAMPLING_RATE)
            is_stromberg = model.is_stromberg(audio)
            if(is_stromberg):
                hits = hits + 1
            print("Segment is stromberg: " + str(is_stromberg))
            print("\n\n")
            total = total + 1
    print("\n\n\n Hits/Total: " + str(hits) + "/" + str(total))


if __name__ == "__main__":
    try:
        path = "E:\\Python_Projects\\StrombergAI\\videos\\the_movie.mkv"
        output = "E:\\Python_Projects\\StrombergAI\\audio\\movie\\movie.wav"
        # 1.
        # extract_audio_from_video(path, output)
        # 2.
        # transcribe(output)
        # 3.
        # extract_segments_as_wav('movie')
        print("Loading the stromberg recognizer model...")
        model = StrombergAnnotator(MODEL_PATH)
        print("Loaded model")
        annotate_stromberg_episode(model, 'movie')
    except Exception as e:
        print(str(e))
        traceback.print_exc()
