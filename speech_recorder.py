import os
import time
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO

def save_audio_as_mp3(audio_data, file_name):
    audio_segment = AudioSegment.from_wav(BytesIO(audio_data.get_wav_data()))
    mp3_path = os.path.join(TARGET_DIRECTORY, f"{file_name}.mp3")
    audio_segment.export(mp3_path, format="mp3")
    print(f"Audio saved as: {mp3_path}")

def record_with_break(timeout = 2):
    # initialize recognizer
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        # adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)
        print("Beginning to listen")

        while True:
            try:
                audio_data = recognizer.listen(source, timeout = timeout)
                current_time = time.strftime("%m_%d_%Y-%H_%M_%S")
                save_audio_as_mp3(audio_data, f"audio_{current_time}")

            except sr.WaitTimeoutError:
                print("Break heard")
                continue


TARGET_DIRECTORY = "audio_files"

# find current directory
current_directory = os.getcwd()

# Define the full path to the target directory
target_dir_path = os.path.join(current_directory, TARGET_DIRECTORY)

# Create the target directory if it doesn't exist
if not os.path.exists(target_dir_path):
    os.makedirs(target_dir_path)
    print(f"Created directory: {target_dir_path}")
else:
    print(f"Directory already exists: {target_dir_path}")


record_with_break(timeout = 2)