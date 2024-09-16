import os
import time
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
import torch
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import ollama

import warnings

# Suppress only deprecation warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

def save_audio_as_mp3(audio_data, file_name):
    audio_segment = AudioSegment.from_wav(BytesIO(audio_data.get_wav_data()))
    mp3_path = os.path.join(TARGET_DIRECTORY, f"{file_name}.mp3")
    audio_segment.export(mp3_path, format="mp3")
    #print(f"Audio saved as: {mp3_path}")

def record_with_break(timeout, userInput, fileOut = False):
    # initialize recognizer
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        # adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)
        #print("Beginning to listen")

        while True:
            try:
                audio_data = recognizer.listen(source, timeout = timeout)
                current_time = time.strftime("%m_%d_%Y-%H_%M_%S")
                save_audio_as_mp3(audio_data, f"audio_{current_time}")

            except sr.WaitTimeoutError:
                #print("Break heard")
                continue

            for file_path in Path(TARGET_DIRECTORY).iterdir():
                if file_path.is_file():
                    sample = str(file_path)

                    result = pipe(sample, generate_kwargs = {"language": "en"})
                    #result = pipe(sample)
                    print(result["text"])

                    file_path.unlink()  # This deletes the file
                    #print(f"Deleted file: {file_path.name}")
            
            # switch case for the different features

            #print(userInput)

            match userInput:
                # voice transcribe
                case '1':
                    #print(result["text"])
                    if fileOut:
                        createDirectoryWithName("transcriptions")
                        current_time = time.strftime("%m_%d_%Y_Transcriptions")
                        textfile_path =  os.path.join("transcriptions", current_time)
                        with open(textfile_path, "a") as file:
                            file.write(result["text"] + "\n")
                        print("successfully written to file")
                
                # chatbot
                case '2':
                    stream = ollama.chat(
                    model='gemma2:27b',
                    messages=[{'role': 'user', 'content': result["text"]}],
                    stream=True,
                    )
                    
                    #print(response["message"]["content"])
                    
                    for chunk in stream:
                        print(chunk['message']['content'], end='', flush=True)

                    print("\n")

                # translation
                case '3':
                    #tbd
                    return

def createDirectoryWithName(name):
    # find current directory
    current_directory = os.getcwd()

    # Define the full path to the target directory
    target_dir_path = os.path.join(current_directory, name)

    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir_path):
        os.makedirs(target_dir_path)
        #print(f"Created directory: {target_dir_path}")
    #else:
        #print(f"Directory already exists: {target_dir_path}")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model = model,
    tokenizer = processor.tokenizer,
    feature_extractor = processor.feature_extractor,
    torch_dtype = torch_dtype,
    device = device,
)

#dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
#sample = dataset[0]["audio"]

TARGET_DIRECTORY = "audio_files"
createDirectoryWithName(TARGET_DIRECTORY)

while True:
    userInput = False
    #userInput = input("Hello there User, what feature would you like to use? The avaliable options are:\n1. Voice transcribing\n2. Chatbot\n3. Translation to another language\n") 
    userInput = input("Hello there User, what feature would you like to use? The avaliable options are:\n1. Voice transcribing\n2. Chatbot\n") 
    if userInput in "123":
        if userInput == "1":
            fileOut = False
            while True:
                fileOut = input("Would you like the transcription to be saved in a txt file? (Y/N)\n").lower()
                if fileOut in "yn":
                    if fileOut == "y":
                        fileOut == True
                    else:
                        fileOut == False
                    break
                else:
                    print("Sorry, that's not a valid input. Please input a choice between the provided choices.\n")

        else:
            """
            while True:
                shouldSpeak = input("Would you like the responses to be spoken back to you? (Y/N)\n").lower()
                if shouldSpeak in "yn":
                    if shouldSpeak == "y":
                        shouldSpeak == True
                    else:
                        shouldSpeak == False
                    break
                else:
                    print("Sorry, that's not a valid input. Please input a choice between the provided choices.\n")
            """
        break
    else:
        print("Sorry, that's not a valid input. Please input a choice between the provided choices.\n")

record_with_break(0.5, userInput, fileOut)
