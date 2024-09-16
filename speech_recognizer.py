import torch
import time
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


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

TARGET_DIRECTORY = Path("audio_files")

while True:
#print(len(list(TARGET_DIRECTORY.iterdir())))
    for file_path in TARGET_DIRECTORY.iterdir():
        if file_path.is_file():
            sample = str(file_path)

            result = pipe(sample, generate_kwargs = {"language": "english"})
            print(result["text"])

            file_path.unlink()  # This deletes the file
            print(f"Deleted file: {file_path.name}")
    time.sleep(1)

"""
import speech_recognition as sr
import pyttsx3

#initialize recognizer
r = sr.Recognizer()

def record_text():
    while(1):
        try:
            with sr.Microphone() as source:
                # prepare microphone, adjust for ambient noise

                #r.adjust_for_ambient_noise(source, duration = 0.2)
                r.adjust_for_ambient_noise(source)

                # start listening to input
                audio = r.record(source, duration = 3)
                #audio = r.listen(source)

                # Use google to recognize audio
                #MyText = r.recognize_amazon(audio2)
                MyText = r.recognize_google(audio, language='en-USA')

                return MyText

        except sr.RequestError as e:
            print(f"Request error; {e}")

        except sr.UnknownValueError:
            print("Unknown error")

    return

def output_text(text):
    return

while(1):
    text = record_text()
    #print("hello there")
    print(text)
"""