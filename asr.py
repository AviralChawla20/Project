import torch
import re
import requests
import spacy
import speech_recognition as sr
from pydub import AudioSegment
from googletrans import Translator
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import asyncio

# Set device and data types
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")

# Load Whisper model and processor
model_id = "openai/whisper-medium"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype).to(
    device
)
processor = AutoProcessor.from_pretrained(model_id)

# Whisper ASR pipeline
whisper_asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
    torch_dtype=torch_dtype,
    generate_kwargs={"language": "<|hi|>", "task": "transcribe"},
)

# Whisper Translation pipeline
whisper_translation_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
    torch_dtype=torch_dtype,
    generate_kwargs={"task": "translate"},
)


def convert_audio_to_text(audio_file_path):
    recognizer = sr.Recognizer()

    if not audio_file_path.lower().endswith(".wav"):
        audio = AudioSegment.from_file(audio_file_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export("temp_audio.wav", format="wav")
        audio_file_path = "temp_audio.wav"

    with sr.AudioFile(audio_file_path) as source:
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.record(source)
        try:
            hindi_text = recognizer.recognize_google(audio_data, language="hi-IN")
            return hindi_text
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            return None


async def translate_text_to_english(hindi_text):
    if hindi_text:
        translator = Translator()
        try:
            translated = await asyncio.to_thread(
                translator.translate, hindi_text, src="hi", dest="en"
            )
            return translated.text
        except Exception as e:
            return None
    return None


def extract_important_information(text):
    if not text:
        return None

    doc = nlp(text)
    phone_pattern = r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    age_pattern = r"(\d{1,3})\s*(years old|age)"

    extracted_data = {
        "name": [ent.text for ent in doc.ents if ent.label_ == "PERSON"],
        "address": [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]],
        "phone_number": re.findall(phone_pattern, text),
        "email": re.findall(email_pattern, text),
        "age": [age[0] for age in re.findall(age_pattern, text)],
        "other_entities": [
            ent.text for ent in doc.ents if ent.label_ not in ["PERSON", "GPE", "LOC"]
        ],
    }
    return extracted_data


async def process_audio(audio_file_path):
    print("Processing with Whisper...")
    whisper_transcription = whisper_asr_pipeline(audio_file_path)["text"]
    whisper_translation = whisper_translation_pipeline(audio_file_path)["text"]

    print("Processing with SpeechRecognition...")
    speechrecognition_transcription = convert_audio_to_text(audio_file_path)
    speechrecognition_translation = await translate_text_to_english(
        speechrecognition_transcription
    )

    print("\n--- Whisper Transcription ---\n", whisper_transcription)
    print("\n--- Whisper Translation ---\n", whisper_translation)

    print(
        "\n--- SpeechRecognition Transcription ---\n", speechrecognition_transcription
    )
    print("\n--- SpeechRecognition Translation ---\n", speechrecognition_translation)

    important_info = extract_important_information(whisper_translation)
    print("\nExtracted Information:")
    for key, value in important_info.items():
        print(f"{key.capitalize()}: {value}")


# Run the processing function
audio_file_path = "audio.mp3"  # Replace with actual path
asyncio.run(process_audio(audio_file_path))
