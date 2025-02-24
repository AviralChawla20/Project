import torch
import re
import spacy
import speech_recognition as sr
from pydub import AudioSegment
from translate import Translator
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import logging
import warnings

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set device and data types
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
logging.info(f"Using device: {device}")

# Load spaCy NER model
try:
    nlp = spacy.load("en_core_web_sm")
    logging.info("Successfully loaded spaCy model")
except Exception as e:
    logging.error(f"Error loading spaCy model: {e}")
    raise

# Load Whisper model and processor
try:
    model_id = "openai/whisper-medium"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    logging.info("Successfully loaded Whisper model and processor")
except Exception as e:
    logging.error(f"Error loading Whisper model: {e}")
    raise

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
    
    try:
        logging.info(f"Processing audio file: {audio_file_path}")
        
        if not audio_file_path.lower().endswith(".wav"):
            logging.info("Converting audio to WAV format...")
            audio = AudioSegment.from_file(audio_file_path)
            logging.info(f"Original audio properties: {audio.channels} channels, {audio.frame_rate} Hz")
            
            audio = audio.set_channels(1).set_frame_rate(16000)
            logging.info("Audio converted to: 1 channel, 16000 Hz")
            
            audio.export("temp_audio.wav", format="wav")
            audio_file_path = "temp_audio.wav"
            logging.info("Conversion complete: saved as temp_audio.wav")
        
        with sr.AudioFile(audio_file_path) as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            logging.info("Recording audio data...")
            audio_data = recognizer.record(source)
            
            logging.info("Attempting speech recognition...")
            hindi_text = recognizer.recognize_google(audio_data, language="hi-IN")
            logging.info(f"Recognition successful: {hindi_text}")
            return hindi_text
            
    except sr.UnknownValueError:
        logging.error("Speech Recognition could not understand the audio")
        logging.error("Possible issues:\n- Audio quality too low\n- No speech detected\n- Language mismatch")
        return None
    except sr.RequestError as e:
        logging.error(f"Could not request results from Speech Recognition service; {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during audio processing: {type(e).__name__}")
        logging.error(f"Error details: {str(e)}")
        return None

def translate_text_to_english(hindi_text):
    if hindi_text:
        try:
            logging.info("Attempting to translate text...")
            translator = Translator(to_lang="en", from_lang="hi")
            translated = translator.translate(hindi_text)
            logging.info(f"Translation successful: {translated}")
            return translated
        except Exception as e:
            logging.error(f"Translation error: {e}")
            return None
    return None

def extract_important_information(text):
    if not text:
        logging.warning("No text provided for information extraction")
        return None

    logging.info("Extracting important information from text...")
    doc = nlp(text)
    phone_pattern = r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    age_pattern = r"(\d{1,3})\s*(years old|age)"
    
    # First find all age numbers to exclude them from general number extraction
    age_matches = re.finditer(age_pattern, text)
    age_numbers = set()
    for match in age_matches:
        age_numbers.add(match.group(1))
    
    # Pattern to match standalone numbers while excluding phone numbers
    number_pattern = r'\b(?!(?:\d{3}[-.\s]?)?\d{3}[-.\s]?\d{4}\b)\d+(?:\.\d+)?\b'
    
    # Find all numbers that aren't ages or phone numbers
    all_numbers = re.findall(number_pattern, text)
    numbers = [num for num in all_numbers if num not in age_numbers]

    extracted_data = {
        "name": [ent.text for ent in doc.ents if ent.label_ == "PERSON"],
        "address": [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]],
        "phone_number": re.findall(phone_pattern, text),
        "email": re.findall(email_pattern, text),
        "age": list(age_numbers),
        "numbers": numbers,
        "other_entities": [
            ent.text for ent in doc.ents 
            if ent.label_ not in ["PERSON", "GPE", "LOC", "CARDINAL", "QUANTITY"] 
            and not any(char.isdigit() for char in ent.text)
        ]
    }
    logging.info("Information extraction complete")
    return extracted_data

def process_audio(audio_file_path):
    logging.info("Starting audio processing...")
    
    try:
        logging.info("Processing with Whisper...")
        whisper_transcription = whisper_asr_pipeline(audio_file_path)["text"]
        whisper_translation = whisper_translation_pipeline(audio_file_path)["text"]
        
        logging.info("Processing with SpeechRecognition...")
        speechrecognition_transcription = convert_audio_to_text(audio_file_path)
        speechrecognition_translation = translate_text_to_english(speechrecognition_transcription)

        # Print results
        print("\n=== Results ===")
        print("\nWhisper Results:")
        print(f"Transcription: {whisper_transcription}")
        print(f"Translation: {whisper_translation}")
        
        print("\nSpeechRecognition Results:")
        print(f"Transcription: {speechrecognition_transcription}")
        print(f"Translation: {speechrecognition_translation}")

        if whisper_translation:
            important_info = extract_important_information(whisper_translation)
            print("\nExtracted Information:")
            for key, value in important_info.items():
                print(f"{key.capitalize()}: {value}")
        
        return {
            "whisper": {
                "transcription": whisper_transcription,
                "translation": whisper_translation
            },
            "speech_recognition": {
                "transcription": speechrecognition_transcription,
                "translation": speechrecognition_translation
            }
        }
        
    except Exception as e:
        logging.error(f"Error in process_audio: {str(e)}")
        return None

if __name__ == "__main__":
    # Replace with your audio file path
    audio_file_path = "testaudio.mp3"
    
    try:
        results = process_audio(audio_file_path)
        if results:
            logging.info("Audio processing completed successfully")
        else:
            logging.error("Audio processing failed")
    except Exception as e:
        logging.error(f"Main execution error: {str(e)}")