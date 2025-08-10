from gtts import gTTS
from pydub import AudioSegment
import os

# 10 Spanish scam phrases based on your keywords
scam_phrases = [
    "eliminar deuda",
    "marketing por correo",
    "100% gratis",
    "oferta exclusiva",
    "requiere inversión inicial",
    "dinero rápido",
    "sin riesgo",
    "satisfacción garantizada",
    "regístrate gratis hoy",
    "obténlo ahora"
]

# 10 Spanish non-scam (normal) phrases
non_scam_phrases = [
    "buenos días",
    "¿cómo estás?",
    "me gusta la música",
    "hace buen tiempo hoy",
    "voy a la escuela",
    "tengo una mascota",
    "la comida está deliciosa",
    "quiero aprender español",
    "el cine está cerrado",
    "nos vemos mañana"
]

base_dir = os.path.dirname(os.path.dirname(__file__))
output_folder = os.path.join(base_dir, "datasets/es/test_audio_spanish")
os.makedirs(output_folder, exist_ok=True)

def save_phrase_as_wav(phrase, filename):
    # Generate TTS audio (Spanish)
    tts = gTTS(text=phrase, lang="es")
    mp3_path = filename + ".mp3"
    wav_path = filename + ".wav"
    tts.save(mp3_path)
    
    # Convert MP3 to WAV with 16kHz sample rate and mono channel
    sound = AudioSegment.from_mp3(mp3_path)
    sound = sound.set_frame_rate(16000).set_channels(1)
    sound.export(wav_path, format="wav")
    
    # Delete the intermediate MP3 file
    os.remove(mp3_path)

# Generate scam WAV files
for i, phrase in enumerate(scam_phrases, start=1):
    filename = os.path.join(output_folder, f"scam_{i:02}")
    save_phrase_as_wav(phrase, filename)

# Generate non-scam WAV files
for i, phrase in enumerate(non_scam_phrases, start=1):
    filename = os.path.join(output_folder, f"nonscam_{i:02}")
    save_phrase_as_wav(phrase, filename)

print(f"Generated 10 scam and 10 non-scam Spanish WAV files in '{output_folder}' at 16kHz mono.")
