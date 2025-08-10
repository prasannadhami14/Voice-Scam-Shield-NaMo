from gtts import gTTS
from pydub import AudioSegment
import os

# 10 French scam phrases based on keywords
scam_phrases = [
    "éliminer la dette",
    "marketing par e-mail",
    "100% gratuit",
    "offre exclusive",
    "exige un investissement initial",
    "argent rapide",
    "sans risque",
    "satisfaction garantie",
    "inscrivez-vous gratuitement aujourd'hui",
    "obtenez-le maintenant",
]

# 10 French non-scam (normal) phrases
non_scam_phrases = [
    "bonjour",
    "comment ça va?",
    "j'aime la musique",
    "il fait beau aujourd'hui",
    "je vais à l'école",
    "j'ai un animal de compagnie",
    "la nourriture est délicieuse",
    "je veux apprendre le français",
    "le cinéma est fermé",
    "à demain",
]

base_dir = os.path.dirname(os.path.dirname(__file__))
output_folder = os.path.join(base_dir, "datasets/fr/test_audio_french")
os.makedirs(output_folder, exist_ok=True)

def save_phrase_as_wav(phrase, filename):
    tts = gTTS(text=phrase, lang="fr")
    mp3_path = filename + ".mp3"
    wav_path = filename + ".wav"
    tts.save(mp3_path)
    sound = AudioSegment.from_mp3(mp3_path)
    sound = sound.set_frame_rate(16000).set_channels(1)
    sound.export(wav_path, format="wav")
    os.remove(mp3_path)

for i, phrase in enumerate(scam_phrases, start=1):
    filename = os.path.join(output_folder, f"scam_{i:02}")
    save_phrase_as_wav(phrase, filename)

for i, phrase in enumerate(non_scam_phrases, start=1):
    filename = os.path.join(output_folder, f"non_scam_{i:02}")
    save_phrase_as_wav(phrase, filename)

print(f"Generated 10 scam and 10 non-scam French WAV files in '{output_folder}' at 16kHz mono.")


