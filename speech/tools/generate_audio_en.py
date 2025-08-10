from gtts import gTTS
from pydub import AudioSegment
import os

# 10 English scam phrases based on keywords
scam_phrases = [
    "eliminate debt",
    "marketing by email",
    "100% free",
    "exclusive deal",
    "requires initial investment",
    "fast cash",
    "risk free",
    "satisfaction guaranteed",
    "sign up free today",
    "get it now",
]

# 10 English non-scam (normal) phrases
non_scam_phrases = [
    "hello",
    "how are you?",
    "I like music",
    "it's nice weather today",
    "I'm going to school",
    "I have a pet",
    "the food is delicious",
    "I want to learn English",
    "the cinema is closed",
    "see you tomorrow",
]

base_dir = os.path.dirname(os.path.dirname(__file__))
output_folder = os.path.join(base_dir, "datasets/en/test_audio_english")
os.makedirs(output_folder, exist_ok=True)

def save_phrase_as_wav(phrase, filename):
    tts = gTTS(text=phrase, lang="en")
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

print(f"Generated 10 scam and 10 non-scam English WAV files in '{output_folder}' at 16kHz mono.")
