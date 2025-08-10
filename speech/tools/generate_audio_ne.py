from gtts import gTTS
from pydub import AudioSegment
import os

# 10 Nepali scam phrases based on keywords
scam_phrases = [
    "ऋण हटाउनुहोस्",
    "मार्केटिङ",
    "100% निःशुल्क",
    "विशेष डील",
    "प्रारम्भिक लगानी आवश्यक",
    "द्रुत नगद",
    "जोखिममुक्त",
    "सन्तुष्टि ग्यारेन्टी",
    "आज निःशुल्क साइन अप गर्नुहोस्",
    "यसलाई अहिले प्राप्त गर्नुहोस्",
]

# 10 Nepali non-scam (normal) phrases
non_scam_phrases = [
    "नमस्ते",
    "तपाईं कसरी छन्?",
    "मलाई संगीत मन पर्छ",
    "आज राम्रो मौसम छ",
    "म विद्यालय जान्छु",
    "मसँग पाल्तु जनावर छ",
    "खाना स्वादिलो छ",
    "म नेपाली सिक्न चाहन्छु",
    "सिनेमा बन्द छ",
    "भोलि भेटौंला",
]

base_dir = os.path.dirname(os.path.dirname(__file__))
output_folder = os.path.join(base_dir, "datasets/ne/test_audio_nepali")
os.makedirs(output_folder, exist_ok=True)

def save_phrase_as_wav(phrase, filename):
    tts = gTTS(text=phrase, lang="ne")
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

print(f"Generated 10 scam and 10 non-scam Nepali WAV files in '{output_folder}' at 16kHz mono.")
