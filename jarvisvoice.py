from typing import Any, Optional
import sounddevice as sd
import numpy as np
from jarvis_tts.tts import StyleTTS2
import time

# Default natural language tokenizer table may need to be downloaded to venv before this will work.
# import nltk
# nltk.download("punkt_tab") 

# Samples per second
sps = 44100

# Frequency/Pitch
freq_hz = 440.0

# Duration
duration_s = 5.0

# Attenuation
atten = 0.3

class JarvisStyleTTS2(StyleTTS2):

    def __init__(self, device : str = "cpu"):
        super().__init__()
        self.device = device

class JarvisVoice:

    def __init__(self):
        self.tts : JarvisStyleTTS2 = JarvisStyleTTS2()

    def speakInference(self, text : str):
        return self.tts.inference(text=text, output_sample_rate = 24000, output_wav_file = "test.wav")

    def calcWaveForm(self, data, freq : float = 440.0, sps : int = 44100, atten : Optional[float] = None):
        waveform =  np.sin(2 * np.pi * freq * data / sps)
        #waveform : ndarray[Any] = sin(2 * pi * each_sample_number * freq / sps)
        if atten:
            return waveform * atten
        else:
            return waveform
    
    def preprocess(self, text : str) -> str:
        return text.replace("\n", "")

    def speak(self, text : str):
        audio = self.speakInference(self.preprocess(text))
        rev_audio = self.calcWaveForm(data = audio, freq = 1240.0, sps = 44100)
        sd.play(data = rev_audio, samplerate=24000, blocking=True)


jv = JarvisVoice()

jv.speak("Hello! It is so nice to finally meet you. My name is Mel, and I will be your assistant today.")
