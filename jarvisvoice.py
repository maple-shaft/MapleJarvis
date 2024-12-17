from typing import Any, Optional
import sounddevice as sd
import soundfile as sf
import numpy as np
import jarvis_tts.synthesize2 as s 
import time
import io
from pydub import AudioSegment

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

class JarvisVoice:

    def __init__(self):
        s.initialize()

    def speakInference(self, text : str):
        return s.infer_from_text(text)
        #return self.tts.inference(text=text, output_sample_rate = 24000, output_wav_file = "test.wav")

    def calcWaveForm(self, data, freq : float = 440.0, sps : int = 44100, atten : Optional[float] = None):
        waveform =  np.sin(2 * np.pi * freq * data / sps)
        #waveform : ndarray[Any] = sin(2 * pi * each_sample_number * freq / sps)
        if atten:
            return waveform * atten
        else:
            return waveform
    
    def preprocess(self, text : str) -> str:
        return text.replace("\n", "")
    
    def convert_to_ogg(self, audio_wav) -> bytes:
        buffer = io.BytesIO()
        try:
            # Normalize and convert NumPy array to int16 PCM
            audio_wav = (audio_wav * 32767).astype(np.int16).tobytes()
            # Create an in-memory buffer for raw audio
            raw_audio = io.BytesIO(audio_wav)
            # Convert raw audio into an AudioSegment
            audio_segment = AudioSegment.from_raw(raw_audio, sample_width=2, frame_rate=27100, channels=1)
            # Export to Opus format (WebM container) in memory
            audio_segment.export(buffer, format="webm", codec="libopus")
            return buffer.getvalue()
            # Write the NumPy array as OGG format to the buffer
        finally:
            buffer.close()

    def speak(self, text : str, play : bool = True) -> bytes | None:
        audio = self.speakInference(self.preprocess(text))
        rev_audio = self.calcWaveForm(data = audio, freq = 1240.0, sps = 44100)
        if play:
            sd.play(data = rev_audio, samplerate=24000, blocking=True)
        else:
            return self.convert_to_ogg(rev_audio)


#jv = JarvisVoice()

#jv.speak("Hello! It is so nice to finally meet you. My name is Mel, and I will be your assistant today.")
