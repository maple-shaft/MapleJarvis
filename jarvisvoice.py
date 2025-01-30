from typing import Any, Optional
import sounddevice as sd
import soundfile as sf
import numpy as np
import numpy.typing as npt
import jarvis_tts.synthesize2 as s 
import io
from pydub import AudioSegment

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

    def _split_string_into_chunks_OLD(self, string, chunk_size) -> list[str]:
        """Splits a string into equal-sized chunks."""
        return [string[i:i+chunk_size] for i in range(0, len(string), chunk_size)]
    
    def _split_string_into_chunks(self, text : str, chunk_size : int) -> list[str]:
        tokens = text.split()
        tokens.reverse()
        ret : list[str] = []
        working = ""
        while len(tokens) > 0:
            token = tokens.pop()
            if (len(token) + len(working) + 1) >= 100:
                # Add working to ret, clear working back to ""
                ret.append(working + " ")
                working = ""
            working += " " + token
        ret.append(working)
        return ret


    def speakInference(self, text : str, model_name : str) -> npt.NDArray:
        ret : AudioSegment = None
        arrs = []
        for t in self._split_string_into_chunks(text, 512):
            print(f"Split text part: {t}")
            seg = s.infer_from_text(text = t, reference_file=model_name + ".wav")
            arrs.append(seg)
        try:
            return np.concatenate(arrs, 0)
        except Exception as e:
            print(e)
        # Convert the bytes object to a NumPy array
        #return np.frombuffer(ret.raw_data, dtype=np.int16)

    def calcWaveForm(self, data, freq : float = 440.0, sps : int = 44100, atten : Optional[float] = None):
        waveform =  np.sin(2 * np.pi * freq * data / sps)
        #waveform : ndarray[Any] = sin(2 * pi * each_sample_number * freq / sps)
        if atten:
            return waveform * atten
        else:
            return waveform
    
    def preprocess(self, text : str) -> str:
        return text.replace("\n", "")
    
    def convert_to_audiosegment(self, audio_wav) -> AudioSegment:
        # Normalize and convert NumPy array to int16 PCM
        audio_wav = (audio_wav * 32767).astype(np.int16).tobytes()
        # Create an in-memory buffer for raw audio
        raw_audio = io.BytesIO(audio_wav)
        # Convert raw audio into an AudioSegment
        return AudioSegment.from_raw(raw_audio, sample_width=2, frame_rate=44050, channels=1)

    def convert_to_ogg(self, audio_wav) -> bytes:
        buffer = io.BytesIO()
        
        try:
            # Convert raw audio into an AudioSegment
            audio_segment = self.convert_to_audiosegment(audio_wav)
            # Export to Opus format (WebM container) in memory
            audio_segment.export(buffer, format="webm", codec="libopus")
            
            return buffer.getvalue()
            # Write the NumPy array as OGG format to the buffer
        finally:
            buffer.close()

    def speak(self, text : str, play : bool = True, model_name : str = "Mario", ogg_format : bool = True) -> bytes | None:
        audio = self.speakInference(self.preprocess(text), model_name=model_name)
        rev_audio = self.calcWaveForm(data = audio, freq = 1440.0, sps = 44050)
        if play:
            sd.play(data = rev_audio, samplerate=44050, blocking=True)
        elif ogg_format:
            return self.convert_to_ogg(rev_audio)
        else:
            return rev_audio # return as wav


#jv = JarvisVoice()

#jv.speak("Hello! It is so nice to finally meet you. My name is Mel, and I will be your assistant today.")
