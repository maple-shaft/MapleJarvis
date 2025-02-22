from typing import Any, Optional
import sounddevice as sd
import soundfile as sf
import numpy as np
import numpy.typing as npt
import jarvis_tts.synthesize2 as s 
import io
from pydub import AudioSegment
import pyrubberband as pyr

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
            if t is None or t == "":
                continue
            seg = s.infer_from_text(text = t, reference_file=model_name + ".wav")
            arrs.append(seg)
        try:
            if len(arrs) == 0:
                return None
            return np.concatenate(arrs, 0)
        except Exception as e:
            print(e)
        # Convert the bytes object to a NumPy array
        #return np.frombuffer(ret.raw_data, dtype=np.int16)

    def calcWaveForm(self, data, freq : float = 440.0, sps : int = 44100, atten : Optional[float] = None):
        waveform =  np.sin(2 * np.pi * freq * data / sps)
        if atten:
            return waveform * atten
        else:
            return waveform
    
    def convert_to_audiosegment(self, audio_wav, frame_rate = 44050) -> AudioSegment:
        # Normalize and convert NumPy array to int16 PCM
        audio_wav = (audio_wav * 32767).astype(np.int16).tobytes()
        # Create an in-memory buffer for raw audio
        raw_audio = io.BytesIO(audio_wav)
        # Convert raw audio into an AudioSegment
        return AudioSegment.from_raw(raw_audio, sample_width=2, frame_rate=frame_rate, channels=1)

    def convert_to_numpy(self, aus : AudioSegment):
        dtype = getattr(np, "int{:d}".format(aus.sample_width * 8))  # Or could create a mapping: {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}
        arr = np.ndarray((int(aus.frame_count()), aus.channels), buffer=aus.raw_data, dtype=dtype)
        print("\n", aus.frame_rate, arr.shape, arr.dtype, arr.size, len(aus.raw_data), len(aus.get_array_of_samples()))  # @TODO: Comment this line!!!
        return arr, aus.frame_rate

    def speak(self, text : str, play : bool = True, model_name : str = "Mario") -> bytes | None:
        from scipy.signal import resample_poly
        # This returns a float32 dtype numpy array.  Probably sampled at 44100.
        audio = self.speakInference(text, model_name=model_name)
        if audio is None:
            return None
        #rev_audio = self.calcWaveForm(data = audio, freq = 1440.0, sps = 21000, atten = 8)
        #rev_audio = (audio * 32767).astype(np.int16)
        #import pydub.scipy_effects as se
        #import pydub.effects as ef
        
        #aus = AudioSegment.from_raw(io.BytesIO(rev_audio.tobytes()), sample_width=2, frame_rate=44100, channels=1)
        #aus = ef.speedup(aus, playback_speed=1.75)
        #import pydub.playback as pb
        #num_samples = int(len(rev_audio) * 21050 / 44100)
        #rev_audio = resample(rev_audio, num_samples)

        rev_audio = audio
        #audio.clip(max=6000)
        #rev_audio = resample_poly(audio, 44100, 16000)
        audio_wav = (rev_audio * 32767).astype(np.int16)
        stretched_audio = pyr.time_stretch(y=audio_wav, sr=21000,rate=1.5)
        stretched_audio = stretched_audio.astype(np.float32)
        #y = np.array(aus.get_array_of_samples())
        #if aus.channels == 2:
        #    y = y.reshape((-1, 2))
        #if normalized:
        #rev_audio = np.float32(y) / 2**15
        #else:
        #    return a.frame_rate, y
        #rev_audio, rev_framerate = self.convert_to_numpy(aus)
        #p(aus)
        sd.play(stretched_audio, 21000, blocking=True)


jv = JarvisVoice()

jv.speak("Hello! It is so nice to finally meet you. My name is Freddy.", model_name="freddy_fazbear")
