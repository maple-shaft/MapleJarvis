import faster_whisper
import numpy as np

MODEL_PATH = "tiny"
DOWNLOAD_ROOT = None
COMPUTE_TYPE = "default"
GPU_DEVICE_INDEX = 0
DEVICE = "cpu"
BEAM_SIZE = 5
INITIAL_PROMPT = None
SUPPRESS_TOKENS = [-1]

class TranscribeAudio:

    def __init__(self, model_path = MODEL_PATH, device = DEVICE, compute_type = COMPUTE_TYPE,
                 gpu_device_index = GPU_DEVICE_INDEX, download_root = DOWNLOAD_ROOT,
                 beam_size = BEAM_SIZE, initial_prompt = INITIAL_PROMPT, suppress_tokens = SUPPRESS_TOKENS):
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self.gpu_device_index = gpu_device_index
        self.download_root = download_root
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.suppress_tokens = suppress_tokens

    def setup_model(self):
        print("Setting up Faster Whisper model for audio transcription.")
        try:
            self.model = faster_whisper.WhisperModel(
                model_size_or_path=self.model_path,
                device=self.device,
                compute_type=self.compute_type,
                device_index=self.gpu_device_index,
                download_root=self.download_root
            )

            dummy_audio = np.zeros(16000, dtype=np.float32)
            self.model.transcribe(dummy_audio, language="en", beam_size=1)
            print("Faster_whisper main speech to text transcription model initialized successfully")
        except Exception as e:
            print(f"Exception encountered in setup_model: {e}")
            raise

    def transcribe(self, data : np.int16):
        print("Starting transcription of client received data...")
        try:
            audio = data.astype(np.float32) / 32768.0
            segments, info = self.model.transcribe(
                audio,
                language=None,
                beam_size=self.beam_size,
                initial_prompt=self.initial_prompt,
                suppress_tokens=self.suppress_tokens
            )

            transcription = " ".join(seg.text for seg in segments).strip()
            print(f"Transcribed the following text: {transcription}")
            return transcription
        except Exception as e:
            print(f"Exception encountered in transcription: {e}")
            raise