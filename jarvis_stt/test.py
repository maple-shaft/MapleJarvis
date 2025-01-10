import pyaudio as pa
import wave

CHANNELS = 1
RATE = 44100
CHUNK = 1024
FORMAT = pa.paInt16
SAMPLEWIDTH = 2
DEVICE_INDEX = 2

aui = pa.PyAudio()

def list_devices(audio_interface : pa.PyAudio):
    try:
        device_count = audio_interface.get_device_count()
        for i in range(device_count):
            device_info = audio_interface.get_device_info_by_index(i)
            device_name = device_info.get("name")
            print(f"list_devices -> Device ID: {i}, Device Name: {device_name}")
            if (device_info.get("maxInputChannels", 0) > 0):
                # Only consider devices with record capability
                print(f"list_devices -> Recording Device Found!")

    except Exception as e:
        print(f"Exception encountered: {e}")
    finally:
        if audio_interface:
            audio_interface.terminate()

def open_stream(audio_interface : pa.PyAudio):
    return audio_interface.open(
        rate=RATE,
        channels=CHANNELS,
        format=FORMAT,
        input=True,
        input_device_index=DEVICE_INDEX,
        frames_per_buffer=CHUNK
    )

def write(data, rate = RATE):
    with wave.open(f"/tmp/pcm_DUSTIN.wav", "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLEWIDTH)
        wav_file.setframerate(rate)
        wav_file.writeframes(data)

def preprocess(chunk, target_sample_rate) -> bytes:
    from scipy import signal as sig
    import numpy as np
    # chunk must be bytes
    #self._dab_write(chunk)
    chunk = np.frombuffer(chunk, dtype=np.int16)
    # resample if necessary
    if RATE != target_sample_rate:
        num_samples = int(len(chunk) * target_sample_rate / RATE)
        chunk = sig.resample(chunk, num_samples)
        # Ensure it is fp16
        chunk = chunk.astype(np.int16)
    return chunk.tobytes()

stream = open_stream(aui)

while stream.read(1, False) == 0:
    pass

data = stream.read(CHUNK, False)

for i in range(1,200):
    data += stream.read(CHUNK, False)

data = preprocess(data, 16000)
write(data=data, rate=16000)

stream.close()