from fastapi import FastAPI, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from jarvisclient import JarvisClient
from jarvisdao import JarvisDAO
from asyncio import Queue, sleep
from starlette.types import Message
from jarvis_stt.audio_to_text import AudioRecorder
from jarvis_stt.audio_input import BufferedAudioInput
import threading
import time
import torch
import av
from io import BytesIO
from pydub import AudioSegment

# Model Names
MODEL_FREDDY = "freddy_fazbear"
MODEL_RP = "rp"
MODEL_MARIO = "Mario"
MODEL_BONNIE = "bonnie"
MODEL_FOXY = "foxy"

# Setup Jarvis Client
#jarvisDAO = JarvisDAO()
#jarvisClient = JarvisClient(dao = jarvisDAO, model = MODEL_RP)
#jarvisClient.createModel(modelName = MODEL_RP)
#jarvisClient.setModel(model = MODEL_RP)

# Setup FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="html"), name="static")

# Shared dictionary to store audio data keyed by WebSocket client ID
audio_data_store = {}
# Shared dictionary to store client speech transcribed text by client id
prompt_data_store : dict[str, str] = {}
# Client AudioInput instances
audio_inputs : dict[str, BufferedAudioInput] = {}
# Client AudioRecorder instances
audio_recorders : dict[str, AudioRecorder] = {}

@app.websocket("/audio/{client_id}")
async def audio(websocket : WebSocket, client_id : str):
    await websocket.accept()
    print(f"WS Audio: client_id = {client_id}")
    try:
        while True:
            await sleep(0.1)
            if client_id in prompt_data_store:
                print("web_client_talk: audio: A text prompt was found for this client id.")
                text_prompt : str = prompt_data_store.pop(client_id)
                #jarvis_audio = jarvisClient.prompt(prompt=text_prompt, give_voice = True)
                #await websocket.send_bytes(jarvis_audio)
            else:
                continue
    except WebSocketDisconnect:
        print("Client Disconnected")
        audio_data_store.pop(client_id, None)
        prompt_data_store.pop(client_id, None)
        aui = audio_inputs.pop(client_id, None)
        if aui:
            aui.stop_event.set()
            aui.cleanup()

@app.websocket("/promptaudio/{client_id}")
async def prompt(websocket : WebSocket, client_id : str):
    await websocket.accept()
    print(f"WS Audio: client_id = {client_id}")

    if not audio_inputs.get(client_id):
        audio_inputs[client_id] = BufferedAudioInput(client_id=client_id)

    if not audio_recorders.get(client_id):
        silero_vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            verbose=False,
            onnx=False
        )
        audio_recorders[client_id] = AudioRecorder(audio_input = audio_inputs[client_id], silero_vad_model=silero_vad_model)

    try:
        buffer = BytesIO()
        codec_context = None
        bufferlength = 0
        while True:
            data = await websocket.receive_bytes()
            bufferlength += len(data)
            buffer.write(data)
            if bufferlength < 32768:
                continue
            else:
                bufferlength = 0
            buffer.seek(0)
            try:
                #if codec_context is None:
                #    container = av.open(BytesIO(), format="webm")
                #    codec_context = next(s.codec_context for s in container.streams if s.type == 'audio')
                codec_context = av.CodecContext()
                codec_context.create(codec="webm")

                container = av.open(buffer)
                for packet in container.demux():
                    if packet.dts is not None:
                        for frame in packet.decode():
                            pcm_data = frame.to_ndarray()
                            print(f"Decoded {pcm_data.shape} audio frame(s)")
                            await audio_inputs[client_id].write_chunk(pcm_data)
            except Exception as e:
                continue
                #print(f"Exception in promptaudio as {e}")
                #await audio_inputs[client_id].write_chunk(audio_inputs[client_id].convert_webm_opus_to_wav(buffer))
            finally:
                buffer.seek(0)
                buffer.truncate(0)

            #print(f"Received from client {client_id}: {data}")
            #jarvis_response : bytes = jarvisClient.prompt(data, give_voice = True)
            #response_type = type(jarvis_response)
            #print(f"Jarvis Response is a type of: {response_type}")
            #print(f"response looks like: {jarvis_response} and length = {jarvis_response.__len__()}")
            #audio_data_store[repr(client_id)] = jarvis_response
            #await websocket.send_text(repr(client_id))
    except WebSocketDisconnect:
        print("Client Disconnected")
        audio_data_store.pop(client_id, None)
        prompt_data_store.pop(client_id, None)
        aui = audio_inputs.pop(client_id, None)
        if aui:
            aui.stop_event.set()
            aui.cleanup()

async def waitforaudio(clientid):
    counter = 0
    while counter < 15 and not clientid in audio_data_store:
        counter = counter + 1
        print(f"counter {counter} and audio_data_store = {audio_data_store.keys()}")
        print(f"audio_data_store[clientid] = {audio_data_store[clientid]} and type {type(audio_data_store[clientid])}")
        await sleep(.5)

async def audio_stream(websocket: WebSocket):
    await websocket.accept()
    print("Awaited the audio connection.")
    try:
        while True:
            print("check receive")
            d : Message = await websocket.receive()
            print(f"Received a message {d}")
            test = "text" in d
            print(f"The statement text in d = {test}")
            #test = d["text"] is type(str)
            #print(f"The statement text in d = {test}")
            
            if "text" in d:
                print("Received text at /audio")
                clientid = d["text"]
                print(f"Client id {clientid}")
                # check to see if we have audio for this
                await waitforaudio(clientid)
                if clientid in audio_data_store:
                    audio_data = audio_data_store[clientid]
                    print(f"type of audio data: {type(audio_data)}")
                    await websocket.send_bytes(audio_data)
                    print("sent audio bytes")
            else:
                await sleep(0.5)
    except WebSocketDisconnect:
        print("Client disconnected")

def recording_worker():
    """A threaded worker that actively looks for audio data to transcribe for client ids"""
    print("web_client_talk: initializing recording worker")
    while True:
        time.sleep(0.5)
        for client_id, audio_recorder in audio_recorders.items():
            print(f"web_client_talk: recording_worker: checking client_id: {client_id}")
            transcribed_text = audio_recorder.text()
            print(f"web_client_talk: recording_worker: transcribed_text = {transcribed_text}")
            prompt_data_store[client_id] = transcribed_text

    
thread = threading.Thread(target = recording_worker, args = ())
thread.daemon = True
thread.start()

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)