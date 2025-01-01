from fastapi import FastAPI, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from jarvisclient import JarvisClient
from jarvisdao import JarvisDAO
from asyncio import Queue, sleep
from starlette.types import Message

# Model Names
MODEL_FREDDY = "freddy_fazbear"
MODEL_RP = "rp"
MODEL_MARIO = "Mario"
MODEL_BONNIE = "bonnie"
MODEL_FOXY = "foxy"

jarvisDAO = JarvisDAO()
jarvisClient = JarvisClient(dao = jarvisDAO, model = MODEL_MARIO)
jarvisClient.createModel(modelName = MODEL_MARIO)
jarvisClient.setModel(model = MODEL_MARIO)

app = FastAPI()

app.mount("/static", StaticFiles(directory="html"), name="static")

# Shared dictionary to store audio data keyed by WebSocket client ID
audio_data_store = {}

@app.websocket("/prompt")
async def prompt(websocket : WebSocket):
    await websocket.accept()
    client_id = id(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            #print(f"Received from client {client_id}: {data}")
            jarvis_response : bytes = jarvisClient.prompt(data, give_voice = True)
            response_type = type(jarvis_response)
            print(f"Jarvis Response is a type of: {response_type}")
            #print(f"response looks like: {jarvis_response} and length = {jarvis_response.__len__()}")
            audio_data_store[repr(client_id)] = jarvis_response
            await websocket.send_text(repr(client_id))
    except WebSocketDisconnect:
        print("Client Disconnected")
        audio_data_store.pop(client_id, None)

async def waitforaudio(clientid):
    counter = 0
    while counter < 15 and not clientid in audio_data_store:
        counter = counter + 1
        print(f"counter {counter} and audio_data_store = {audio_data_store.keys()}")
        print(f"audio_data_store[clientid] = {audio_data_store[clientid]} and type {type(audio_data_store[clientid])}")
        await sleep(.5)

@app.websocket("/audio")
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

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)