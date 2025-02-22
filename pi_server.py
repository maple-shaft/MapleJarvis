import socket
import pickle
import time
import numpy as np
import asyncio
from typing import Any, AsyncGenerator, List
from transcribe_server import TranscribeAudio
from jarvisdao import JarvisDAO
from jarvisclient import JarvisClient
from jarvisvoice import JarvisVoice

SPACE : str = " "
MODEL_NAME : str = "freddy_fazbear"
HOST : str = "10.0.0.169"
PORT : int = 9100

class PiServer:

    def __init__(self, host : str = HOST, port : int = PORT,
                 model_name : str = MODEL_NAME):
        self.host = host
        self.port = port
        self.model_name = model_name
        self.transcribe = TranscribeAudio()
        self.transcribe.setup_model()
        self.jd = JarvisDAO()
        self.jv = JarvisVoice()
        self.jc = JarvisClient(dao = self.jd, model = self.model_name, conversation_id=None, isAsync=True, voice=self.jv)

    def start(self):
        el = asyncio.get_event_loop()
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen()
            print(f"Server listening on {self.host}:{self.port}")

            while True:
                conn, addr = self.socket.accept()
                with conn:
                    print(f"Connected by {addr}")
                    while True:
                        try:
                            obj = self.receive_audio(conn)
                            text = self.process_from_client(obj)
                            if text:
                                # sending text to Ollama
                                el.run_until_complete(self.process_model_response(text, conn))
                        except Exception as e:
                            print(f"Exception encountered: {e}")
                            break
        except KeyboardInterrupt:
            print("SIGINT captured, gracefully shutting down the server.")
            self.close()
            el.close()

    def close(self):
        print("Shutting down the PiServer.")
        self.jv.close()
        self.jd.close()
        self.socket.close()

    def receive_audio(self, conn : socket.socket):
        try:
            # first get serialized data length, first 8 bytes
            length_bytes = conn.recv(8)
            length = int(length_bytes.decode())
            full_length = length
            message = None

            # loop until length is zeroed out
            while length > 0:
                chunk_len = min(128, length)
                length -= chunk_len
                chunk = conn.recv(chunk_len)
                if message is None:
                    message = chunk
                else:
                    message = message + chunk
            
            while len(message) < full_length:
                chunk_len = min(128, full_length - len(message))
                chunk = conn.recv(chunk_len)
                message = message + chunk
        
            obj = pickle.loads(message)
            print(f"PiServer.receive_audio: obj received is {obj}")
            self.debug_write_bytes_to_file(obj)
            return obj    
        except Exception as e:
            print(f"PiServer.receive_audio: Exception encountered: {e}")
            raise

    async def collect_model_sentences(self, text : str):
        try:
            # Get an async generator from Ollama
            ag : AsyncGenerator[Any,Any] = self.jc.prompt_async(text)
            # Process the async generator to get words one at a time from the model
            working_sentence = []
            sentences = []
            async for ob in ag:
                word : str = ob
                working_sentence.append(word)
                if word.__contains__(".") or word.__contains__("!") or word.__contains__("?"):
                    # End of sentence marked
                    #ws = SPACE.join(working_sentence)
                    ws = "".join(working_sentence)
                    sentences.extend(ws)
                    working_sentence = []
                    # yield the sentence back

                    yield ws
            if len(working_sentence) > 0:
                yield SPACE.join(working_sentence)
        except Exception as e:
            print(e)

    async def process_model_response(self, text : str, conn : socket.socket):
        try:
            sentence_generator : AsyncGenerator[str, Any] = self.collect_model_sentences(text=text)
            async for sentence in sentence_generator:
                print(f"Complete sentence: {sentence}")
                ret = self.jv.speak(text=sentence, play=False, model_name=self.model_name, ogg_format=False)
                if ret is None:
                    print("PiServer.process_model_response: JarvisVoice.speak returned None, skipping to the next sentence.")
                    continue
                else:
                    print(f"PiServer.process_model_response: JarvisVoice.speak returned a {type(ret)}")
                #byte_data : bytes = ret
                #message = pickle.dumps(byte_data)
                self.debug_write_ndarray_to_file(ret)
                
                # Preprocess the float32 array into an int16
                #ret = (ret * 32767).astype(np.int16)
                message = pickle.dumps(ret)
                #print(f"PiServer.process_model_response: pickled message = {message}")
                length = len(message)
                length = str(length).rjust(8,"0")
                print("PiServer.process_model_response: About to send Jarvis audio back to the client.")
                conn.sendall(bytes(length, "utf-8"))
                conn.sendall(message)
        except Exception as e:
            print(f"PiServer.process_model_response: Exception encountered: {e}")
            raise

    def process_from_client(self, data : np.int16) -> str:
        print(f"Processing data from client: length = {len(data)}")
        text = self.transcribe.transcribe(data)
        return text
    
    def debug_write_bytes_to_file(self, audio16, filepath : str = f"/tmp/pitest{time.time()}.wav"):
        """Write a debug wave file from a bytes array. The byte data is expected to be 16 bit mono."""
        import wave
        try:
            audio16 = np.frombuffer(audio16, dtype=np.int16)
            with wave.open(filepath, "wb") as wv:
                wv.setnchannels(1)
                wv.setframerate(16000)
                wv.setsampwidth(2)
                wv.writeframes(audio16)
        except Exception as e:
            print(f"main:debug_write_to_file: Exception encountered: {e}")

    def debug_write_ndarray_to_file(self, audio : np.ndarray, filepath : str = f"/tmp/pitest{time.time()}.wav"):
        """Write a numpy ndarray to a debug file. This MUST be int16 dtype so act accordingly."""
        try:
            if audio.dtype == np.float32:
                # Convert it to int16
                audio = (audio * 32767).astype(np.int16)
            self.debug_write_bytes_to_file(audio16=audio, filepath=filepath)
        except Exception as e:
            print(f"PiServer.debug_write_ndarray_to_file: Exception encoutnered, {e}")


if __name__ == "__main__":
    server = PiServer(host=HOST, port=PORT, model_name=MODEL_NAME)
    server.start()
