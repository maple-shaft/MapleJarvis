import cv2
import os
import time
import pandas as pd
from PIL import Image
from io import BytesIO
from ollama import AsyncClient, Client
from threading import Event, Thread
from multiprocessing import Queue
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import Executor, Future

IMAGE_MODEL : str = "llava"
PROMPT : str = "describe this image and make sure to include anything notable about it (include text you see in the image):"
OUTPUT_FILE_PATH : str = "./data/webcam_descriptions.csv"
FRAME_RATE : int = 40
WIDTH : int = 640
HEIGHT : int = 480

class JarvisVision:

    def __init__(self,
                 ollama_client : Client,
                 shutdown_event : Event,
                 model : str = IMAGE_MODEL,
                 frame_queue : Queue = Queue(maxsize=200),
                 frame_rate : int = FRAME_RATE,
                 executor : Executor = None,
                 output_file_path : str = OUTPUT_FILE_PATH):
        print("Init JarvisVision")
        # Initialize the webcam
        self.video_capture = cv2.VideoCapture(0)
        # Check if the webcam is opened successfully
        if not self.video_capture.isOpened():
            raise IOError("Cannot open webcam")
        
        # Get the default frame width and height
        self.frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Frame Width: {self.frame_width}")
        print(f"Frame Height: {self.frame_height}")
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.frame_queue = frame_queue
        self.frame_rate = frame_rate
        self.executor = executor
        self.output_file_path = output_file_path
        self.data_frame = self._load_or_create_dataframe()
        self.shutdown_event = shutdown_event
        self.ollama_client = ollama_client
        self.model = model

    # Load the DataFrame from a CSV file, or create a new one if the file doesn't exist
    def _load_or_create_dataframe(self) -> pd.DataFrame:
        filename = self.output_file_path
        if os.path.isfile(filename):
            df = pd.read_csv(filename)
        else:
            df = pd.DataFrame(columns=['image_file', 'description'])
        return df
    
    def start(self) -> Future | Thread:
        if self.executor:
            return self.executor.submit(self.watch)
        else:
            t = Thread(self.watch)
            t.daemon = True
            t.start()
            return t

    def get_image_bytes(self, format : str = "PNG") -> bytes:
        ret, frame = self.video_capture.read()
        if not ret:
            return None
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        with BytesIO() as buffer:
            image.save(buffer, format=format)
            return buffer.getvalue()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.video_capture:
            self.video_capture.release()
        self.data_frame.to_csv(self.output_file_path, index=False)

    def watch(self):
        frame_rate_millis : float = 1000 / self.frame_rate
        if not self.video_capture:
            return None
        try:
            while not self.shutdown_event.is_set():
                frame_bytes = self.get_image_bytes()
                full_response = ""
                for response in self.ollama_client.generate(model = self.model,
                                            prompt=PROMPT,
                                            images=[frame_bytes],
                                            stream=True
                                            ):
                    print(response['response'], end='', flush=True)
                    full_response += response['response']
                
                # Trim up unnecessary line feeds and spacing
                full_response = full_response.replace("\n", " ")
                # Add a new row to the DataFrame
                self.data_frame.loc[len(self.data_frame)] = [time.time(), full_response]
                time.sleep(10.0)
        except Exception as e:
            print(f"Exception encountered in JarvisVision.watch: {e}")

def main():
    q = Queue(maxsize=200)
    shutdown_event = Event()
    ollama_client = Client(host="http://10.0.0.169:11434")
    
    with ThreadPoolExecutor() as executor:
        with JarvisVision(frame_queue = q, executor=executor, ollama_client=ollama_client, shutdown_event=shutdown_event) as jv:
            try:
                t_or_f = jv.start()
                if type(t_or_f) == Thread:
                    t_or_f.join()
                
                last_save = time.time()
                while True:
                    current_time = time.time() - last_save
                    if current_time > 60:
                        jv.data_frame.to_csv(jv.output_file_path)
                        last_save = time.time()
                    else:
                        time.sleep(5)
            except KeyboardInterrupt:
                print("Keyboard interrupt captured")
                shutdown_event.set()

if __name__ == "__main__":
    main()