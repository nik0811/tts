import asyncio
import websockets
import json
import wave
import sys
import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np
import random
import signal

SAMPLE_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Welcome to the text to speech system.",
    "Today is a beautiful day for programming.",
    "Artificial intelligence is transforming the world.",
    "I love learning new programming concepts.",
]

async def test_tts():
    uri = "ws://195.242.13.4:8888"
    max_retries = 3
    retry_delay = 2  # seconds
    connection_timeout = 30  # increased from 10
    message_timeout = 60     # increased from 30
    
    running = True
    def signal_handler(sig, frame):
        nonlocal running
        print("\nShutting down gracefully...")
        running = False
    signal.signal(signal.SIGINT, signal_handler)
    
    while running:
        for attempt in range(max_retries):
            try:
                print(f"Connection attempt {attempt + 1}/{max_retries}")
                async with asyncio.timeout(connection_timeout):
                    async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as websocket:
                        print("Connected to server!")
                        
                        while running:
                            try:
                                async with asyncio.timeout(message_timeout):
                                    message = {
                                        "text": random.choice(SAMPLE_SENTENCES),
                                        "language": "EN",
                                        "speed": 1.0
                                    }
                                    
                                    print(f"\nSending: {message['text']}")
                                    await websocket.send(json.dumps(message))
                                    
                                    audio_data = bytearray()
                                    received_audio = False
                                    
                                    stream = sd.OutputStream(
                                        samplerate=24000,
                                        channels=1,
                                        dtype='int16'
                                    )
                                    stream.start()

                                    while running:
                                        try:
                                            async with asyncio.timeout(10):
                                                chunk = await websocket.recv()
                                            if isinstance(chunk, str):
                                                try:
                                                    msg = json.loads(chunk)
                                                    print(f"Received message: {msg}")
                                                    if 'error' in msg:
                                                        print(f"Error from server: {msg['error']}")
                                                        break
                                                    if msg.get('status') == 'finished':
                                                        print("Sentence complete, starting next one...")
                                                        break
                                                except json.JSONDecodeError:
                                                    print(f"Received non-JSON string: {chunk}")
                                                    pass
                                            else:
                                                received_audio = True
                                                audio_array = np.frombuffer(chunk, dtype=np.int16)
                                                stream.write(audio_array)
                                                audio_data.extend(chunk)
                                        except asyncio.TimeoutError:
                                            print("Chunk reception timed out, trying next message...")
                                            break
                                        except websockets.exceptions.ConnectionClosed:
                                            print("Connection lost while receiving chunks...")
                                            running = False
                                            break

                                    stream.stop()
                                    stream.close()

                                    if received_audio:
                                        print("Audio playback complete, moving to next sentence...")
                                        await asyncio.sleep(1)
                                        continue
                                    
                                    await asyncio.sleep(1)
                            
                            except asyncio.TimeoutError:
                                print("Message cycle timed out, reconnecting...")
                                break
                            except websockets.exceptions.ConnectionClosed:
                                print("Connection closed by server, reconnecting...")
                                break
                        
                        if not running:
                            return
                            
            except asyncio.TimeoutError:
                print(f"Connection attempt {attempt + 1} timed out.")
            except ConnectionRefusedError:
                print(f"Connection attempt {attempt + 1} refused.")
            except Exception as e:
                print(f"An error occurred on attempt {attempt + 1}: {e}")
            
            if attempt < max_retries - 1 and running:
                print(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                if running:
                    print("Max retries reached. Waiting before starting over...")
                    await asyncio.sleep(5)
                    break
                else:
                    print("Shutdown requested. Exiting.")
                    return

if __name__ == "__main__":
    asyncio.run(test_tts())