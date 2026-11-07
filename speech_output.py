"""Serialize Windows speech on a dedicated worker, without blocking the UI."""
import queue
import threading
import io
import asyncio
import re
import wave
from pathlib import Path


def speak_edge_online(text, voice):
    """Use an Edge neural voice; keep generated audio in memory."""
    import edge_tts
    import miniaudio
    import numpy as np
    import sounddevice as sd

    async def collect_audio():
        chunks = []
        async for chunk in edge_tts.Communicate(text, voice).stream():
            if chunk['type'] == 'audio':
                chunks.append(chunk['data'])
        return b''.join(chunks)

    encoded = asyncio.run(collect_audio())
    if not encoded:
        raise RuntimeError('The online voice returned no audio.')
    decoded = miniaudio.decode(encoded, output_format=miniaudio.SampleFormat.SIGNED16)
    samples = np.frombuffer(decoded.samples, dtype=np.int16)
    if decoded.nchannels > 1:
        samples = samples.reshape(-1, decoded.nchannels)
    sd.play(samples, samplerate=decoded.sample_rate)
    sd.wait()


def speak_english_online(text):
    speak_edge_online(text, 'en-IN-PrabhatNeural')


def speak_hindi_online(text):
    speak_edge_online(text, 'hi-IN-MadhurNeural')


def speak_online(text):
    """Speak English or Hindi through Gemini; play the WAV in memory."""
    from google import genai
    from google.genai import types
    import numpy as np
    import sounddevice as sd
    from online_gemini import load_gemini_config

    config = load_gemini_config(Path(__file__).resolve().parent)
    if config is None:
        raise RuntimeError('Gemini API key is missing.')
    client = genai.Client(api_key=config.api_key)
    response = client.models.generate_content(
        model='gemini-3.8-flash-lite-tts',
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=['AUDIO'],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Kore')
                )
            ),
        ),
    )
    data = response.candidates[0].content.parts[0].inline_data.data
    with wave.open(io.BytesIO(data), 'rb') as wav:
        channels, width, rate = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
        if width != 2:
            raise ValueError('Unsupported audio format from Gemini.')
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')
        if channels > 1:
            samples = samples.reshape(-1, channels)
        sd.play(samples, samplerate=rate)
        sd.wait()


class SpeechOutput:
    def __init__(self, on_error):
        self._queue = queue.Queue()
        self._on_error = on_error
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, text):
        if text and text.strip():
            self._queue.put(text.strip())

    def close(self):
        self._queue.put(None)

    def _run(self):
        engine = None
        while True:
            text = self._queue.get()
            if text is None:
                self._queue.task_done()
                break
            try:
                if re.search(r'[\u0900-\u097f]', text):
                    speak_hindi_online(text)
                else:
                    speak_english_online(text)
            except Exception:
                if re.search(r'[\u0900-\u097f]', text):
                    self._on_error('Online Hindi voice unavailable. Check internet and sound output; the reply remains visible.')
                else:
                    try:
                        if engine is None:
                            import pyttsx3
                            engine = pyttsx3.init()
                            engine.setProperty('rate', 165)
                            engine.setProperty('volume', 1.0)
                        self._on_error('Online voice unavailable; using the Windows voice for this reply.')
                        engine.say(text)
                        engine.runAndWait()
                    except Exception:
                        self._on_error('Could not speak the reply. Check your internet and Windows sound output.')
            finally:
                self._queue.task_done()
        try:
            if engine is not None:
                engine.stop()
        except Exception:
            pass

