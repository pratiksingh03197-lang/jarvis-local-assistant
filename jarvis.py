# =====================================================
# PROJECT JARVIS - PERSONAL AI ASSISTANT
# VERSION 4.1
# Local wake word + Hindi/English/Hinglish ASR
# Safe, allowlisted Windows actions
# =====================================================

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import webbrowser
import winsound
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

from serpapi_live_search import (
    SearchDemoError,
    detect_response_language,
    search_and_summarize,
)

import numpy as np
import pyttsx3
import requests
import sounddevice as sd
import webrtcvad
from scipy.signal import resample_poly

from faster_whisper import WhisperModel
from openwakeword.model import Model


# Make Hindi transcripts safe to print from legacy Windows consoles as well as
# Windows Terminal. Without this, a valid Hindi transcript can crash on cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


# =====================================================
# SETTINGS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
CUSTOM_WAKE_MODEL = BASE_DIR / "jarvis_wake.onnx"

AI_MODEL = "qwen3.5:2b"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

GOOGLE_HOME_URL = "https://www.google.com/"
YOUTUBE_HOME_URL = "https://www.youtube.com/"
YOUTUBE_SEARCH_URL = "https://www.youtube.com/results?search_query="
ALLOWED_BROWSER_URL_PREFIXES = (
    GOOGLE_HOME_URL,
    "https://www.google.com/search?",
    YOUTUBE_HOME_URL,
    YOUTUBE_SEARCH_URL,
)
CHROME_CANDIDATES = (
    Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    / "Google/Chrome/Application/chrome.exe",
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    / "Google/Chrome/Application/chrome.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
)

# openWakeWord, WebRTC VAD, and Whisper all use this rate.
PROCESSING_SAMPLE_RATE = 16_000

# The Surface Laptop's built-in WASAPI microphone works natively at 48 kHz.
# JARVIS captures there when available, then resamples to 16 kHz locally.
PREFERRED_CAPTURE_RATE = 48_000
FALLBACK_CAPTURE_RATE = 16_000
INPUT_CHANNELS = 1

# Wake-word tuning. Lower only if wake detection is genuinely missing speech.
WAKE_FRAME_SAMPLES = 1_280  # 80 ms at 16 kHz
WAKE_THRESHOLD = 0.45
WAKE_CONSECUTIVE_HITS = 1

# If a wake model misses an actual spoken phrase, JARVIS makes a short,
# local-only speech check. It retains no audio and only accepts "Jarvis".
ASR_WAKE_RMS_THRESHOLD = 0.008
ASR_WAKE_PRE_ROLL_SECONDS = 0.60
ASR_WAKE_SILENCE_SECONDS = 0.65
ASR_WAKE_MAX_SECONDS = 2.5
WAKE_FALLBACK_MODEL = "base"

# Command capture tuning. The longer end-silence and pre-roll prevent clipped
# Hinglish/Hindi phrases and let a speaker pause naturally.
COMMAND_FRAME_MS = 30
COMMAND_FRAME_SAMPLES = int(PROCESSING_SAMPLE_RATE * COMMAND_FRAME_MS / 1000)
PRE_ROLL_SECONDS = 0.75
MAX_WAIT_SECONDS = 12.0
MAX_COMMAND_SECONDS = 45.0
MIN_COMMAND_SECONDS = 0.75
END_SILENCE_SECONDS = 2.8

# VAD mode: 0 is most permissive, 3 is most aggressive.
VAD_MODE = 1
MIN_AUDIO_RMS = 0.0020
NOISE_MULTIPLIER = 3.0
NOISE_CALIBRATION_FRAMES = 15  # 450 ms of room tone before energy fallback

# "small" is materially more accurate for Hindi/English code-switching. It is
# downloaded once, for free, on the first run. If it cannot load, JARVIS falls
# back safely to the already-installed "base" model.
DEFAULT_WHISPER_MODEL = os.environ.get("JARVIS_WHISPER_MODEL", "small")
FALLBACK_WHISPER_MODEL = "base"
WHISPER_CPU_THREADS = min(4, os.cpu_count() or 4)
WHISPER_BEAM_SIZE = 5

SYSTEM_PROMPT = """
You are JARVIS, a local personal AI assistant.

You understand English, Hindi, and Hinglish. Reply in concise, simple English
so the installed Windows voice can speak clearly.

You help with productivity, business planning, research, analysis, and daily
assistance. You do not execute operating-system commands, browse private data,
send messages, change files, or control the computer. The Python safety layer
handles a very small, explicit allowlist of computer actions separately.

Web searches are handled separately through SerpApi only after the user gives
spoken confirmation. Do not claim that you searched the web without results.

Never claim that a computer action happened unless it was completed. Do not use
Markdown in spoken responses.
"""

TRANSCRIPTION_PROMPT = (
    "This is a spoken personal-assistant command in Hindi, English, or Hinglish. "
    "Transcribe exactly; do not translate. Keep names and app names as spoken. "
    "Common words: Jarvis, Notepad, Calculator, Google, YouTube, SerpApi, "
    "search the web, खोलो, खोजो, कैल्कुलेटर, नोटपैड."
)

conversation_history: list[dict[str, str]] = []


# =====================================================
# AUDIO DEVICE SELECTION
# =====================================================

@dataclass(frozen=True)
class Microphone:
    index: int
    name: str
    hostapi: str
    capture_rate: int

    @property
    def is_resampled(self) -> bool:
        return self.capture_rate != PROCESSING_SAMPLE_RATE


def _hostapi_name(device: dict) -> str:
    try:
        return str(sd.query_hostapis(int(device["hostapi"]))["name"])
    except Exception:
        return "Unknown host API"


def _input_devices() -> list[tuple[int, dict, str]]:
    devices: list[tuple[int, dict, str]] = []
    for index, device in enumerate(sd.query_devices()):
        if int(device["max_input_channels"]) > 0:
            devices.append((index, device, _hostapi_name(device)))
    return devices


def _supports_input(index: int, rate: int) -> bool:
    try:
        sd.check_input_settings(
            device=index,
            channels=INPUT_CHANNELS,
            samplerate=rate,
            dtype="int16",
        )
        return True
    except Exception:
        return False


def list_microphones() -> None:
    """Show the input devices JARVIS can use; this does not record audio."""
    default_input = None
    try:
        default_input = int(sd.default.device[0])
    except Exception:
        pass

    print("\nAvailable input devices (no audio is recorded):")
    for index, device, hostapi in _input_devices():
        rates = []
        for rate in (PREFERRED_CAPTURE_RATE, FALLBACK_CAPTURE_RATE):
            if _supports_input(index, rate):
                rates.append(f"{rate // 1000} kHz")
        marker = "  <-- Windows default" if index == default_input else ""
        usable = ", ".join(rates) if rates else "not usable by JARVIS"
        print(
            f"  [{index}] {device['name']} | {hostapi} | "
            f"{int(device['max_input_channels'])} channel(s) | {usable}{marker}"
        )


def _requested_device_matches(
    requested: str, devices: list[tuple[int, dict, str]]
) -> list[int]:
    requested = requested.strip()
    if not requested:
        return []

    if requested.isdigit():
        requested_index = int(requested)
        return [index for index, _, _ in devices if index == requested_index]

    needle = requested.casefold()
    return [
        index
        for index, device, hostapi in devices
        if needle in str(device["name"]).casefold()
        or needle in hostapi.casefold()
    ]


def choose_microphone(requested: Optional[str] = None) -> Microphone:
    """
    Select one validated input route. A named/numbered --mic choice wins. Without
    one, prefer the built-in Microphone Array through low-latency WASAPI.
    """
    devices = _input_devices()
    if not devices:
        raise RuntimeError("No recording input devices were found.")

    requested = requested or os.environ.get("JARVIS_MIC_DEVICE")
    if requested:
        candidate_indices = _requested_device_matches(requested, devices)
        if not candidate_indices:
            raise RuntimeError(
                f"No input device matches {requested!r}. Run with --list-devices."
            )
    else:
        default_input = None
        try:
            default_input = int(sd.default.device[0])
        except Exception:
            pass

        def priority(item: tuple[int, dict, str]) -> tuple[int, int]:
            index, device, hostapi = item
            name = str(device["name"]).casefold()
            api = hostapi.casefold()
            if "microphone array" in name and "wasapi" in api:
                return (0, index)
            if "wasapi" in api:
                return (1, index)
            if index == default_input:
                return (2, index)
            if "microphone" in name:
                return (3, index)
            return (4, index)

        candidate_indices = [index for index, _, _ in sorted(devices, key=priority)]

    details = {index: (device, hostapi) for index, device, hostapi in devices}
    for index in candidate_indices:
        device, hostapi = details[index]
        for rate in (PREFERRED_CAPTURE_RATE, FALLBACK_CAPTURE_RATE):
            if _supports_input(index, rate):
                return Microphone(
                    index=index,
                    name=str(device["name"]),
                    hostapi=hostapi,
                    capture_rate=rate,
                )

    raise RuntimeError(
        "JARVIS could not open a mono 48 kHz or 16 kHz stream on the chosen "
        "microphone. Run with --list-devices and choose a usable number with --mic."
    )


def _capture_frames(mic: Microphone, processing_frames: int) -> int:
    frames = processing_frames * mic.capture_rate / PROCESSING_SAMPLE_RATE
    rounded = round(frames)
    if not np.isclose(frames, rounded):
        raise RuntimeError(
            f"{mic.capture_rate} Hz cannot be cleanly converted to "
            f"{PROCESSING_SAMPLE_RATE} Hz in real time."
        )
    return int(rounded)


def _open_input_stream(mic: Microphone, processing_frames: int) -> sd.RawInputStream:
    return sd.RawInputStream(
        device=mic.index,
        samplerate=mic.capture_rate,
        blocksize=_capture_frames(mic, processing_frames),
        channels=INPUT_CHANNELS,
        dtype="int16",
        latency="low",
    )


def _to_processing_rate(raw_audio: bytes, mic: Microphone) -> np.ndarray:
    samples = np.frombuffer(raw_audio, dtype=np.int16).copy()

    if mic.capture_rate == PROCESSING_SAMPLE_RATE:
        return samples

    # The validated preferred Surface path is 48 kHz -> 16 kHz. scipy's
    # polyphase resampler provides the anti-aliasing that simple decimation lacks.
    converted = resample_poly(
        samples.astype(np.float32),
        up=PROCESSING_SAMPLE_RATE,
        down=mic.capture_rate,
    )
    converted = np.clip(np.rint(converted), -32768, 32767).astype(np.int16)
    return converted


def _read_processing_frame(
    stream: sd.RawInputStream, mic: Microphone, processing_frames: int
) -> tuple[np.ndarray, bool]:
    raw_audio, overflowed = stream.read(_capture_frames(mic, processing_frames))
    audio = _to_processing_rate(raw_audio, mic)

    # Do not pass a malformed frame to WebRTC VAD/openWakeWord.
    if len(audio) != processing_frames:
        audio = audio[:processing_frames]
    return audio, bool(overflowed)


def audio_metrics(audio: np.ndarray) -> dict[str, float]:
    if len(audio) == 0:
        return {"seconds": 0.0, "rms": 0.0, "peak": 0.0, "clipping": 0.0}

    normalized = np.asarray(audio, dtype=np.float32)
    if np.issubdtype(audio.dtype, np.integer):
        normalized = normalized / 32768.0

    return {
        "seconds": len(normalized) / PROCESSING_SAMPLE_RATE,
        "rms": float(np.sqrt(np.mean(np.square(normalized, dtype=np.float64)))),
        "peak": float(np.max(np.abs(normalized))),
        "clipping": float(np.mean(np.abs(normalized) >= 0.995)),
    }


def print_audio_metrics(label: str, audio: np.ndarray, overflows: int = 0) -> None:
    metrics = audio_metrics(audio)
    print(
        f"{label}: {metrics['seconds']:.1f}s | RMS {metrics['rms']:.4f} | "
        f"peak {metrics['peak']:.3f} | clipping {metrics['clipping'] * 100:.2f}% | "
        f"overflows {overflows}"
    )
    if metrics["rms"] < MIN_AUDIO_RMS:
        print(
            "Audio was quiet. Move closer to the microphone or choose another "
            "device with --list-devices."
        )
    if metrics["clipping"] > 0.01:
        print("Audio is clipping. Move farther from the microphone or lower input gain.")


def _asr_detects_jarvis(
    wake_whisper: WhisperModel, captured: np.ndarray, show_debug: bool
) -> bool:
    """
    Check one short, already-detected speech burst locally. This is deliberately
    not continuous transcription and the buffer is released immediately after.
    """
    if len(captured) == 0:
        return False

    try:
        segments, _ = wake_whisper.transcribe(
            captured.astype(np.float32) / 32768.0,
            language=None,
            task="transcribe",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt="The wake phrase is Jarvis.",
            hotwords="Jarvis",
            vad_filter=False,
            without_timestamps=True,
            no_speech_threshold=0.60,
            log_prob_threshold=-1.2,
        )
        transcript = " ".join(
            segment.text.strip() for segment in segments if segment.text.strip()
        )
    except Exception as error:
        if show_debug:
            print(f"Wake speech fallback error: {error}")
        return False

    # The transcript is printed only in explicit debug mode. Normal standby
    # avoids exposing non-wake background speech in the console.
    if show_debug:
        print(f"Wake speech candidate: {transcript or '[none]'}")

    normalized = re.sub(r"[^a-z0-9]+", " ", transcript.casefold())
    return bool(re.search(r"\bjarvis\b", normalized))


def microphone_test(mic: Microphone, seconds: float) -> None:
    """
    Explicit opt-in diagnostic. It measures only in memory and never saves audio.
    """
    seconds = max(1.0, min(seconds, 20.0))
    print(
        f"\nMicrophone test: JARVIS will listen to {seconds:.0f} seconds from "
        f"'{mic.name}'. It will not save or transmit audio."
    )
    input("Press Enter when you are ready to speak, or Ctrl+C to cancel. ")

    frames: list[np.ndarray] = []
    overflows = 0
    started = time.monotonic()
    with _open_input_stream(mic, COMMAND_FRAME_SAMPLES) as stream:
        while time.monotonic() - started < seconds:
            frame, overflowed = _read_processing_frame(
                stream, mic, COMMAND_FRAME_SAMPLES
            )
            frames.append(frame)
            overflows += int(overflowed)

    joined = np.concatenate(frames) if frames else np.array([], dtype=np.int16)
    print_audio_metrics("Microphone test", joined, overflows)
    print("Test complete. No audio file was created.")


# =====================================================
# VOICE CAPTURE
# =====================================================

def wait_for_wake_word(
    mic: Microphone,
    wake_detector: Model,
    wake_whisper: Optional[WhisperModel],
    show_scores: bool = False,
) -> None:
    print("\n-----------------------------------")
    print("JARVIS STATUS: STANDBY")
    print(f"Microphone: {mic.name} ({mic.hostapi}, {mic.capture_rate // 1000} kHz)")
    print("Say 'Jarvis' or 'Hey Jarvis' to activate.")
    print("-----------------------------------")

    wake_detector.reset()
    consecutive_hits = 0
    overflow_count = 0
    last_debug_at = 0.0
    max_score_since_debug = 0.0
    max_rms_since_debug = 0.0
    latest_scores: dict[str, float] = {}

    # This buffer is only used after an actual short speech burst. It gives
    # plain "Jarvis" a reliable local fallback when a custom wake model fails.
    asr_pre_roll = deque(
        maxlen=max(
            1,
            round(
                ASR_WAKE_PRE_ROLL_SECONDS
                * PROCESSING_SAMPLE_RATE
                / WAKE_FRAME_SAMPLES
            ),
        )
    )
    asr_recent_voice = deque(maxlen=3)
    asr_recorded: list[np.ndarray] = []
    asr_active = False
    asr_started_at: Optional[float] = None
    asr_last_voice_at: Optional[float] = None

    with _open_input_stream(mic, WAKE_FRAME_SAMPLES) as stream:
        while True:
            frame, overflowed = _read_processing_frame(
                stream, mic, WAKE_FRAME_SAMPLES
            )
            overflow_count += int(overflowed)
            if len(frame) != WAKE_FRAME_SAMPLES:
                continue

            scores = wake_detector.predict(frame)
            score = max((float(value) for value in scores.values()), default=0.0)
            latest_scores = {name: float(value) for name, value in scores.items()}
            frame_rms = audio_metrics(frame)["rms"]
            max_score_since_debug = max(max_score_since_debug, score)
            max_rms_since_debug = max(max_rms_since_debug, frame_rms)

            now = time.monotonic()
            if show_scores and now - last_debug_at >= 1.0:
                score_summary = ", ".join(
                    f"{name}: {value:.3f}" for name, value in latest_scores.items()
                )
                print(
                    f"Wake scores: {score_summary} | max: "
                    f"{max_score_since_debug:.3f} | input RMS: "
                    f"{max_rms_since_debug:.4f}"
                )
                last_debug_at = now
                max_score_since_debug = 0.0
                max_rms_since_debug = 0.0

            if score >= WAKE_THRESHOLD:
                consecutive_hits += 1
                if consecutive_hits >= WAKE_CONSECUTIVE_HITS:
                    if overflow_count:
                        print(f"Wake listener had {overflow_count} input overflow(s).")
                    print("\nWake word detected!")
                    return
            else:
                consecutive_hits = 0

            if wake_whisper is None:
                continue

            frame_has_voice = frame_rms >= ASR_WAKE_RMS_THRESHOLD
            if not asr_active:
                asr_pre_roll.append(frame)
                asr_recent_voice.append(frame_has_voice)
                if sum(asr_recent_voice) >= 2:
                    asr_active = True
                    asr_started_at = now
                    asr_last_voice_at = now
                    asr_recorded = list(asr_pre_roll)
                continue

            asr_recorded.append(frame)
            if frame_has_voice:
                asr_last_voice_at = now

            assert asr_started_at is not None
            assert asr_last_voice_at is not None
            phrase_finished = (
                now - asr_last_voice_at >= ASR_WAKE_SILENCE_SECONDS
                or now - asr_started_at >= ASR_WAKE_MAX_SECONDS
            )
            if not phrase_finished:
                continue

            candidate = np.concatenate(asr_recorded)
            asr_recorded = []
            asr_active = False
            asr_pre_roll.clear()
            asr_recent_voice.clear()

            if _asr_detects_jarvis(wake_whisper, candidate, show_scores):
                print("\nWake phrase detected through local speech fallback!")
                return
def _adaptive_speech_threshold(noise_samples: deque[float]) -> float:
    if not noise_samples:
        return MIN_AUDIO_RMS
    noise_floor = float(np.median(np.asarray(noise_samples, dtype=np.float32)))
    return max(MIN_AUDIO_RMS, min(0.05, noise_floor * NOISE_MULTIPLIER + 0.0005))


def record_command(mic: Microphone) -> np.ndarray:
    """
    Record a natural spoken command. The VAD is used only to find boundaries;
    every frame between the pre-roll and post-speech tail is preserved.
    """
    print("\nJARVIS ACTIVATED!")
    try:
        winsound.Beep(1000, 150)
    except RuntimeError:
        pass

    # Let the beep decay before opening the microphone, avoiding an audio-feedback
    # false start on laptop speakers.
    time.sleep(0.12)
    print("Speak normally. JARVIS waits for you to finish.")

    vad = webrtcvad.Vad(VAD_MODE)
    pre_roll = deque(
        maxlen=max(1, round(PRE_ROLL_SECONDS * 1000 / COMMAND_FRAME_MS))
    )
    recent_speech = deque(maxlen=5)
    noise_samples: deque[float] = deque(maxlen=40)

    recorded: list[np.ndarray] = []
    speech_started = False
    speech_started_at: Optional[float] = None
    last_voice_at: Optional[float] = None
    overflow_count = 0
    waiting_started_at = time.monotonic()

    with _open_input_stream(mic, COMMAND_FRAME_SAMPLES) as stream:
        while True:
            frame, overflowed = _read_processing_frame(
                stream, mic, COMMAND_FRAME_SAMPLES
            )
            overflow_count += int(overflowed)
            if len(frame) != COMMAND_FRAME_SAMPLES:
                continue

            frame_rms = audio_metrics(frame)["rms"]
            vad_speech = vad.is_speech(frame.tobytes(), PROCESSING_SAMPLE_RATE)

            # Learn a short room-noise baseline before trusting level alone.
            # WebRTC VAD remains active during this period, so normal immediate
            # speech is still detected and pre-roll preserves its beginning.
            calibrating_noise = (
                not speech_started and len(noise_samples) < NOISE_CALIBRATION_FRAMES
            )
            if calibrating_noise:
                noise_samples.append(frame_rms)
                energy_speech = False
            else:
                energy_speech = frame_rms >= _adaptive_speech_threshold(noise_samples)

            is_speech = vad_speech or energy_speech
            now = time.monotonic()

            if not speech_started and not is_speech and not calibrating_noise:
                noise_samples.append(frame_rms)

            if not speech_started:
                pre_roll.append(frame)
                recent_speech.append(is_speech)

                # Two likely speech frames within 150 ms is responsive without
                # treating a single click or bump as a command.
                if sum(recent_speech) >= 2:
                    speech_started = True
                    speech_started_at = now
                    last_voice_at = now
                    recorded.extend(pre_roll)
                    print("Speech detected. Listening...")
                    continue

                if now - waiting_started_at >= MAX_WAIT_SECONDS:
                    print("No speech detected.")
                    return np.array([], dtype=np.float32)

                continue

            recorded.append(frame)
            if is_speech:
                last_voice_at = now

            assert speech_started_at is not None
            assert last_voice_at is not None
            speech_duration = now - speech_started_at

            if (
                speech_duration >= MIN_COMMAND_SECONDS
                and now - last_voice_at >= END_SILENCE_SECONDS
            ):
                print("End of speech detected.")
                break

            if speech_duration >= MAX_COMMAND_SECONDS:
                print("Maximum recording time reached.")
                break

    if not recorded:
        return np.array([], dtype=np.float32)

    captured = np.concatenate(recorded)
    print_audio_metrics("Captured command", captured, overflow_count)
    return captured.astype(np.float32) / 32768.0


# =====================================================
# SPEECH TO TEXT
# =====================================================

def _prepare_audio_for_whisper(audio: np.ndarray) -> np.ndarray:
    prepared = np.asarray(audio, dtype=np.float32).copy()
    if len(prepared) == 0:
        return prepared

    # Removing the small DC offset some Windows drivers add improves VAD features.
    prepared -= float(np.mean(prepared))
    peak = float(np.max(np.abs(prepared)))
    rms = float(np.sqrt(np.mean(np.square(prepared, dtype=np.float64))))

    if rms < MIN_AUDIO_RMS:
        return np.array([], dtype=np.float32)

    # Carefully boost a quiet but non-silent recording. The cap avoids magnifying
    # room noise into a false command.
    if 0.004 < peak < 0.15:
        gain = min(4.0, 0.25 / peak)
        prepared *= gain
        print(f"Boosted quiet input by {gain:.1f}x for transcription.")

    return np.clip(prepared, -1.0, 1.0)


def _looks_like_repetitive_hallucination(text: str) -> bool:
    """Reject obvious non-speech loops such as a repeated tone becoming one glyph."""
    compact = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
    if len(compact) >= 4 and len(set(compact)) == 1:
        return True

    tokens = normalize_command(text).split()
    return len(tokens) >= 3 and len(set(tokens)) == 1


def _has_meaningful_spoken_text(text: str) -> bool:
    """A symbol, whitespace, or control-like transcript is never a command."""
    words = re.findall(r"[A-Za-z]+|[\u0900-\u097F]+", text)
    return any(len(word) >= 2 for word in words)


def _transcribe_once(
    whisper: WhisperModel, audio: np.ndarray, language: Optional[str]
) -> tuple[str, object, list[object], float, float]:
    """Run one local Whisper pass and return transcript plus confidence signals."""
    temperature: float | list[float]
    temperature = [0.0, 0.2, 0.4] if language is None else 0.0

    segments_iterator, info = whisper.transcribe(
        audio,
        language=language,
        task="transcribe",
        beam_size=WHISPER_BEAM_SIZE,
        best_of=WHISPER_BEAM_SIZE,
        temperature=temperature,
        condition_on_previous_text=False,
        initial_prompt=TRANSCRIPTION_PROMPT,
        hotwords=(
            "Jarvis, Notepad, Calculator, Google, Google Chrome, Chrome, "
            "YouTube, BMW, F450GS"
        ),
        vad_filter=False,
        no_speech_threshold=0.60,
        log_prob_threshold=-1.2,
        compression_ratio_threshold=2.4,
        hallucination_silence_threshold=1.0,
    )
    segments = list(segments_iterator)
    text = " ".join(
        segment.text.strip() for segment in segments if segment.text.strip()
    ).strip()
    average_logprob = float(
        np.mean([getattr(segment, "avg_logprob", -10.0) for segment in segments])
    )
    highest_no_speech = max(
        (float(getattr(segment, "no_speech_prob", 0.0)) for segment in segments),
        default=0.0,
    )
    return text, info, segments, average_logprob, highest_no_speech


def _transcription_score(
    text: str, info: object, average_logprob: float, highest_no_speech: float
) -> float:
    """Rank only useful English/Hindi candidates after an auto-detect failure."""
    if not _has_meaningful_spoken_text(text):
        return -1000.0
    if _looks_like_repetitive_hallucination(text):
        return -1000.0

    detected = str(getattr(info, "language", "unknown"))
    language_bonus = 0.2 if detected in {"en", "hi"} else -1.0
    return average_logprob - highest_no_speech + language_bonus


def recognize_speech(
    whisper: WhisperModel, audio: np.ndarray, language: Optional[str]
) -> str:
    prepared = _prepare_audio_for_whisper(audio)
    if len(prepared) == 0:
        print("No usable speech was captured.")
        return ""

    print("\nUnderstanding your speech...")
    result = _transcribe_once(whisper, prepared, language)
    recognized_text, info, _, average_logprob, highest_no_speech = result
    detected_language = str(getattr(info, "language", "unknown"))

    # The assistant is designed for English, Hindi, and Hinglish. If automatic
    # detection produces another language or non-words, retry locally in the two
    # supported languages before accepting anything.
    needs_retry = (
        language is None
        and (
            detected_language not in {"en", "hi"}
            or not _has_meaningful_spoken_text(recognized_text)
            or _looks_like_repetitive_hallucination(recognized_text)
        )
    )
    if needs_retry:
        print("Language uncertain; retrying locally in English and Hindi...")
        candidates = [
            _transcribe_once(whisper, prepared, "en"),
            _transcribe_once(whisper, prepared, "hi"),
        ]
        result = max(
            candidates,
            key=lambda candidate: _transcription_score(
                candidate[0], candidate[1], candidate[3], candidate[4]
            ),
        )
        recognized_text, info, _, average_logprob, highest_no_speech = result
        detected_language = str(getattr(info, "language", "unknown"))

    language_probability = float(getattr(info, "language_probability", 0.0))
    print(
        f"Language: {detected_language} "
        f"({language_probability * 100:.0f}% confidence)"
    )

    if (
        not _has_meaningful_spoken_text(recognized_text)
        or _looks_like_repetitive_hallucination(recognized_text)
    ):
        print("JARVIS could not understand that. Please try again.")
        return ""

    spoken_words = re.findall(r"[A-Za-z]+|[\u0900-\u097F]+", recognized_text)
    if average_logprob < -1.3 and (
        highest_no_speech > 0.45 or len(spoken_words) <= 1
    ):
        print("Recognition confidence was too low. Please repeat the command.")
        return ""

    print(f"\nYOU: {recognized_text}")
    return recognized_text
# =====================================================
# SAFE COMPUTER CONTROL
# =====================================================

@dataclass(frozen=True)
class PendingAction:
    description: str
    url: Optional[str]
    expires_at: float
    action: str = "browser"
    query: Optional[str] = None
    response_language: str = "en"


def _extract_web_search_query(user_text: str) -> Optional[str]:
    """Recognize an explicit live-web search request in English or Hindi."""
    command = normalize_command(user_text)
    patterns = (
        r"^(?:search|find|look up)\s+(?:the\s+)?(?:web|internet|online)(?:\s+for)?\s+(.+)$",
        r"^(?:search|find)\s+(.+?)\s+(?:on|across)\s+(?:the\s+)?(?:web|internet)$",
        r"^(?:web|internet)\s+(?:par|pe)\s+(?:search(?:\s+karo)?|khojo|dhoondho|dhundho|dhoondo|dhundo)\s+(.+)$",
        r"^(?:वेब|इंटरनेट)\s+(?:पर|पे)\s+(?:सर्च\s+करो|खोजो|ढूँढो|ढूंढो|ढूंढें)\s+(.+)$",
        r"^(?:hindi\s+mein|hindi\s+me)\s+(?:search\s+karo|khojo|dhundho|dhoondo)\s+(.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, command, flags=re.IGNORECASE)
        if match:
            query = match.group(1).strip()
            query = re.sub(
                r"\s+(?:in\s+hindi|hindi\s+mein|hindi\s+me|हिंदी\s+में|हिन्दी\s+में|in\s+english|english\s+mein|english\s+me)$",
                "",
                query,
                flags=re.IGNORECASE,
            )
            return query.strip()
    return None


def _show_search_sources(sources: list) -> None:
    print("\nWEB SOURCES — open links only if you choose to:")
    for index, source in enumerate(sources, start=1):
        print(
            f"[{index}] {source.title}\n"
            f"    {source.url}\n"
            f"    {source.snippet}"
        )


def normalize_command(text: str) -> str:
    cleaned = re.sub(r"[^\w\s\u0900-\u097F]", " ", text.casefold())
    return " ".join(cleaned.split())


def _has_open_intent(command: str) -> bool:
    return (
        command.startswith(("open ", "launch ", "start ", "kholo ", "खोलो "))
        or command.endswith((" kholo", " खोलो"))
    )


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _is_yes(command: str) -> bool:
    return command in {"yes", "yeah", "yep", "haan", "ha", "हाँ", "हां", "confirm"}


def _is_no(command: str) -> bool:
    return command in {"no", "nope", "nah", "nahi", "nahin", "नहीं", "नहि", "cancel"}


def _find_chrome() -> Optional[Path]:
    for candidate in CHROME_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def _open_allowed_url(url: str) -> bool:
    """Open only the fixed Google/YouTube pages constructed by this module."""
    if not any(url.startswith(prefix) for prefix in ALLOWED_BROWSER_URL_PREFIXES):
        print(f"Refused unexpected browser URL: {url!r}")
        return False

    try:
        chrome = _find_chrome()
        if chrome is not None:
            subprocess.Popen([str(chrome), "--new-tab", url], shell=False)
            return True
        return bool(webbrowser.open(url, new=2))
    except OSError as error:
        print(f"Browser action error: {error}")
        return False


def launch_safe_action(action: str) -> str:
    """
    Only fixed executables, folders, and approved URLs are used here. No
    transcript or model output is passed to a shell or executable.
    """
    try:
        if action == "notepad":
            subprocess.Popen(["notepad.exe"], shell=False)
            return "Opening Notepad."

        if action == "calculator":
            subprocess.Popen(["calc.exe"], shell=False)
            return "Opening Calculator."

        if action == "explorer":
            subprocess.Popen(["explorer.exe"], shell=False)
            return "Opening File Explorer."

        if action == "settings":
            os.startfile("ms-settings:")  # type: ignore[attr-defined]
            return "Opening Settings."

        if action == "jarvis_folder":
            os.startfile(str(BASE_DIR))  # type: ignore[attr-defined]
            return "Opening the JARVIS folder."

        if action == "chrome":
            chrome = _find_chrome()
            if chrome is None:
                return "Google Chrome could not be found."
            subprocess.Popen([str(chrome)], shell=False)
            return "Opening Chrome."

        if action == "google":
            if _open_allowed_url(GOOGLE_HOME_URL):
                return "Opening Google in Chrome."
            return "I could not open the browser."

        if action == "youtube":
            if _open_allowed_url(YOUTUBE_HOME_URL):
                return "Opening YouTube in Chrome."
            return "I could not open the browser."

    except OSError as error:
        print(f"Computer action error: {error}")
        return "Sorry, I could not complete that action."

    return "That action is not available."


def _extract_youtube_video_query(command: str) -> Optional[str]:
    """
    Accept only an explicit YouTube video request, then keep the query bounded
    and URL-encode it before it reaches a browser.
    """
    if "youtube" not in command or "video" not in command:
        return None
    if re.search(
        r"\b(?:search|find|look\s+up)\s+(?:the\s+)?(?:web|internet|online)\b",
        command,
    ):
        return None

    match = re.search(
        r"\b(?:find|search(?:\s+for)?|look\s+for|show)\s+"
        r"(?:me\s+)?(?:a\s+)?"
        r"(?:video(?:\s+(?:of|about|for))?\s*)?"
        r"(?P<query>.+)$",
        command,
    )
    if match is None:
        return None

    query = match.group("query").strip(" .,!?:")
    query = re.sub(r"\s+on\s+youtube$", "", query)
    if not 2 <= len(query) <= 160:
        return None
    return query


def control_computer(
    user_text: str, pending: Optional[PendingAction]
) -> tuple[Optional[str], Optional[PendingAction]]:
    """
    Return (reply, pending_action). Fixed app launches happen immediately.
    Google/YouTube queries require a nearby spoken yes/no before any query text
    is sent to the site.
    """
    command = normalize_command(user_text)
    now = time.monotonic()

    if pending is not None:
        if now > pending.expires_at:
            pending = None
            print("The pending action expired.")
        elif _is_yes(command):
            if pending.action == "serpapi":
                if not pending.query:
                    return "I could not read that search question. Please try again.", None
                return (
                    run_serpapi_search(pending.query, pending.response_language),
                    None,
                )
            if _open_allowed_url(pending.url):
                return f"Confirmed. {pending.description}", None
            return "I could not open the browser.", None
        elif _is_no(command):
            return "Canceled.", None
        else:
            # A new command cancels an old confirmation rather than unexpectedly
            # carrying it forward.
            pending = None

    blocked_phrases = (
        "delete",
        "erase",
        "format",
        "shutdown",
        "restart",
        "reboot",
        "lock computer",
        "lock the computer",
        "close all",
        "kill ",
        "powershell",
        "command prompt",
        "terminal",
        "install",
        "uninstall",
        "send email",
        "send message",
    )
    if _contains_any(command, blocked_phrases):
        return (
            "For safety, voice control cannot delete data, run terminals, install "
            "software, power off the computer, or send messages.",
            None,
        )

    if command in {"what time is it", "time", "current time", "समय क्या है"}:
        return time.strftime("It is %I:%M %p."), None

    if command in {"what date is it", "date", "today s date", "आज की तारीख"}:
        return time.strftime("Today is %A, %d %B %Y."), None

    if command in {"computer controls", "voice controls", "help controls"}:
        return (
            "You can ask me to open Chrome, Notepad, Calculator, File Explorer, "
            "Settings, the JARVIS folder, Google, or YouTube. Say search the web "
            "for, and I will ask before sending the question to SerpApi. For a "
            "YouTube search, I will also ask first.",
            None,
        )

    # This composite pattern must be checked before the one-action commands
    # below so "open Chrome then open YouTube then find a video..." is handled
    # as the intended sequence.
    video_query = _extract_youtube_video_query(command)
    if video_query is not None:
        setup_replies: list[str] = []
        if "chrome" in command:
            setup_replies.append(launch_safe_action("chrome"))
        setup_replies.append(launch_safe_action("youtube"))

        url = YOUTUBE_SEARCH_URL + quote_plus(video_query)
        setup = " ".join(setup_replies)
        return (
            f"{setup} I can search YouTube for {video_query}. "
            "This sends those words to YouTube. Wake me again, wait for the beep, "
            "then say yes to continue or no to cancel.",
            PendingAction(
                description=f"Searching YouTube for {video_query}.",
                url=url,
                expires_at=now + 30.0,
            ),
        )

    web_query = _extract_web_search_query(user_text)
    if web_query:
        language = detect_response_language(user_text)
        language_name = "Hindi" if language == "hi" else "English"
        return (
            f"I can search the web for {web_query} through SerpApi and summarize "
            f"the results in {language_name}. This sends your search words to "
            "SerpApi and its search provider. Wake me again, then say yes to "
            "continue or no to cancel.",
            PendingAction(
                description=f"Searching the web for {web_query}.",
                url=None,
                expires_at=now + 30.0,
                action="serpapi",
                query=web_query,
                response_language=language,
            ),
        )

    if _has_open_intent(command):
        if "chrome" in command:
            return launch_safe_action("chrome"), None
        if _contains_any(command, ("notepad", "नोटपैड")):
            return launch_safe_action("notepad"), None
        if _contains_any(command, ("calculator", "कैलकुलेटर")):
            return launch_safe_action("calculator"), None
        if _contains_any(command, ("file explorer", "explorer", "फाइल एक्सप्लोरर")):
            return launch_safe_action("explorer"), None
        if _contains_any(command, ("settings", "सेटिंग्स")):
            return launch_safe_action("settings"), None
        if _contains_any(command, ("jarvis folder", "jarvis")) and "folder" in command:
            return launch_safe_action("jarvis_folder"), None
        if "youtube" in command:
            return launch_safe_action("youtube"), None
        if "google" in command:
            return launch_safe_action("google"), None

    # A generic Google search also requires nearby spoken confirmation.
    search_match = re.match(
        r"^\s*(?:search(?:\s+for)?|google\s+search(?:\s+for)?)\s+(.+?)\s*$",
        user_text,
        flags=re.IGNORECASE,
    )
    if search_match:
        query = search_match.group(1).strip()
        if query:
            url = "https://www.google.com/search?q=" + quote_plus(query)
            return (
                f"I can search Google for {query}. This sends those words to "
                "Google. Wake me again, wait for the beep, then say yes to "
                "continue or no to cancel.",
                PendingAction(
                    description=f"Searching Google for {query}.",
                    url=url,
                    expires_at=now + 30.0,
                ),
            )

    return None, None


def run_serpapi_search(query: str, language: str) -> str:
    """Run an already-confirmed search and return a short spoken response."""
    try:
        answer, sources = search_and_summarize(
            query,
            language,
            model=AI_MODEL,
            ollama_url=OLLAMA_URL,
        )
    except SearchDemoError as error:
        return str(error)

    _show_search_sources(sources)
    print("\nJARVIS SEARCH ANSWER:")
    print(answer)
    spoken_answer = re.sub(r"\[(\d+)\]", r"source \1", answer)
    if language == "hi":
        return f"{spoken_answer} स्रोतों के लिंक स्क्रीन पर दिख रहे हैं।"
    return f"{spoken_answer} The source links are shown on screen."


# =====================================================
# LOCAL AI AND SPEECH
# =====================================================

def make_voice() -> pyttsx3.Engine:
    voice = pyttsx3.init()
    voice.setProperty("rate", 165)
    voice.setProperty("volume", 1.0)
    return voice


def speak(voice: pyttsx3.Engine, text: str) -> None:
    if not text:
        return

    print(f"\nJARVIS: {text}")
    default_voice_id = None
    try:
        default_voice_id = voice.getProperty("voice")
        if re.search(r"[\u0900-\u097f]", text):
            for candidate in voice.getProperty("voices"):
                voice_details = " ".join(
                    [
                        str(getattr(candidate, "id", "")),
                        str(getattr(candidate, "name", "")),
                        " ".join(str(item) for item in getattr(candidate, "languages", [])),
                    ]
                )
                if re.search(r"hindi|\bhi[_-][a-z]{2}\b", voice_details, re.IGNORECASE):
                    voice.setProperty("voice", candidate.id)
                    break
        voice.say(text)
        voice.runAndWait()
    except Exception as error:
        print(f"Text-to-speech error: {error}")
    finally:
        if default_voice_id:
            try:
                voice.setProperty("voice", default_voice_id)
            except Exception:
                pass


def run_typed_search_demo() -> int:
    """Run the live-search flow without a microphone for a reliable demo."""
    try:
        voice = make_voice()
    except Exception as error:
        print(f"Text-to-speech setup error: {error}")
        return 2

    print("\nJARVIS + SERPAPI LIVE SEARCH DEMO")
    print("Type a question in English or Hindi. Type 'quit' to leave.")
    print("JARVIS will ask before sending each question to SerpApi.")
    while True:
        try:
            question = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDemo closed.")
            return 0

        if question.casefold() in {"quit", "exit", "q"}:
            print("Demo closed.")
            return 0
        if not question:
            continue

        language = detect_response_language(question)
        print(
            "This will send your question to SerpApi and its search provider. "
            "The result excerpts will be summarized by your local Ollama model."
        )
        try:
            confirmation = input("Continue? Type yes to search, or no to cancel: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDemo closed.")
            return 0
        if normalize_command(confirmation) not in {"yes", "y", "haan", "हाँ", "हां"}:
            print("Canceled. No search request was sent.")
            continue

        speak(voice, run_serpapi_search(question, language))


def ask_jarvis(user_text: str) -> str:
    global conversation_history

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += conversation_history[-10:]
    messages.append({"role": "user", "content": user_text})

    print("\nJARVIS is thinking...")
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": AI_MODEL,
            "messages": messages,
            "stream": False,
            "think": False,
        },
        timeout=180,
    )
    response.raise_for_status()

    answer = response.json()["message"]["content"].strip()
    conversation_history.extend(
        [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer},
        ]
    )
    conversation_history = conversation_history[-20:]
    return answer


def load_wake_detector() -> Model:
    print("Loading wake-word engine...")
    # Keep the user's custom model, but pair it with the maintained built-in
    # model so a broken custom model never makes JARVIS completely unreachable.
    wake_models = ["hey_jarvis"]
    if CUSTOM_WAKE_MODEL.exists():
        wake_models.insert(0, str(CUSTOM_WAKE_MODEL))
    return Model(wakeword_models=wake_models, inference_framework="onnx")


def load_whisper_model(model_name: str) -> WhisperModel:
    print(f"Loading speech recognition model: {model_name}")
    model_args = {
        "device": "cpu",
        "compute_type": "int8",
        "cpu_threads": WHISPER_CPU_THREADS,
        "num_workers": 1,
    }
    try:
        return WhisperModel(model_name, **model_args)
    except Exception as error:
        if model_name == FALLBACK_WHISPER_MODEL:
            raise
        print(
            f"Could not load {model_name!r}: {error}\n"
            f"Falling back to the installed {FALLBACK_WHISPER_MODEL!r} model."
        )
        return WhisperModel(FALLBACK_WHISPER_MODEL, **model_args)


def load_wake_fallback_model(main_model: WhisperModel, main_model_name: str) -> WhisperModel:
    """Use the faster base model for the short, local-only plain-Jarvis fallback."""
    if main_model_name.casefold() == WAKE_FALLBACK_MODEL:
        return main_model

    print(f"Loading local wake fallback model: {WAKE_FALLBACK_MODEL}")
    return WhisperModel(
        WAKE_FALLBACK_MODEL,
        device="cpu",
        compute_type="int8",
        cpu_threads=WHISPER_CPU_THREADS,
        num_workers=1,
    )


# =====================================================
# PROGRAM ENTRY
# =====================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local JARVIS with safer microphone and Windows controls."
    )
    parser.add_argument(
        "--search-demo",
        action="store_true",
        help="Run the typed SerpApi live-search demo without the microphone.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List microphone inputs without recording audio.",
    )
    parser.add_argument(
        "--mic",
        help=(
            "Microphone number or part of its name. Overrides JARVIS_MIC_DEVICE "
            "for this run."
        ),
    )
    parser.add_argument(
        "--mic-test",
        action="store_true",
        help="Run an explicit, in-memory microphone level test; no audio is saved.",
    )
    parser.add_argument(
        "--mic-test-seconds",
        type=float,
        default=6.0,
        help="Length of the opt-in microphone test (1 to 20 seconds).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_WHISPER_MODEL,
        help=(
            "faster-whisper model to use (default: %(default)s). Use base for "
            "faster responses or small for better Hindi/English accuracy."
        ),
    )
    parser.add_argument(
        "--language",
        choices=("auto", "hi", "en"),
        default=os.environ.get("JARVIS_LANGUAGE", "auto"),
        help="Keep auto for Hindi/English/Hinglish; force hi or en only to test.",
    )
    parser.add_argument(
        "--wake-debug",
        action="store_true",
        help="Print wake-model scores and local audio level once per second.",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show the selected microphone and configuration, then exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    if args.list_devices:
        list_microphones()
        return 0

    if args.search_demo:
        return run_typed_search_demo()

    try:
        mic = choose_microphone(args.mic)
    except Exception as error:
        print(f"Microphone setup error: {error}")
        return 2

    print("\n===================================")
    print("       JARVIS AI SYSTEM v4.1")
    print("===================================")
    print(
        f"Selected microphone: [{mic.index}] {mic.name} | {mic.hostapi} | "
        f"{mic.capture_rate // 1000} kHz"
    )
    if mic.is_resampled:
        print(f"Local resampling: {mic.capture_rate // 1000} kHz -> 16 kHz")

    if args.show_config:
        print(f"Whisper model: {args.model}")
        print(f"Whisper CPU threads: {WHISPER_CPU_THREADS}")
        print(f"Language mode: {args.language}")
        print(f"Wake threshold: {WAKE_THRESHOLD}")
        return 0

    if args.mic_test:
        microphone_test(mic, args.mic_test_seconds)
        return 0

    try:
        wake_detector = load_wake_detector()
        whisper = load_whisper_model(args.model)
        wake_whisper = load_wake_fallback_model(whisper, args.model)
        voice = make_voice()
    except Exception as error:
        print(f"Startup error: {error}")
        return 3

    language = None if args.language == "auto" else args.language
    pending_action: Optional[PendingAction] = None

    print("\nJARVIS IS ONLINE")
    print("Features: wake word, adaptive recording, Hindi/English/Hinglish ASR,")
    print("local Ollama chat, confirmed SerpApi search, and safe computer control.")
    print("Wake options: 'Jarvis' (local speech fallback) or 'Hey Jarvis'.")
    print("Press Ctrl+C to stop JARVIS.")

    while True:
        try:
            wait_for_wake_word(
                mic, wake_detector, wake_whisper, args.wake_debug
            )

            audio = record_command(mic)
            user_text = recognize_speech(whisper, audio, language)
            if not user_text:
                print("\nNo valid command. Returning to standby.")
                continue

            command = normalize_command(user_text)
            if command in {"stop jarvis", "exit jarvis", "shutdown jarvis"}:
                speak(voice, "Shutting down JARVIS. Goodbye.")
                return 0

            control_reply, pending_action = control_computer(
                user_text, pending_action
            )
            if control_reply is not None:
                speak(voice, control_reply)
                time.sleep(0.6)
                continue

            answer = ask_jarvis(user_text)
            speak(voice, answer)

            # Avoid re-triggering on JARVIS's own final speech audio.
            time.sleep(0.6)

        except KeyboardInterrupt:
            print("\nJARVIS stopped.")
            return 0
        except requests.RequestException as error:
            print(f"\nOllama connection error: {error}")
            print("Check that Ollama is running, then JARVIS will return to standby.")
            time.sleep(1.0)
        except Exception as error:
            print(f"\nJARVIS error: {error}")
            print("Returning to standby...")
            time.sleep(1.0)


if __name__ == "__main__":
    sys.exit(main())
