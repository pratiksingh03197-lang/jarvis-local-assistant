"""Optional Gemini cloud services for Project JARVIS.

The microphone and all computer-control decisions remain in ``jarvis.py``.
This module is deliberately limited to two cloud operations after the user has
activated JARVIS: transcribing an in-memory command and generating a text
reply.  It never stores a recording on the laptop and it has no Windows or
browser-control capability.
"""

from __future__ import annotations

import io
import base64
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from google import genai
    from google.genai import types
except ImportError:  # Keep local mode usable if the optional package is absent.
    genai = None
    types = None


GEMINI_TRANSCRIPTION_MODEL = "gemini-3.5-transcribe"
GEMINI_CHAT_MODEL = "gemini-3.5-flash-lite"
COMMAND_SAMPLE_RATE = 16_000
COMMAND_VOCABULARY = (
    "Jarvis",
    "JARVIS",
    "Chrome",
    "Google Chrome",
    "YouTube",
    "BMW",
    "F450GS",
    "Notepad",
    "Calculator",
    "File Explorer",
    "SerpApi",
    "Ollama",
)


class OnlineServiceError(RuntimeError):
    """A safe, key-free explanation of an online-service failure."""


class OnlineSetupError(RuntimeError):
    """Raised when the user asks for cloud-only mode without a usable setup."""


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str
    transcription_model: str = GEMINI_TRANSCRIPTION_MODEL
    chat_model: str = GEMINI_CHAT_MODEL


def _value_from_dotenv(base_dir: Path, name: str) -> str:
    """Read one simple key/value from the project's ignored .env file."""
    env_file = base_dir / ".env"
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""

    for line in lines:
        candidate = line.strip()
        if not candidate or candidate.startswith("#"):
            continue
        if candidate.startswith("export "):
            candidate = candidate[7:].lstrip()
        key, separator, value = candidate.partition("=")
        if separator and key.strip() == name:
            return value.strip().strip('"').strip("'")
    return ""


def load_gemini_config(base_dir: Path) -> GeminiConfig | None:
    """Find a Gemini key without printing or exposing it."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(name, "").strip() or _value_from_dotenv(base_dir, name)
        if value and value.casefold() not in {
            "your_gemini_api_key_here",
            "replace_me",
        }:
            return GeminiConfig(api_key=value)
    return None


def _as_wav_buffer(audio: np.ndarray) -> io.BytesIO:
    """Encode 16 kHz mono audio in memory; no recording is written to disk."""
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not len(samples):
        raise OnlineServiceError("JARVIS did not capture any usable audio.")

    pcm = np.clip(samples, -1.0, 1.0)
    pcm = np.rint(pcm * 32767.0).astype("<i2", copy=False)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(COMMAND_SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())
    buffer.seek(0)
    return buffer


def _as_gemini_contents(messages: Sequence[dict[str, str]]) -> list[dict[str, object]]:
    """Translate the small local conversation history to Gemini's roles."""
    contents: list[dict[str, object]] = []
    for message in messages:
        text = str(message.get("content", "")).strip()
        role = str(message.get("role", "user"))
        if not text:
            continue
        contents.append(
            {
                "role": "model" if role == "assistant" else "user",
                "parts": [{"text": text}],
            }
        )
    return contents


class GeminiOnline:
    """Small, deliberately non-agentic Gemini client for JARVIS."""

    def __init__(self, config: GeminiConfig) -> None:
        if genai is None or types is None:
            raise OnlineSetupError(
                "Online mode needs the google-genai package. Run the project setup "
                "again, then start JARVIS."
            )
        self.config = config
        self._client = genai.Client(api_key=config.api_key)

    def transcribe_command(self, audio: np.ndarray, language: str | None = None) -> str:
        """Send one activated command inline, without a separate file upload."""
        try:
            payload = _as_wav_buffer(audio)
            interaction = self._client.interactions.create(
                model=self.config.transcription_model,
                input=[
                    {
                        "type": "audio",
                        "data": base64.b64encode(payload.getvalue()).decode("ascii"),
                        "mime_type": "audio/wav",
                    }
                ],
                generation_config={
                    "transcription_config": {
                        # Let Gemini switch naturally between Hindi and English.
                        "language_codes": {'hi': ['hi-IN'], 'en': ['en-IN']}.get(language, []),
                        "custom_vocabulary": list(COMMAND_VOCABULARY),
                        # Verbatim preserves pauses and self-corrections; smart
                        # mode can rewrite the meaning of an action command.
                    }
                },
            )
            answer = str(getattr(interaction, "output_text", "") or "").strip()
            if not answer:
                raise OnlineServiceError("The online speech service returned no text.")
            return answer
        except OnlineServiceError:
            raise
        except Exception as error:
            # Do not surface the raw exception: API libraries can include request
            # details, and a key must never be printed in a terminal or log.
            if getattr(error, "code", None) == 429:
                raise OnlineServiceError(
                    "The Gemini speech service reached its usage limit. "
                    "Wait for the quota to reset or check your account limits."
                ) from error
            raise OnlineServiceError(
                "JARVIS could not reach Gemini transcription. Check your internet "
                "connection and API key, then try again."
            ) from error

    def chat(self, system_prompt: str, messages: Sequence[dict[str, str]]) -> str:
        """Generate a text reply only.  No Gemini tools are enabled."""
        contents = _as_gemini_contents(messages)
        if not contents:
            raise OnlineServiceError("JARVIS did not receive a question to answer.")
        try:
            response = self._client.models.generate_content(
                model=self.config.chat_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.35,
                    max_output_tokens=350,
                ),
            )
            answer = str(getattr(response, "text", "") or "").strip()
            if not answer:
                raise OnlineServiceError("The online assistant returned no answer.")
            return answer
        except OnlineServiceError:
            raise
        except Exception as error:
            raise OnlineServiceError(
                "JARVIS could not reach Gemini chat. Check your internet connection "
                "and API key, then try again."
            ) from error

