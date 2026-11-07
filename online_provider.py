"""Choose the configured online provider without exposing credentials."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import requests

from online_gemini import (GeminiOnline, OnlineServiceError, OnlineSetupError,
                           _as_wav_buffer, _value_from_dotenv, load_gemini_config)

GROQ_BASE = 'https://api.groq.com/openai/v1'
# Favor accuracy over the somewhat faster Turbo model for personal commands.
GROQ_SPEECH_MODEL = 'whisper-large-v3'
GROQ_COMPARISON_MODEL = 'whisper-large-v3-turbo'
GROQ_CHAT_MODEL = 'qwen/qwen3.8-27b'


def groq_key(base_dir: Path) -> str:
    key = os.environ.get('GROQ_API_KEY', '').strip() or _value_from_dotenv(base_dir, 'GROQ_API_KEY')
    if key.casefold() in {'your_groq_api_key_here', 'replace_me'}:
        key = ''
    if not key:
        try:
            for line in (base_dir / '.groq.env').read_text(encoding='utf-8').splitlines():
                name, sep, value = line.partition('=')
                if sep and name.strip() == 'GROQ_API_KEY':
                    key = value.strip().strip('"').strip("'")
                    break
        except OSError:
            pass
    return '' if key.casefold() in {'your_groq_api_key_here', 'replace_me'} else key


class GroqOnline:
    name = 'Groq'

    def __init__(self, key: str):
        self._key = key
        self.last_transcription_language = None

    def _post(self, endpoint: str, **kwargs):
        try:
            response = requests.post(GROQ_BASE + endpoint,
                headers={'Authorization': 'Bearer ' + self._key}, timeout=(10, 60), **kwargs)
        except requests.RequestException:
            raise OnlineServiceError('Could not reach Groq. Check your internet connection.') from None
        if response.status_code == 429:
            raise OnlineServiceError('Groq has reached its free-plan limit. Wait for the limit to reset.')
        if response.status_code in (401, 403):
            raise OnlineServiceError('Groq rejected the API key or access to the model.')
        if response.status_code >= 400:
            raise OnlineServiceError(f'Groq returned an error (HTTP {response.status_code}).')
        try:
            return response.json()
        except ValueError:
            raise OnlineServiceError('Groq returned an unreadable response.') from None

    def transcribe_command(self, audio, language=None, model=None) -> str:
        wav = _as_wav_buffer(audio)
        model = model or GROQ_SPEECH_MODEL
        if model not in (GROQ_SPEECH_MODEL, GROQ_COMPARISON_MODEL):
            raise ValueError('Unsupported speech model.')
        fields = {'model': model, 'response_format': 'verbose_json',
                  'temperature': '0', 'prompt': 'Jarvis, Chrome, YouTube, Notepad, Calculator, BMW.'}
        if language in ('hi', 'en'):
            fields['language'] = language
        result = self._post('/audio/transcriptions',
            files={'file': ('command.wav', wav, 'audio/wav')},
            data=fields)
        self.last_transcription_language = result.get('language')
        answer = result.get('text', '').strip()
        if not answer:
            raise OnlineServiceError('Groq could not detect any words in the recording.')
        return answer

    def chat(self, system_prompt: str, messages: Sequence[dict[str, str]]) -> str:
        result = self._post('/chat/completions', json={
            'model': GROQ_CHAT_MODEL, 'messages': [{'role': 'system', 'content': system_prompt}] + list(messages),
            'temperature': 0.35, 'max_completion_tokens': 350, 'reasoning_effort': 'none',
        })
        try:
            answer = str(result['choices'][0]['message']['content']).strip()
        except (KeyError, IndexError, TypeError):
            raise OnlineServiceError('Groq returned an unreadable answer.') from None
        if not answer:
            raise OnlineServiceError('Groq returned an empty answer.')
        return answer


def create_online_client(base_dir: Path):
    """A Groq key opts into Groq; otherwise preserve existing Gemini setup."""
    key = groq_key(base_dir)
    if key:
        return GroqOnline(key)
    config = load_gemini_config(base_dir)
    return GeminiOnline(config) if config else None


def provider_name(client) -> str:
    return getattr(client, 'name', 'Gemini')

