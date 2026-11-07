"""A small online assistant session shared by user-facing interfaces."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from online_gemini import OnlineServiceError
from online_provider import create_online_client
from reply_language import response_language, language_instruction, localize_control_reply


PROMPT = """You are JARVIS, a helpful personal assistant. Understand English,
Hindi, and Hinglish, and respond in the user's preferred language. Be useful,
clear, and honest about uncertainty. You cannot see the screen, access private
files, browse the web, or perform computer actions through this conversation.
Never claim an action or a web search happened unless the application reports it.
Do not invent recent facts. Ask a short question when the request is unclear."""


class AssistantSession:
    """Route fixed actions locally and ordinary conversation to Gemini."""

    def __init__(self, base_dir: Path, router: Callable | None = None,
                 online_client=None):
        self.base_dir = base_dir
        self._router = router
        self._client = online_client
        self._history: list[dict[str, str]] = []
        self.pending = None
        self.pending_language = None

    def respond(self, text: str) -> str:
        text = text.strip()
        if not text:
            return 'Please type a message.'
        if self._router is None:
            # Delay heavyweight audio imports until a command is submitted.
            from jarvis import control_computer
            self._router = control_computer
        prior_pending = self.pending
        language = response_language(text)
        if prior_pending is not None and text.casefold().strip(' .!?') in {'yes','no','haan','nahi','हाँ','हां','नहीं'}:
            language = self.pending_language or language
        reply, self.pending = self._router(text, self.pending)
        self.pending_language = language if self.pending is not None else None
        if reply is not None:
            # Voice instructions in the older router are not suitable for typing.
            reply = reply.replace(
                'Wake me again, wait for the beep, then say yes to continue or no to cancel.',
                'Type yes to continue or no to cancel.'
            ).replace(
                'Wake me again, then say yes to continue or no to cancel.',
                'Type yes to continue or no to cancel.'
            )
            return localize_control_reply(reply, language)
        if self._client is None:
            self._client = create_online_client(self.base_dir)
            if self._client is None:
                return 'Online setup is missing. Add a Groq or Gemini API key to your JARVIS settings.'
        messages = self._history[-10:] + [{'role': 'user', 'content': text}]
        try:
            answer = self._client.chat(PROMPT + '\n' + language_instruction(language), messages)
        except OnlineServiceError as error:
            return str(error)
        self._history.extend([{'role': 'user', 'content': text},
                              {'role': 'assistant', 'content': answer}])
        self._history = self._history[-20:]
        return answer

