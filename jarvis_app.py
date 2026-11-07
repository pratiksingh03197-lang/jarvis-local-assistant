"""Windows desktop front end for the existing JARVIS voice assistant."""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext, ttk
from assistant_session import AssistantSession
from speech_output import SpeechOutput

ROOT = Path(__file__).resolve().parent
BG, PANEL, TEXT, MUTED, ACCENT = '#09111e', '#111f30', '#ecf6ff', '#92a9bc', '#50d9ee'


class JarvisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('JARVIS — Personal Assistant')
        self.geometry('1000x720')
        self.minsize(700, 560)
        self.configure(bg=BG)
        self.process = None
        self.events = queue.Queue()
        self.session = AssistantSession(ROOT)
        self.speech = SpeechOutput(lambda message: self.events.put(('speech_error', message)))
        self.protocol('WM_DELETE_WINDOW', self.close_app)

        header = tk.Frame(self, bg=BG, padx=34, pady=22)
        header.pack(fill='x')
        tk.Label(header, text='◉  J A R V I S', bg=BG, fg=TEXT,
                 font=('Segoe UI', 18, 'bold')).pack(side='left')
        tk.Label(header, text='YOUR PERSONAL ASSISTANT', bg=BG, fg=MUTED,
                 font=('Segoe UI', 9)).pack(side='right')

        body = tk.Frame(self, bg=BG, padx=34)
        body.pack(fill='both', expand=True)
        hero = tk.Frame(body, bg=PANEL, padx=24, pady=20)
        hero.pack(fill='x')
        tk.Label(hero, text='Your assistant, ready when you are.', bg=PANEL, fg=TEXT,
                 font=('Segoe UI', 19, 'bold'), anchor='w').pack(fill='x')
        tk.Label(hero, text='Start listening, say “Jarvis”, wait for the beep, then speak.',
                 bg=PANEL, fg=MUTED, font=('Segoe UI', 10), anchor='w').pack(fill='x', pady=(5, 16))
        controls = tk.Frame(hero, bg=PANEL)
        controls.pack(fill='x')
        self.start_button = tk.Button(controls, text='  Start listening  ', command=self.start,
            bg=ACCENT, fg=BG, activebackground='#8eeefa', relief='flat', font=('Segoe UI', 10, 'bold'), padx=14, pady=8)
        self.start_button.pack(side='left')
        self.stop_button = tk.Button(controls, text='Stop', command=self.stop, state='disabled',
            bg='#24384c', fg=TEXT, disabledforeground=MUTED, relief='flat', font=('Segoe UI', 10), padx=18, pady=8)
        self.stop_button.pack(side='left', padx=10)
        self.check_button = tk.Button(controls, text='Test voice', command=self.start_voice_check,
            bg='#24384c', fg=TEXT, relief='flat', font=('Segoe UI', 10), padx=14, pady=8)
        self.check_button.pack(side='left')
        self.done_button = tk.Button(controls, text='Done speaking', command=self.finish_speaking,
            state='disabled', bg='#24384c', fg=TEXT, disabledforeground=MUTED,
            relief='flat', font=('Segoe UI', 10), padx=12, pady=8)
        self.done_button.pack(side='left', padx=10)
        language_bar = tk.Frame(hero, bg=PANEL)
        language_bar.pack(fill='x', pady=(12, 0))
        tk.Label(language_bar, text='Speech language:', bg=PANEL, fg=MUTED,
                 font=('Segoe UI', 9)).pack(side='left')
        self.language = tk.StringVar(value='Automatic')
        self.language_selector = ttk.Combobox(language_bar, textvariable=self.language,
            values=('Automatic', 'Hindi', 'English'), state='readonly', width=12)
        self.language_selector.pack(side='left', padx=8)
        self.status = tk.StringVar(value='OFFLINE  •  Microphone off')
        tk.Label(controls, textvariable=self.status, bg=PANEL, fg=ACCENT,
                 font=('Segoe UI', 10, 'bold')).pack(side='right')

        tk.Label(body, text='ACTIVITY', bg=BG, fg=MUTED, anchor='w',
                 font=('Segoe UI', 9, 'bold')).pack(fill='x', pady=(24, 8))
        self.log = scrolledtext.ScrolledText(body, bg=PANEL, fg=TEXT, insertbackground=TEXT,
            relief='flat', font=('Consolas', 10), padx=16, pady=14, wrap='word', state='disabled')
        self.log.pack(fill='both', expand=True)
        self.append('JARVIS: Hello. Type a message below or start voice listening.\n')
        composer = tk.Frame(body, bg=BG, pady=10)
        composer.pack(fill='x')
        self.message = tk.Entry(composer, bg='#1b3044', fg=TEXT, insertbackground=TEXT,
                                relief='flat', font=('Segoe UI', 11))
        self.message.pack(side='left', fill='x', expand=True, ipady=10, padx=(0, 10))
        self.message.bind('<Return>', self.send_message)
        self.send_button = tk.Button(composer, text='Send', command=self.send_message,
            bg=ACCENT, fg=BG, relief='flat', font=('Segoe UI', 10, 'bold'), padx=18, pady=8)
        self.send_button.pack(side='right')
        tk.Label(body, text='Messages and activated voice commands are sent to the configured online provider. Voice and typed conversations are separate.',
                 bg=BG, fg=MUTED, font=('Segoe UI', 9), anchor='w', wraplength=850).pack(fill='x', pady=14)
        self.after(100, self.poll)

    def append(self, text):
        self.log.configure(state='normal')
        self.log.insert('end', text)
        self.log.see('end')
        self.log.configure(state='disabled')

    def start(self):
        self._start_voice_process(voice_check=False)

    def start_voice_check(self):
        self._start_voice_process(voice_check=True)

    def _start_voice_process(self, voice_check):
        if self.process and self.process.poll() is None:
            return
        python = ROOT / '.venv' / 'Scripts' / 'python.exe'
        if not python.exists():
            self.append('\nSetup required: the JARVIS Python environment is missing.\n')
            return
        try:
            arguments = [str(python), '-u', str(ROOT / 'jarvis.py'), '--mode', 'online', '--manual-finish']
            arguments.extend(['--language', {'Automatic': 'auto', 'Hindi': 'hi', 'English': 'en'}[self.language.get()]])
            if voice_check:
                arguments.append('--voice-check')
                self.append('\nVoice test: after the beep speak one sentence. Audio goes to your online provider; Groq compares two models (two requests). No action is run.\n')
            self.process = subprocess.Popen(
                arguments,
                cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace',
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except OSError as exc:
            self.append(f'\nCould not start JARVIS: {exc}\n')
            return
        self.status.set('STARTING  •  Loading assistant')
        self.start_button.configure(state='disabled')
        self.check_button.configure(state='disabled')
        self.language_selector.configure(state='disabled')
        self.stop_button.configure(state='normal')
        threading.Thread(target=self.read_output, args=(self.process,), daemon=True).start()

    def send_message(self, event=None):
        if str(self.send_button['state']) == 'disabled':
            return
        text = self.message.get().strip()
        if not text:
            return
        if self.process and self.process.poll() is None:
            self.stop()
            self.append('\nVoice listening stopped while replying to your typed message. Start it again when ready.\n')
        self.message.delete(0, 'end')
        self.append(f'\nYOU: {text}\n')
        self.send_button.configure(state='disabled')
        threading.Thread(target=self.respond, args=(text,), daemon=True).start()

    def respond(self, text):
        try:
            reply = self.session.respond(text)
        except Exception:
            reply = 'Something went wrong while processing that request. Please try again.'
        self.events.put(('reply', reply))

    def finish_speaking(self):
        if self.process and self.process.poll() is None and self.process.stdin:
            try:
                self.process.stdin.write('finish\n')
                self.process.stdin.flush()
                self.done_button.configure(state='disabled')
            except (OSError, ValueError):
                self.append('\nCould not finish the recording. Try the voice test again.\n')

    def read_output(self, process):
        for line in process.stdout:
            self.events.put(('line', line))
        self.events.put(('exit', process.wait()))

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'exit':
                    self.status.set('OFFLINE  •  Microphone off')
                    self.start_button.configure(state='normal')
                    self.check_button.configure(state='normal')
                    self.language_selector.configure(state='readonly')
                    self.stop_button.configure(state='disabled')
                    self.done_button.configure(state='disabled')
                    self.append(f'\nJARVIS stopped (code {value}).\n')
                elif kind == 'reply':
                    self.append(f'\nJARVIS: {value}\n')
                    self.speech.say(value)
                    self.send_button.configure(state='normal')
                elif kind == 'speech_error':
                    self.append(f'\nVOICE ERROR: {value}\n')
                else:
                    self.append(value)
                    if 'JARVIS STATUS: STANDBY' in value:
                        self.status.set('LISTENING  •  Say “Jarvis”')
                    elif 'JARVIS ACTIVATED!' in value:
                        self.status.set('ACTIVE  •  Listening to command')
                    elif 'You can also click Done speaking' in value:
                        self.done_button.configure(state='normal')
                    elif ('End of speech detected' in value or
                          'Finished speaking;' in value or
                          'No speech' in value or
                          'Maximum recording time reached' in value):
                        self.done_button.configure(state='disabled')
                    elif 'Understanding your speech' in value:
                        self.status.set('PROCESSING  •  Please wait')
        except queue.Empty:
            pass
        self.after(100, self.poll)

    def stop(self):
        if self.process and self.process.poll() is None:
            # The Windows virtual-environment launcher can spawn a Python child.
            # Stop only the process tree that this window started.
            subprocess.run(['taskkill', '/PID', str(self.process.pid), '/T', '/F'],
                           capture_output=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self.status.set('STOPPING  •  Please wait')
            self.stop_button.configure(state='disabled')

    def close_app(self):
        self.stop()
        self.speech.close()
        self.destroy()


if __name__ == '__main__':
    JarvisApp().mainloop()

