# Project JARVIS

**A local AI voice assistant for Windows, built around Hindi, English, and Hinglish commands.**

JARVIS connects speech recognition, local language-model conversations, spoken replies, and a small set of Windows actions. The project explores a practical question: how can a voice assistant remain useful on a CPU laptop while keeping its computer controls predictable?

**Status:** working prototype, version 4.2 · Windows · local or optional cloud inference · active development

## What it does

- Activates with “Hey Jarvis,” with a short local speech-recognition fallback for “Jarvis.”
- Captures commands with pre-roll, voice activity detection, room-noise calibration, and quiet-input boosting.
- Uses faster-whisper for English/Hindi transcription, including language retries and rejection of obvious nonsense or repeated text.
- Runs conversations through Ollama on the same computer and replies using the installed Windows voice.
- Can instead use Gemini online transcription and chat after a local wake word, which is better suited to Hindi/English code-switching. Online mode needs the user's own Gemini API key.
- Searches current Google results through SerpApi after the user confirms, summarizes returned excerpts locally with Ollama, and prints numbered source links.
- Opens Chrome, Notepad, Calculator, File Explorer, Settings, the project folder, Google, or YouTube through fixed actions.
- Requests spoken confirmation before sending a search query to SerpApi, Google, or YouTube.
- Offers microphone selection, configuration display, an opt-in microphone test, and wake diagnostics.

## How it works

![JARVIS voice pipeline](docs/architecture.svg)

**Stack:** Python · Gemini API (optional) · Groq API (optional) · SerpApi Google Search API · faster-whisper · openWakeWord / ONNX Runtime · Ollama · WebRTC VAD · sounddevice · NumPy · SciPy · pyttsx3 · edge-tts

The default speech model is Whisper `small`, using CPU `int8` inference. The main speech loader can fall back to `base`; the short wake fallback also uses `base`. Conversations use `qwen3.5:2b` through `http://127.0.0.1:11434/api/chat`.

## Run locally

The development environment uses **Windows 11 and Python 3.11**. A microphone and an installed Windows speech voice are required. Other operating systems are not supported by this implementation.

1. Install [Python](https://www.python.org/downloads/windows/) and [Ollama for Windows](https://ollama.com/download/windows).
2. Open PowerShell in this project folder and prepare the environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe -c "from openwakeword.utils import download_models; download_models(model_names=['hey_jarvis'])"
   ollama pull qwen3.5:2b
   ```

3. Keep Ollama running. Review the selected microphone without recording:

   ```powershell
   .\.venv\Scripts\python.exe jarvis.py --show-config
   ```

4. Double-click `Start_JARVIS.cmd`, or run:

   ```powershell
   .\.venv\Scripts\python.exe jarvis.py
   ```

The first speech-model load may download Whisper models. Initial dependency and model setup requires internet access. Requirements list the imported libraries and selected ONNX runtime without version pins; a clean installation has not yet been validated as a reproducible release.

## Run online with Gemini

JARVIS also supports Groq for online speech recognition and chat. When `GROQ_API_KEY` is present in the environment, local `.env`, or private `.groq.env`, Groq is selected for both; otherwise the existing Gemini configuration remains in use. Groq receives activated command audio and conversation text. Groq is not silently switched on until a key is configured. Its free plan has usage limits, and availability and accuracy should be tested with the user's own voice. Hindi and English spoken replies use the online Edge voice service. Do not share or distribute `.env`, `.groq.env`, or API keys.

Use this mode when the local recognizer is not accurate enough for Hindi, English, or Hinglish. It keeps the wake listener and all computer-control rules on the laptop. Only the completed command after JARVIS activates, and ordinary chat text after that, are sent to Gemini.

1. Create a personal Gemini API key in [Google AI Studio](https://aistudio.google.com/apikey). At the time this version was built, Gemini's documentation listed a free tier for Gemini 3.5 Transcribe, subject to account and rate limits. Review the current [pricing](https://ai.google.dev/gemini-api/docs/pricing) before use.
2. Double-click `Set_Up_Online_JARVIS.cmd`. It creates the ignored local `.env` settings file if needed, then opens it in Notepad.
3. Replace only `your_gemini_api_key_here` with the key you created. Do not share the file or its key. Save and close Notepad.
4. Double-click `Start_JARVIS_Online.cmd`.

The first online start should say `Cloud mode: ON — Gemini transcribes commands and handles normal chat.` If it says that no Gemini key was found, check that the file is named exactly `.env`, is in the JARVIS folder, and contains `GEMINI_API_KEY=...` on one line.

Gemini online mode sends an in-memory WAV of the activated command directly in the transcription request. JARVIS does not create a local recording or a separate uploaded file. On Google's free tier, submitted content may be used to improve Google products, so do not use cloud mode for passwords, banking, private documents, or sensitive dictation. See Google's [API-key guide](https://ai.google.dev/gemini-api/docs/api-key) and [data-use terms](https://ai.google.dev/gemini-api/terms).

`Start_JARVIS.cmd` stays in automatic mode: it uses Gemini if a valid key is found and otherwise uses local models. Add `--mode local` after the launcher command to force local operation at any time.

### Windows desktop preview

Double-click `Open_JARVIS_App.cmd` for the desktop interface. Close any existing JARVIS console first so only one copy listens to the microphone. Select **Start listening** to start online voice mode; **Stop** stops the voice process. Closing the window also stops the process started by that window. Activity text remains in the window only for that session.

You can also type into the message box and press **Send**. Typed messages use Gemini online and retain a short conversation history for the current window. Typed commands use the existing local allowlist, with typed yes/no confirmation for searches. Typed and voice sessions currently have separate conversation histories and confirmations. See `REBUILD_NOTES.md` for the reassessment and remaining work.

Select **Test voice** to check recognition without the wake word. Wait for the beep and say a sentence; the audio goes to the configured online provider, the transcript appears in the Activity panel, and no action is executed. This helps distinguish wake-word problems from microphone or transcription problems. Stop any existing voice listener first. JARVIS currently waits about 1.4 seconds of detected quiet before transcription; pauses longer than this can split a sentence. Online processing takes additional time and varies with service and network conditions.

The **Speech language** selector applies to voice tests and voice commands after the listener starts. Choose Automatic for mixed speech; if automatic detection uses the wrong language or writing system, try Hindi or English for a test. The voice test shows the actual recognized text as `VOICE TEST HEARD` even when the safety filter rejects it; this is diagnostic only and never executes a command. A forced language can misrecognize a different language, so use Automatic when switching languages frequently.

With Groq, the normal speech model is Whisper Large V3. The voice test sends the same captured sentence twice to Groq to compare it with the faster Turbo model; this consumes two requests of the Groq plan. The two versions are displayed for comparison and never run as computer commands. Say the wake word, wait for the beep, and only then speak the command; wake-word audio and command capture are separate.

If background noise prevents automatic stopping, click **Done speaking** after your sentence. This immediately ends the current capture and sends it for transcription; **Stop** instead cancels the voice process. The room-noise estimate is collected before the activation beep, so wait for that beep before speaking.

Typed replies are spoken using the selected Windows voice. Sending a typed message stops voice listening so JARVIS does not mistake its own reply for a wake phrase; select **Start listening** to resume. If speech fails, the app shows a voice error; check the selected Windows speaker and volume. Computer access remains limited to the listed actions and does not include general screen navigation or unrestricted file access.

JARVIS selects Hindi for Devanagari and common Hinglish phrases, and English for English requests. You can explicitly say “in Hindi” or “in English”; short or ambiguous Hinglish can still be misclassified. Hindi replies use Gemini's online speech service; English replies use Microsoft Edge's online neural voice through the third-party `edge-tts` library. Reply text is sent to the respective service, and audio is played from memory. If the online English voice fails, JARVIS reports this and falls back to the Windows voice. If the online Hindi voice fails, the reply remains visible. Availability, free-tier limits, and paid-tier charges depend on the service and account.

This is a local preview, not a portable installer. It depends on this folder, its `.venv`, and a locally configured API key. Android requires a separate application and platform-specific microphone and action integrations. Do not distribute the `.env`, `.venv`, or your personal API key with an app package.

### Run the live-web demo without a microphone

1. Create a SerpApi key from your account and copy the local example file:

   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```

2. Replace `your_serpapi_key_here` with your key. The real `.env` file is ignored by Git. Keep the key private and do not paste it into source code, screenshots, or a public repository.
3. Keep Ollama running with the `qwen3.5:2b` model, then start the no-microphone demo:

   ```powershell
   .\.venv\Scripts\python.exe jarvis.py --search-demo
   ```

Type a question in English or Hindi. JARVIS asks for confirmation before it sends the query to SerpApi and its search provider. It shows the result links, asks the local Ollama model to summarize only the returned excerpts, and speaks the answer. The live API path needs a valid key and a search credit; no search key is included in this project.

Hindi text is always printed. Spoken Hindi uses a Hindi Windows voice if one is installed; otherwise JARVIS falls back to the system's default voice.

The request follows SerpApi's documented [Google Search API](https://serpapi.com/search-api) JSON endpoint. See its [Python integration guide](https://serpapi.com/integrations/python) for the official SDK alternative.

### What the demo makes visible

- JARVIS repeats the query and waits for a clear yes before any search request is sent.
- The Google search is localized to India, with Hindi or English response language.
- The screen shows the result excerpts and numbered source links, so viewers can compare the answer with its evidence.
- JARVIS never opens result pages automatically. Search text is sent to SerpApi; only result excerpts go to the local Ollama model for summarizing.
- API keys stay in an ignored local `.env` file. Errors are written without exposing the key-bearing request URL.

The same search works by voice in normal JARVIS mode. Say **“Search the web for [question]”** or **“Web par search karo [question].”** JARVIS repeats what it plans to search and waits for a separate spoken “yes” before making the request. A request in Hindi (including a Hindi response cue) gets Hindi output; otherwise it defaults to English.

Say **“Jarvis”** or **“Hey Jarvis,”** wait for the activation beep, then speak your command. Stop the app with `Ctrl+C` or the voice command “stop Jarvis.” Keep the terminal open while it runs.

### Example commands

| Say after activation | Behavior |
| --- | --- |
| “Open Calculator” | Launches Calculator. |
| “Notepad kholo” | Launches Notepad. |
| “What time is it?” | Speaks the current local time. |
| “Search the web for the SerpApi Python SDK in Hindi” | Asks before calling SerpApi, then summarizes results locally and shows the source links. |
| “Search for Python documentation” | Asks before opening a Google search in the browser. |
| “Open Chrome, then open YouTube, then find me a video of BMW F450GS” | Opens Chrome/YouTube and asks before submitting the video search. |
| “Jarvis, Chrome kholo aur usme YouTube pe koi bhi video chalao” | Opens a fixed sample video about Earth in Chrome. The browser or YouTube may require a manual Play click. |

For a pending search, wake JARVIS again, wait for the beep, then say **“yes”** or **“no.”** Confirmation expires after 30 seconds; a new command clears the pending request. The YouTube action opens search results; it does not choose or play a video.

### Diagnostics and tuning

```powershell
# List inputs without recording
.\.venv\Scripts\python.exe jarvis.py --list-devices

# Inspect a selected input without recording
.\.venv\Scripts\python.exe jarvis.py --mic "Microphone Array" --show-config

# Explicit six-second in-memory level test; waits for Enter first
.\.venv\Scripts\python.exe jarvis.py --mic-test

# Print wake scores and local speech candidates while listening
.\.venv\Scripts\python.exe jarvis.py --wake-debug

# Try the smaller model or force one transcription language
.\.venv\Scripts\python.exe jarvis.py --model base --language en
```

`--mic` accepts a device number or part of its name. The `JARVIS_MIC_DEVICE`, `JARVIS_WHISPER_MODEL`, and `JARVIS_LANGUAGE` environment variables provide defaults. Microphone selection prefers WASAPI, captures at 48 kHz when supported, and resamples locally to 16 kHz. Command timing and thresholds are constants near the top of `jarvis.py`.

## Data and action boundaries

In local mode, speech processing and Ollama conversation requests run locally after model setup. Audio is held in memory; this script does not create recordings. It listens while running, and the local wake fallback can briefly transcribe detected background speech. Normal operation prints command transcripts and replies; explicit wake debug mode can also print background speech candidates.

In Gemini online mode, the wake listener stays local. After activation, the completed command audio and ordinary chat text are sent to Gemini. Audio is included directly with the transcription request rather than stored as a separate API file. The cloud model cannot operate Windows: fixed browser/app actions and their confirmations remain in the local Python safety layer.

The live-web feature sends the confirmed search query to SerpApi and its search provider. Result excerpts are then passed to the local Ollama model for a short answer; the source links remain visible in the terminal and are not opened automatically. Browser searches send the confirmed query directly to Google or YouTube. Opening their homepages also contacts those websites. The language model does not execute computer actions: the Python router handles the fixed allowlist. Deletion, arbitrary terminal commands, software installation, shutdown/restart, and sending messages are outside the supported actions.

## Verification and current limits

Previous development checks covered Python compilation, model loading, microphone selection, resampling, transcript guards, and simulated command routing/confirmation. Live development sessions observed successful wake activation. The new SerpApi path has not been run with a real API key, and the latest combined spoken Chrome/YouTube sequence has not been verified end to end.

Recognition and responsiveness still vary with microphone placement, background noise, accent, language mixing, and CPU load. There is no formal accuracy or latency benchmark, general desktop navigation, screenshot understanding, or packaged installer. The project is a prototype for local experimentation.

## Project files

```text
jarvis.py               Voice capture, transcription, command routing and assistant loop
jarvis_app.py           Windows desktop preview for typed chat and voice controls
assistant_session.py    Typed conversation and action-confirmation state
online_gemini.py        Optional Gemini speech and chat client
online_provider.py      Optional Groq speech and chat provider
reply_language.py       Hindi, English and Hinglish reply-language selection
speech_output.py        Hindi/English speech output with local fallback
serpapi_live_search.py  Confirmed SerpApi search, result links and grounded summaries
Open_JARVIS_App.cmd     Desktop preview launcher
Start_JARVIS.cmd        Automatic/local-mode voice launcher
Start_JARVIS_Online.cmd Online voice launcher
Set_Up_Online_JARVIS.cmd Creates/opens local API-key settings in Notepad
requirements.txt        Python dependencies
.env.example            Placeholder template for optional private API keys
.gitignore              Excludes local environments, keys, models, recordings and logs
```

No model weights, audio recordings, local environments, or development backups are included. An optional `jarvis_wake.onnx` can be loaded from this folder if a user supplies a suitable model; no custom model is distributed here.

## Upstream projects and model terms

JARVIS builds on [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [openWakeWord](https://github.com/dscripka/openWakeWord), [Ollama](https://github.com/ollama/ollama), and [Qwen3.5](https://ollama.com/library/qwen3.5:2b), alongside the libraries in `requirements.txt`.

Model weights retain their upstream terms. In particular, openWakeWord documents its bundled pretrained wake models under **CC BY-NC-SA 4.0**, separately from its Apache 2.0 library code; see the [upstream license section](https://github.com/dscripka/openWakeWord#license). This repository does not bundle those weights.

