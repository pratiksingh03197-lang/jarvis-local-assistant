"""Simple explicit English/Hindi selection and fixed action translations."""
import re


def response_language(text: str) -> str:
    text = text.casefold()
    if re.search(r'\b(?:in english|english mein|english me|angrezi mein)\b', text):
        return 'en'
    if re.search(r'[\u0900-\u097f]', text):
        return 'hi'
    if re.search(r'\b(?:hindi mein|hindi me|in hindi|karo|kijiye|kholo|kholna|kya|kaise|hai|hain|khojo|dhundo|dhoondo|batao|bataiye|mujhe|mera|meri|kaun|kab|kyun|kyon|nahin|nahi|haan)\b', text):
        return 'hi'
    return 'en'


def language_instruction(language: str) -> str:
    if language == 'hi':
        return ('Respond in natural Hindi using Devanagari script. Preserve names and '
                'technical terms where useful. Keep spoken answers concise.')
    return 'Respond in clear English. Keep spoken answers concise.'


def localize_control_reply(reply: str, language: str) -> str:
    """Translate deterministic control messages without changing action targets."""
    if language != 'hi':
        return reply
    replacements = {
        'Opening Calculator.': 'कैलकुलेटर खोल रहा हूँ।',
        'Opening Notepad.': 'नोटपैड खोल रहा हूँ।',
        'Opening File Explorer.': 'फाइल एक्सप्लोरर खोल रहा हूँ।',
        'Opening Settings.': 'सेटिंग्स खोल रहा हूँ।',
        'Opening the JARVIS folder.': 'जार्विस का फोल्डर खोल रहा हूँ।',
        'Opening Chrome.': 'क्रोम खोल रहा हूँ।',
        'Opening Google in Chrome.': 'क्रोम में गूगल खोल रहा हूँ।',
        'Opening YouTube in Chrome.': 'क्रोम में यूट्यूब खोल रहा हूँ।',
        'Opening a sample YouTube video about Earth in Chrome. If it does not start automatically, press Play on the video.':
            'क्रोम में पृथ्वी के बारे में एक यूट्यूब वीडियो खोल रहा हूँ। अगर वीडियो अपने आप शुरू न हो, तो प्ले दबाएँ।',
        'Google Chrome could not be found.': 'गूगल क्रोम नहीं मिला।',
        'I could not open the browser.': 'मैं ब्राउज़र नहीं खोल पाया।',
        'Sorry, I could not complete that action.': 'माफ़ कीजिए, मैं यह काम पूरा नहीं कर पाया।',
        'Canceled.': 'रद्द कर दिया।',
        'Wake me again, wait for the beep, then say yes to continue or no to cancel.':
            'जारी रखने के लिए मुझे फिर जगाएँ, बीप के बाद हाँ कहें। रद्द करने के लिए नहीं कहें।',
        'Wake me again, then say yes to continue or no to cancel.':
            'मुझे फिर जगाएँ, जारी रखने के लिए हाँ या रद्द करने के लिए नहीं कहें।',
        'Type yes to continue or no to cancel.':
            'जारी रखने के लिए yes और रद्द करने के लिए no लिखें।',
        'This sends those words to YouTube.': 'इससे ये शब्द यूट्यूब को भेजे जाएँगे।',
        'This sends those words to Google.': 'इससे ये शब्द गूगल को भेजे जाएँगे।',
        'Confirmed.': 'पुष्टि हो गई।',
    }
    for english, hindi in replacements.items():
        reply = reply.replace(english, hindi)
    reply = re.sub(r'I can search YouTube for (.+?)\.', r'मैं यूट्यूब पर \1 खोज सकता हूँ।', reply)
    reply = re.sub(r'I can search Google for (.+?)\.', r'मैं गूगल पर \1 खोज सकता हूँ।', reply)
    reply = re.sub(r'Searching YouTube for (.+?)\.', r'यूट्यूब पर \1 खोज रहा हूँ।', reply)
    reply = re.sub(r'Searching Google for (.+?)\.', r'गूगल पर \1 खोज रहा हूँ।', reply)
    return reply

