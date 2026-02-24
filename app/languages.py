import re
from typing import Dict, Tuple

LANGUAGES: Dict[str, str] = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "cs": "Czech",
    "ru": "Russian",
    "uk": "Ukrainian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "bn": "Bengali",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "ms": "Malay",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "el": "Greek",
    "he": "Hebrew",
    "ro": "Romanian",
    "hu": "Hungarian",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sr": "Serbian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "et": "Estonian",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
    "fa": "Persian",
    "fil": "Filipino",
    "ca": "Catalan",
    "gl": "Galician",
    "eu": "Basque",
    "cy": "Welsh",
    "ga": "Irish",
}

_LANG_CHOICES = ", ".join(f"{code} ({name})" for code, name in LANGUAGES.items())
_CODE_NAME_PATTERN = re.compile(r"^\s*([a-z]{2,3})\s*\(\s*([^)]+)\s*\)\s*$", re.IGNORECASE)


def normalize_language(value: str) -> Tuple[str, str]:
    raw = value.strip()
    if not raw:
        raise ValueError("language is required.")

    match = _CODE_NAME_PATTERN.match(raw)
    if match:
        code = match.group(1).lower()
        if code in LANGUAGES:
            return code, LANGUAGES[code]

    code = raw.lower()
    if code in LANGUAGES:
        return code, LANGUAGES[code]

    lowered = raw.lower()
    for lang_code, lang_name in LANGUAGES.items():
        if lowered == lang_name.lower():
            return lang_code, lang_name

    raise ValueError(
        "Unsupported language value. Use language code (for example 'fr'), "
        f"language name (for example 'French'), or formatted value like 'fr (French)'. "
        f"Supported choices: {_LANG_CHOICES}"
    )


def format_language(code: str) -> str:
    lang_name = LANGUAGES[code]
    return f"{code} ({lang_name})"
