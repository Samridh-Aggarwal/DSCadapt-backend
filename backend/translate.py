"""Translate in on the way through, out on the way back.

The model always writes English; a question in another language is translated
before the pipeline sees it, and the answer is translated back. The reference
list is split off and handled separately so academic citations survive.

One thing this cannot fix from here, recorded because it needs a change on the
other side. The frontend stores the translated answer as conversation history
and sends it back on the next turn, so on a French session the model reads its
own French output while being told never to mirror the language of the
conversation. Translating the whole history on every turn would fix it and
would be billed per character, every turn, forever. The cheap fix is for the
answer to carry its English original alongside the translation and for the
frontend to keep that as history — english_answer in the /ask response exists
for exactly that.
"""

import re

# Frontend display name -> DeepL target code. English is None: no translation.
LANGUAGES = {
    "English": None,
    "Čeština": "CS", "Dansk": "DA", "Deutsch": "DE", "Eesti": "ET",
    "Español": "ES", "Français": "FR", "Gaeilge": "GA", "Hrvatski": "HR",
    "Íslenska": "IS", "Italiano": "IT", "Latviešu": "LV", "Lietuvių": "LT",
    "Malti": "MT", "Magyar": "HU", "Nederlands": "NL", "Norsk": "NB",
    "Polski": "PL", "Português": "PT-PT", "Română": "RO", "Slovenčina": "SK",
    "Slovenščina": "SL", "Suomi": "FI", "Svenska": "SV",
    "Български": "BG", "Ελληνικά": "EL",
}

REFERENCES_SPLIT = re.compile(r"(?i)\n\s*references?\s*\n")


def code_for(language):
    return LANGUAGES.get(language)


class Translator:
    """Wraps the DeepL client. Absent or broken, everything passes through."""

    def __init__(self, client=None):
        self.client = client
        self.characters = 0

    @classmethod
    def from_key(cls, api_key):
        if not api_key:
            print("[DEEPL] no key set, translation disabled")
            return cls(None)
        try:
            import deepl
            client = deepl.Translator(api_key)
            usage = client.get_usage()
            print(f"[DEEPL] ready — {usage.character.count}/{usage.character.limit} characters used")
            return cls(client)
        except Exception as err:
            print(f"[DEEPL] not available: {err}")
            return cls(None)

    @property
    def available(self):
        return self.client is not None

    def _translate(self, text, target):
        if not self.available or not target or not text.strip():
            return text
        try:
            self.characters += len(text)
            return self.client.translate_text(text, target_lang=target).text
        except Exception as err:
            print(f"[DEEPL] translation to {target} failed, returning the original: {err}")
            return text

    def to_english(self, text):
        return self._translate(text, "EN-US")

    def from_english(self, text, target):
        return self._translate(text, target)

    def answer(self, text, target):
        """Translate an answer, keeping the reference list intact.

        Citations translated as prose come back mangled — author names
        localised, journal titles rendered into the target language — so the
        list is split off at the heading and only the heading itself is
        translated.
        """
        if not self.available or not target or not text.strip():
            return text
        parts = REFERENCES_SPLIT.split(text, maxsplit=1)
        if len(parts) != 2:
            return self.from_english(text, target)
        body, references = parts
        heading = self.from_english("References", target)
        return f"{self.from_english(body, target)}\n\n{heading}\n{references}"
