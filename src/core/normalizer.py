import re
import unicodedata
from typing import Dict, List, Pattern

class TextNormalizer:
    """High-performance text cleaning and lexical normalization engine."""

    def __init__(self, lowercase: bool = False, strip_accents: bool = True):
        self.lowercase = lowercase
        self.strip_accents = strip_accents
        
        # Pre-compiled high efficiency regex patterns
        self._html_tag_re: Pattern = re.compile(r"<[^>]+>", re.IGNORECASE)
        self._markdown_link_re: Pattern = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
        self._markdown_symbols_re: Pattern = re.compile(r"[*_~`#>]")
        self._url_re: Pattern = re.compile(r"https?://\S+|www\.\S+")
        self._email_re: Pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        self._multiple_whitespace_re: Pattern = re.compile(r"[ \t]+")
        self._multiple_newline_re: Pattern = re.compile(r"\n{3,}")
        self._token_re: Pattern = re.compile(r"\b\w+\b", re.UNICODE)

    def clean(self, text: str) -> str:
        """Executes full lexical normalization pipeline over input text."""
        if not text or not isinstance(text, str):
            return ""

        # 1. Unicode NFKC normalization
        normalized = unicodedata.normalize("NFKC", text)

        # 2. Filter non-printable control characters safely
        normalized = "".join(ch for ch in normalized if ch.isprintable() or ch in "\n\t\r")

        # 3. Strip HTML markup
        normalized = self._html_tag_re.sub(" ", normalized)

        # 4. Clean Markdown links and formatting
        normalized = self._markdown_link_re.sub(r"\1", normalized)
        normalized = self._markdown_symbols_re.sub(" ", normalized)

        # 5. Clean whitespace & newlines
        normalized = self._multiple_whitespace_re.sub(" ", normalized)
        normalized = self._multiple_newline_re.sub("\n\n", normalized)
        normalized = normalized.strip()

        # 6. Optional Lowercasing
        if self.lowercase:
            normalized = normalized.lower()

        return normalized

    def tokenize(self, text: str) -> List[str]:
        """Extracts word tokens using unicode-aware regex."""
        if not text:
            return []
        return self._token_re.findall(text)

    def compute_stats(self, text: str) -> Dict[str, int]:
        """Computes lexical metrics for quality gating and validation."""
        tokens = self.tokenize(text)
        return {
            "char_count": len(text),
            "word_count": len(tokens),
            "digit_count": sum(1 for c in text if c.isdigit()),
            "whitespace_count": sum(1 for c in text if c.isspace())
        }
