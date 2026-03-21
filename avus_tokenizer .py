
import tiktoken
from typing import List

class AvusTokenizer:
    def __init__(self):
        self._enc = None
        try:
            self._enc = tiktoken.get_encoding("gpt2")
        except ImportError:
            print("tiktoken not found, falling back to basic ASCII encoding.")

    def encode(self, text: str) -> List[int]:
        if self._enc is not None:
            return self._enc.encode(text)
        return [min(b, 255) for b in text.encode("utf-8")]

    def decode(self, tokens: List[int]) -> str:
        if self._enc is not None:
            valid = [t for t in tokens if 0 <= t < 50257]
            try:
                return self._enc.decode(valid)
            except Exception:
                return ""
        return bytes([t for t in tokens if t < 256]).decode("utf-8", errors="replace")
