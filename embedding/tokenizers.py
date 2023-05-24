from typing import List

class BaseTokenizer:
    def __init__(self, text: str) -> None:
        self.tokens: List[str] = []

    def get_tokens(self) -> List[str]:
        return self.tokens
    
class Whitespace(BaseTokenizer):
    def __init__(self, text: str) -> None:
        self.tokens = text.split()