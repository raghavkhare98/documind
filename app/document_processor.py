import re
import string
import unicodedata

class DocumentProcessor:
    def __init__(self, text: str) -> None:
        self.text = text
    
    def clean_whitespace(self) -> str:
        """
        Normalize whitespaces in a page, and retain the document structure
        """
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        lines = [line.strip() for line in text.split("\n")]

        text = "\n".join(lines)

        text = re.sub(r"[ \t]+", " ", text)

        text = re.sub(r"\n{3,}", "\n\n", text)
        
        return text
    
    def remove_special_characters(self, keep_punctuation: bool = True) -> str:
        
        if not self.text:
            return ""

        text = unicodedata.normalize("NFKC", self.text)
        
        replacements = {
            '\u2018': "'",  # Left single quote
            '\u2019': "'",  # Right single quote
            '\u201c': '"',  # Left double quote
            '\u201d': '"',  # Right double quote
            '\u2013': '-',  # En dash
            '\u2014': '-',  # Em dash
            '\u2026': '...',  # Ellipsis
        }

        for old, new in replacements.items():
            text = text.replace(old, new)
        
        allowed_chars = set(string.ascii_letters + string.digits)
        allowed_chars.update('\n\t ')
        if keep_punctuation:
            allowed_chars.update(string.punctuation)
        
        cleaned_text = []
        for char in text:
            if char in allowed_chars:
                cleaned_text.append(char)
            elif unicodedata.category(char) not in ('Cc', 'Cf', 'Cs', 'Co', 'Cn'):
                cleaned_text.append(char)
        
        return "".join(cleaned_text)