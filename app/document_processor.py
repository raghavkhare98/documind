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
        if not self.text:
            return ""

        text = self.text.replace("\r\n", "\n").replace("\r", "\n")

        lines = [line.strip() for line in text.split("\n")]

        text = "\n".join(lines)

        text = re.sub(r"[ \t]+", " ", text)

        text = re.sub(r"\n{3,}", "\n\n", text)
        
        self.text = text

        return self
    
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
        
        self.text = "".join(cleaned_text)
        return self

    def extract_metadata(self, doc_id: str, doc_name: str = None, doc_path: str = None) -> dict:
        
        if not self.text:
            return{
                "doc_id": doc_id,
                "doc_name": doc_name,
                "doc_path": doc_path,
                "word_count": 0,
                "char_count": 0,
                "sentence_count": 0
            }
        words = self.text.split()
        word_count = len(words)
        
        char_count = len(self.text.replace(" ", "").replace("\n", "").replace("\t", ""))

        sentence_endings = re.findall(r'[.!?]+', self.text)
        sentence_count = len(sentence_endings) if sentence_endings else 1

        return {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "doc_path": doc_path,
            "word_count": word_count,
            "char_count": char_count,
            "sentence_count": sentence_count
        }
    
    def preserve_structure(self) -> str:
        if not self.text:
            return ""
        
        lines = self.text.split("\n")
        processed_lines = []
        for line in lines:
            stripped = line.strip()

            if not stripped:
                processed_lines.append('')
                continue
            
            if stripped.startswith("#"):
                processed_lines.append(f'[HEADER] {stripped}')
            elif stripped.isupper() and len(stripped.split()) >= 3:
                processed_lines.append(f'[HEADER] {stripped}')
            elif '|' in stripped or '\t' in line:
                if stripped.count('|') >= 2 or line.count('\t') >= 2:
                    processed_lines.append(f'[TABLE] {stripped}')
                else:
                    processed_lines.append(stripped)
            elif re.match(r'^[\-\*\+•]\s+', stripped):
                processed_lines.append(f'[BULLET] {stripped}')
            elif re.match(r'^\d+[\.\)]\s+', stripped):
                processed_lines.append(f'[NUMBERED] {stripped}')
            else:
                processed_lines.append(stripped)
        
        self.text = '\n'.join(processed_lines)
        return self