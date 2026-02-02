from pypdf import PdfReader
import docx

class DocumentLoader:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load_pdf(self) -> str:
        try:
            reader = PdfReader(self.file_path)
            text = []
            for page in reader.pages:
                text.append(page.extract_text())
            return '\n'.join(text)
        except Exception as e:
            raise ValueError(f"Error reading PDF: {e}")
    
    def load_txt(self, encoding="utf-8") -> str:
        with open(self.file_path, "r", encoding=encoding) as f:
            return f.read()
    
    def load_docx(self) -> str:
        try:
            doc = docx.Document(self.file_path)
            fullText = []
            for para in doc.paragraphs:
                fullText.append(para.text)
            return '\n'.join(fullText)
        except Exception as e:
            raise ValueError(f"Error reading DOCX: {e}")

    def load_md(self) -> str:
        with open(self.file_path, "r") as f:
            return f.read()
    
    def load(self) -> str:
        extension = self.file_path.split(".")[-1]
        if extension == "pdf":
            return self.load_pdf()
        elif extension == "txt":
            return self.load_txt()
        elif extension == "docx":
            return self.load_docx()
        elif extension == "md":
            return self.load_md()
        else:
            raise ValueError(f"Unsupported file format: {self.file_path}")