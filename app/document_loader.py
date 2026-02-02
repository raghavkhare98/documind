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
            
            for table in doc.tables:
                for row in table.rows:
                    row_text = ' | '.join(cell.text for cell in row.cells)
                    fullText.append(row_text)
            
            return '\n'.join(fullText)
        except Exception as e:
            raise ValueError(f"Error reading DOCX: {e}")

    def load_md(self) -> str:
        with open(self.file_path, "r") as f:
            return f.read()
    
    def load(self) -> str:
        
        if not self.file_path:
            raise ValueError("File path is required")

        extension = self.file_path.split(".")[-1].lower()

        loaders = {
            "pdf": self.load_pdf,
            "txt": self.load_txt,
            "docx": self.load_docx,
            "md": self.load_md
        }

        loader = loaders.get(extension)
        if not loader:
            raise ValueError(f"Unsupported file format: {self.file_path}")
        
        return loader()