import pymupdf4llm
import pathlib

def convert_pdfs_to_markdown(pdf_file_path: str) -> None:
    try:
        md_text = pymupdf4llm.to_markdown(pdf_file_path)
        pathlib.Path("output.md").write_bytes(md_text.encode())
    except Exception as e:
        print(f"An error ocurred during pdf to markdown conversion: {e}")

convert_pdfs_to_markdown("/Users/raghavkhare/repos/documind/data/raw/research_papers/advancing_rag_for_structured_data.pdf")
