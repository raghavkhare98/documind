from pymilvus import connections, Collection, utility
import os

def count_words(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def count_characters(text: str, exclude_whitespace: bool = True) -> int:
    """
    Count the number of characters in a text string
    
    Args:
        text: Input text string
        exclude_whitespace: If True, excludes spaces, newlines, and tabs from count
        
    Returns:
        Number of characters in the text
    """
    if not text:
        return 0
    
    if exclude_whitespace:
        return len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
    return len(text)