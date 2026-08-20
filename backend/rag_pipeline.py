from typing import Callable, List


def create_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[str]:
    """
    Split extracted document text into overlapping chunks.

    Args:
        text: Extracted PDF text.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters shared between consecutive chunks.

    Returns:
        A list of text chunks.
    """

    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    text = text.strip()

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


def retrieve_relevant_chunks(
    query: str,
    chunks: List[str],
    retriever: Callable[[str, List[str]], List[str]]
) -> List[str]:
    """
    Retrieve chunks relevant to a user query.

    The actual retrieval implementation will later be
    connected to the embedding + Qdrant component.

    Args:
        query: User's question.
        chunks: Available document chunks.
        retriever: Retrieval function supplied by the retrieval layer.

    Returns:
        Relevant document chunks.
    """

    if not query or not query.strip():
        return []

    if not chunks:
        return []

    return retriever(query, chunks)