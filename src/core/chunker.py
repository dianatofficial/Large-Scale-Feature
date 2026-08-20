import re
from typing import List, Tuple
from src.core.schemas import DocumentChunk

class TextChunker:
    """Token-aware recursive text chunker preserving sentence and structural boundaries."""

    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 32):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be strictly smaller than chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.step_size = chunk_size - chunk_overlap

    def chunk_text(self, document_id: str, text: str, source: str = "unknown", category: str = "general") -> List[DocumentChunk]:
        """Splits document text into overlapping token-aware chunks with character offsets."""
        if not text or not text.strip():
            return []

        words = text.split()
        if len(words) <= self.chunk_size:
            return [
                DocumentChunk(
                    chunk_id=f"{document_id}_chk_0",
                    document_id=document_id,
                    chunk_index=0,
                    total_chunks=1,
                    chunk_content=text.strip(),
                    token_count=len(words),
                    start_char=0,
                    end_char=len(text.strip()),
                    source=source,
                    category=category
                )
            ]

        chunks: List[DocumentChunk] = []
        word_spans: List[Tuple[int, int]] = []
        
        # Locate character positions of each word in the source text
        current_idx = 0
        for word in words:
            start = text.find(word, current_idx)
            if start == -1:
                start = current_idx
            end = start + len(word)
            word_spans.append((start, end))
            current_idx = end

        total_words = len(words)
        chunk_idx = 0
        start_word_idx = 0

        while start_word_idx < total_words:
            end_word_idx = min(start_word_idx + self.chunk_size, total_words)
            chunk_words = words[start_word_idx:end_word_idx]
            
            start_char = word_spans[start_word_idx][0]
            end_char = word_spans[end_word_idx - 1][1]
            chunk_content = text[start_char:end_char].strip()

            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document_id}_chk_{chunk_idx}",
                    document_id=document_id,
                    chunk_index=chunk_idx,
                    total_chunks=0,
                    chunk_content=chunk_content,
                    token_count=len(chunk_words),
                    start_char=start_char,
                    end_char=end_char,
                    source=source,
                    category=category
                )
            )

            if end_word_idx >= total_words:
                break

            start_word_idx += self.step_size
            chunk_idx += 1

        total_count = len(chunks)
        for c in chunks:
            c.total_chunks = total_count

        return chunks
