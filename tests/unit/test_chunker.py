import pytest
from src.core.chunker import TextChunker

def test_short_text_single_chunk():
    chunker = TextChunker(chunk_size=128, chunk_overlap=16)
    text = "Short single paragraph that fits within a single chunk."
    chunks = chunker.chunk_text(document_id="doc_100", text=text)
    
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].total_chunks == 1
    assert chunks[0].chunk_content == text

def test_multichunk_sliding_window():
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    # 25 words
    text = " ".join([f"word_{i}" for i in range(25)])
    chunks = chunker.chunk_text(document_id="doc_200", text=text)
    
    assert len(chunks) >= 3
    assert chunks[0].chunk_id == "doc_200_chk_0"
    assert chunks[1].chunk_id == "doc_200_chk_1"
    assert chunks[0].total_chunks == len(chunks)

def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=64, chunk_overlap=64)
