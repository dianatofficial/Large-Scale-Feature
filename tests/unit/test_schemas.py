import pytest
from src.core.schemas import RawDocument, CleanedDocument, DocumentChunk, VectorPayload, PipelineAuditReport

def test_raw_document_schema():
    doc = RawDocument(
        document_id="doc_001",
        content="Testing raw document ingestion payload",
        source="unit_test",
        category="engineering",
        metadata={"priority": "high"}
    )
    assert doc.document_id == "doc_001"
    assert "Testing" in doc.content
    assert doc.metadata["priority"] == "high"

def test_cleaned_document_schema():
    doc = CleanedDocument(
        document_id="doc_002",
        cleaned_content="Normalized clean content without html tags",
        content_hash="a1b2c3d4e5",
        char_count=42,
        word_count=6,
        lang="en"
    )
    assert doc.word_count == 6
    assert doc.content_hash == "a1b2c3d4e5"

def test_document_chunk_schema():
    chunk = DocumentChunk(
        chunk_id="doc_001_chk_0",
        document_id="doc_001",
        chunk_index=0,
        total_chunks=1,
        chunk_content="Sample chunk content segment",
        token_count=4,
        start_char=0,
        end_char=28
    )
    assert chunk.chunk_id == "doc_001_chk_0"
    assert chunk.token_count == 4

def test_vector_payload_schema():
    payload = VectorPayload(
        chunk_id="chunk_001",
        document_id="doc_001",
        chunk_index=0,
        chunk_content="Test vector",
        vector_fp32=[0.1, 0.2, 0.3],
        scale_factor=0.01968,
        l2_norm=1.0,
        partition_id=1
    )
    assert len(payload.vector_fp32) == 3
    assert payload.partition_id == 1
