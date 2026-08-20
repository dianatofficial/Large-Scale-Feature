from src.core.normalizer import TextNormalizer

def test_html_stripping(sample_raw_text):
    normalizer = TextNormalizer()
    cleaned = normalizer.clean(sample_raw_text)
    
    assert "<html>" not in cleaned
    assert "<b>" not in cleaned
    assert "<a href=" not in cleaned
    assert "Quarterly Technical Audit" in cleaned
    assert "Read RFC documentation here" in cleaned

def test_whitespace_and_control_characters():
    normalizer = TextNormalizer()
    dirty_text = "  Multiple    spaces   and \t tabs \n\n\n\n excessive newlines  "
    cleaned = normalizer.clean(dirty_text)
    
    assert "Multiple spaces and tabs" in cleaned
    assert "excessive newlines" in cleaned
    assert "  " not in cleaned

def test_tokenization_and_stats():
    normalizer = TextNormalizer()
    text = "Distributed batch feature processing with 1000 nodes."
    tokens = normalizer.tokenize(text)
    stats = normalizer.compute_stats(text)
    
    assert "Distributed" in tokens
    assert "processing" in tokens
    assert stats["word_count"] == 7
    assert stats["digit_count"] == 4
