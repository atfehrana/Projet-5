from app.utils import load_spacy_model, process_text

def test_process_text():
    nlp = load_spacy_model("en_core_web_sm")
    text = "This is a test sentence."
    processed_text = process_text(nlp, text)
    assert isinstance(processed_text, list)
    assert len(processed_text) > 0
