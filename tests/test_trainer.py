# tests/test_byte_pair_encoder.py
import pytest
from bpe.encoder.byte_pair_encoder import BytePairEncoder
from bpe.model.trainer import Trainer

@pytest.fixture
def bpe():
    return BytePairEncoder()

def test_trainer_run_and_predict(bpe):

    text = "Hello world from Tunisia"
    symbols = bpe.sentence_to_symbols(text)
    new_tokenized_words = bpe.keep_merging(25, symbols)
    vocab, token_to_id, id_to_token = bpe.build_vocab(symbols, new_tokenized_words)
    ids = bpe.from_text_to_ids(text, new_tokenized_words, token_to_id)
    trainer = Trainer([ids])
    trainer.run()
    text = "Hello"
    ids = bpe.from_text_to_ids(text, new_tokenized_words, token_to_id)
    predicted_next_word = trainer.predict_next(ids)
    print(f"predicted_next_word = {predicted_next_word}")
    expected_next_word = "world</w>"
    assert expected_next_word == id_to_token.get(predicted_next_word)