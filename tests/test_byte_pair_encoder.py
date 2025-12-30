# tests/test_byte_pair_encoder.py
import pytest
from bpe.encoder.byte_pair_encoder import BytePairEncoder

@pytest.fixture
def bpe():
    return BytePairEncoder()

def test_word_to_symbols_should_return_list_of_symbols(bpe):
    text = "Hello world from Tunisia"
    expected_symbols = [['H', 'e', 'l', 'l', 'o', '</w>'], ['w', 'o', 'r', 'l', 'd', '</w>'], ['f', 'r', 'o', 'm', '</w>'], ['T', 'u', 'n', 'i', 's', 'i', 'a', '</w>']]
    assert bpe.sentence_to_symbols(text) == expected_symbols


def test_get_pair_counts_should_return_pair_counts(bpe):
    text = "Hello world from Tunisia"
    expected_pairs_count={('H', 'e'): 1, ('e', 'l'): 1, ('l', 'l'): 1, ('l', 'o'): 1, ('o', '</w>'): 1, ('w', 'o'): 1, ('o', 'r'): 1, 
                          ('r', 'l'): 1, ('l', 'd'): 1, ('d', '</w>'): 1, ('f', 'r'): 1, ('r', 'o'): 1, ('o', 'm'): 1, ('m', '</w>'): 1, 
                          ('T', 'u'): 1, ('u', 'n'): 1, ('n', 'i'): 1, ('i', 's'): 1, ('s', 'i'): 1, ('i', 'a'): 1, ('a', '</w>'): 1}
    symbols = bpe.sentence_to_symbols(text)
    pair_count = bpe.get_pair_counts(symbols)
    assert pair_count == expected_pairs_count


def test_keep_merging(bpe):
    text = "Hello world from Tunisia"
    expected_new_tokenized_words=[('H', 'e'), ('He', 'l'), ('Hel', 'l'), ('Hell', 'o'), ('Hello', '</w>'), ('w', 'o'), ('wo', 'r'), ('wor', 'l'), 
                                  ('worl', 'd'), ('world', '</w>'), ('f', 'r'), ('fr', 'o'), ('fro', 'm'), ('from', '</w>'), ('T', 'u'), ('Tu', 'n'), 
                                  ('Tun', 'i'), ('Tuni', 's'), ('Tunis', 'i'), ('Tunisi', 'a'), ('Tunisia', '</w>')]
    symbols = bpe.sentence_to_symbols(text)
    new_tokenized_words = bpe.keep_merging(25, symbols)
    assert new_tokenized_words == expected_new_tokenized_words

def test_apply_bpe(bpe):
    text = "Hello world from Tunisia"
    expected_token=['Hello', 'from', 'world</w>']
    symbols = bpe.sentence_to_symbols(text)
    new_tokenized_words = bpe.keep_merging(25, symbols)
    print(new_tokenized_words)
    tokens_result = bpe.apply_bpe("Hellofromworld", new_tokenized_words)
    print(tokens_result)
    assert tokens_result == expected_token

def test_build_vocab(bpe):
    text = "Hello world from Tunisia"
    symbols = bpe.sentence_to_symbols(text)
    new_tokenized_words = bpe.keep_merging(25, symbols)
    vocab, token_to_id, id_to_token = bpe.build_vocab(symbols, new_tokenized_words)
    expected_vocab = ['He', 'Hel', 'Hell', 'Hello', 'Hello</w>', 'Hello</w>', 'Tu', 'Tun', 'Tuni', 'Tunis', 'Tunisi', 'Tunisia', 'Tunisia</w>', 
                      'Tunisia</w>', 'fr', 'fro', 'from', 'from</w>', 'from</w>', 'wo', 'wor', 'worl', 'world', 'world</w>', 'world</w>']
    expected_token_to_id = {'He': 0, 'Hel': 1, 'Hell': 2, 'Hello': 3, 'Hello</w>': 5, 'Tu': 6, 'Tun': 7, 'Tuni': 8, 'Tunis': 9, 'Tunisi': 10, 'Tunisia': 11, 
                            'Tunisia</w>': 13, 'fr': 14, 'fro': 15, 'from': 16, 'from</w>': 18, 'wo': 19, 'wor': 20, 'worl': 21, 'world': 22, 'world</w>': 24}
    expected_id_to_token = {0: 'He', 1: 'Hel', 2: 'Hell', 3: 'Hello', 5: 'Hello</w>', 6: 'Tu', 7: 'Tun', 8: 'Tuni', 9: 'Tunis', 10: 'Tunisi', 11: 'Tunisia', 
                            13: 'Tunisia</w>', 14: 'fr', 15: 'fro', 16: 'from', 18: 'from</w>', 19: 'wo', 20: 'wor', 21: 'worl', 22: 'world', 24: 'world</w>'}
    assert vocab == expected_vocab
    assert token_to_id == expected_token_to_id
    assert id_to_token == expected_id_to_token


def test_from_text_to_ids(bpe):
    text = "Hello world from Tunisia"
    symbols = bpe.sentence_to_symbols(text)
    new_tokenized_words = bpe.keep_merging(25, symbols)
    vocab, token_to_id, id_to_token = bpe.build_vocab(symbols, new_tokenized_words)
    ids = bpe.from_text_to_ids(text, new_tokenized_words, token_to_id)
    expected_token_ids = [5, 24, 18, 13]
    assert ids == expected_token_ids