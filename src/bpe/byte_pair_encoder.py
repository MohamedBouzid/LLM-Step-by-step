from itertools import chain

class BytePairEncoder:

    END_TOKEN="</w>"

    def word_to_symbols(self, word: str) -> list[str] :
        result=list(word)
        result.append(self.END_TOKEN)
        return result

    def sentence_to_symbols(self, sentence: str) -> list[str] :
        words = sentence.split() 
        result=[]
        for word in words:
            result.append(self.word_to_symbols(word))
        print(result)
        return result

    def get_pair_counts(self, tokenized_words: list[list[str]]) -> dict[(str,str), int]:
        pair_count_map: dict[str, int] = {}
        for word in tokenized_words:
            for i in range(len(word)-1):
                pair=(word[i], word[i+1])
                pair_count_map[pair] = pair_count_map.get(pair, 0) + 1
        return pair_count_map

    def merge_pair(self, pair_count_map: dict[(str,str), int], tokenized_words: list[list[str]]) -> list[list[str]]:
        top_pair_key = max(pair_count_map, key=pair_count_map.get)
        for j in range(len(tokenized_words)):
            word=tokenized_words[j]
            indexes_to_remove=[]
            for i in range(len(word)-1):
                pair=(word[i], word[i+1])
                if pair == top_pair_key:
                    indexes_to_remove.append(i+1)
                    word[i] = pair[0] + pair[1]
                    tokenized_words[j] = word
                    i+=2
            for i in indexes_to_remove:
                word.pop(i)
        return tokenized_words


    def keep_merging(self, n: int, tokenized_words: list[list[str]]):
        merges=[]
        for i in range(n):
            pairs_count = self.get_pair_counts(tokenized_words)
            if len(pairs_count) == 0:
                break
            top_pair_key = max(pairs_count, key=pairs_count.get)
            merges.append(top_pair_key)
            tokenized_words = self.merge_pair(pairs_count, tokenized_words)
        return merges


    def apply_bpe(self, word: str, merges: list[(str,str)]) -> list[str]:
        symbols=self.word_to_symbols(word)
        for (a,b) in merges:
            i =0
            tokenized=[]
            while i < len(symbols):
                if (i < len(symbols)-1) and symbols[i] == a and symbols[i+1] == b:
                    tokenized.append((symbols[i] + symbols[i+1]))
                    i+=2
                else:
                    tokenized.append(symbols[i])
                    i+=1
            symbols=tokenized
        return tokenized

    def build_vocab(self, symbols: list[str], merges: list[(str, str)]):
        vocab = list(dict.fromkeys(chain.from_iterable(symbols)))
        for (a,b) in merges:
            vocab.append(a+b)
        
        vocab = list(vocab)
        vocab.sort()

        token_to_id = {tok: idx for idx, tok in enumerate(vocab)}
        id_to_token = {idx: tok for tok, idx in token_to_id.items()}
        return vocab, token_to_id, id_to_token

    def one_hot_encode(self, token_id, size):
        vector = [0] * size
        vector[token_id] = 1
        return vector

    def build_embedding_matrix(self, n,k):
        return [self.one_hot_encode(i,k) for i in range(n)]

    def generate_training_pairs(self, symbols: str, token_to_id: dict):
        print(f"symbols = {symbols}")
        pair_ids=[]
        for i in range(len(symbols)-1):
            pair_ids.append((token_to_id.get(symbols[i]), token_to_id.get(symbols[i+1])))

        return pair_ids


if __name__ == "__main__":
    
    bpe = BytePairEncoder()

    text = "Hello world from Tunisia"
    
    print("=== word_to_symbols ===")
    tokenized_words = bpe.word_to_symbols(text)
    symbols = tokenized_words.copy() 
    print(tokenized_words)
    print("\n")

    print("=== get_pair_counts ===")
    pair_count = bpe.get_pair_counts([tokenized_words])
    print(pair_count)
    print("\n")
    '''
    print("=== merge_pair ===")
    tokenized_words = merge_pair(pair_count, [tokenized_words])
    print(tokenized_words)
    print("\n")
    '''
    print("=== keep_merging ===")
    new_tokenized_words = bpe.keep_merging(10, [tokenized_words])
    print(new_tokenized_words)
    print("\n")
    
    print("=== apply_bpe ===")
    tokens_result = bpe.apply_bpe("Hellofromworld", new_tokenized_words)
    print(tokens_result)
    print("\n")

    print("=== build_vocab ===")
    vocab, token_to_id, id_to_token = bpe.build_vocab(symbols, new_tokenized_words)
    print(f"vocab = {vocab}")
    print(f"token_to_id = {token_to_id}")
    print(f"id_to_token = {id_to_token}")
    print("\n")

    print("=== one_hot_encoding ===")
    encoded_vector = bpe.one_hot_encode(3, 64)
    print(f"encoded_vector = {encoded_vector}")
    print("\n")

    print("=== build_embedding_matrix ===")
    embedding_matrix = bpe.build_embedding_matrix(3, 3)
    print(f"embedding matrix = {embedding_matrix}")
    print("\n")

    print("=== generate_training_pairs ===")
    pair_ids = bpe.generate_training_pairs(symbols, token_to_id)
    print(f"pair ids = {pair_ids}")
    print("\n")


    
