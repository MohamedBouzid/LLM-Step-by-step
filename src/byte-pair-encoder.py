
END_TOKEN="</w>"

def word_to_symbols(word: str) -> list[str] :
    result=list(word)
    result.append(END_TOKEN)
    return result

def get_pair_counts(tokenized_words: list[list[str]]) -> dict[(str,str), int]:
    pair_count_map: dict[str, int] = {}
    for word in tokenized_words:
        for i in range(len(word)-1):
            pair=(word[i], word[i+1])
            pair_count_map[pair] = pair_count_map.get(pair, 0) + 1
    return pair_count_map


def merge_pair(pair_count_map: dict[(str,str), int], tokenized_words: list[list[str]]) -> list[list[str]]:
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


def keep_merging(n: int, tokenized_words: list[list[str]]):
    merges=[]
    for i in range(n):
        pairs_count = get_pair_counts(tokenized_words)
        if len(pairs_count) == 0:
           break
        top_pair_key = max(pairs_count, key=pairs_count.get)
        merges.append(top_pair_key)
        tokenized_words = merge_pair(pairs_count, tokenized_words)
    return merges


if __name__ == "__main__":

    print("=== word_to_symbols ===")
    tokenized_words = word_to_symbols("Helloworld")
    print(tokenized_words)
    print("\n")

    print("=== get_pair_counts ===")
    pair_count = get_pair_counts([tokenized_words])
    print(pair_count)
    print("\n")

    print("=== merge_pair ===")
    tokenized_words = merge_pair(pair_count, [tokenized_words])
    print(tokenized_words)
    print("\n")

    print("=== keep_merging ===")
    new_tokenized_words = keep_merging(10, tokenized_words)
    print(new_tokenized_words)
    print("\n")
