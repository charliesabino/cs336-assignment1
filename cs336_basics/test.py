from tokenizer import Tokenizer
vocab = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
text = 'the cat ate'

def test_tokenizer():
    tokenizer = Tokenizer(vocab, merges)
    print(tokenizer.encode(text))

if __name__ == "__main__":
    test_tokenizer()