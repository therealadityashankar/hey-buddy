from heybuddy.tokenize import BERTTokenizer

def test_tokenizer():
    """
    Test the tokenizer class at a baseline using BERT
    """
    tokenizer = BERTTokenizer()
    assert tokenizer("Hello, world!").tolist() == [7592, 1010, 2088, 999]
    # Shoulds be case-insensitive
    assert tokenizer("hello, world!").tolist() == [7592, 1010, 2088, 999]
    assert "hello, world!" == tokenizer.decode([7592, 1010, 2088, 999])
    tokenizer.length = 10
    assert tokenizer("Hello, world!").tolist() == [7592, 1010, 2088, 999, 0, 0, 0, 0, 0, 0]
    assert "hello, world!" == tokenizer.decode([7592, 1010, 2088, 999, 0, 0, 0, 0, 0, 0])
