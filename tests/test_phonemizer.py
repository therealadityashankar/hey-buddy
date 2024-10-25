from heybuddy.phonemizer import get_phonemizer

def test_phonemizer():
    """
    Test the phonemizer class at a baseline
    """
    phonemizer = get_phonemizer(use_deep_phonemizer=True)
    assert phonemizer("hello world") == "[HH][AH][L][OW] [W][ER][L][D]"
    assert phonemizer("grilled cheese") == "[G][R][IH][L][D] [CH][IY][Z]"
    phonemizer = get_phonemizer(use_deep_phonemizer=False)
    assert phonemizer("hello world") == "[HH][AH][L][OW] [W][ER][L][D]"
    assert phonemizer("grilled cheese") == "[G][R][IH][L][D] [CH][IY][Z]"
