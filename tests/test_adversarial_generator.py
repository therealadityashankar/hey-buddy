from heybuddy.util import get_adversarial_text_generator

def test_adversarial_generator():
    generator = get_adversarial_text_generator()
    generator.input_words_ratio = 0.0
    hw_list = list(generator("hello world", num_samples=5, seed=12345))
    assert len(hw_list) == 5
    assert hw_list[0] == "silicone madella"
    assert hw_list[1] == "collosio helionetics"
    assert hw_list[2] == "wiggled headrest"
    assert hw_list[3] == "dunkleberger magliolo"
    assert hw_list[4] == "melatonin armellino"
    generator.input_words_ratio = 0.5
    hw_list = list(generator("hello world", num_samples=5, seed=123456))
    assert hw_list[0] == "fogelberg world"
    assert hw_list[1] == "siebenaler nellie"
    assert hw_list[2] == "cytomegalovirus world"
    assert hw_list[3] == "galanis delmed's"
    assert hw_list[4] == "himalayan world"
