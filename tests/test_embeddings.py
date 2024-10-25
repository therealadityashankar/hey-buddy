import psutil

def test_speech_embeddings():
    import torch
    from heybuddy.embeddings import get_speech_embeddings

    speech_embeddings = get_speech_embeddings()
    audio = torch.randn((17280,))
    embeddings, spectrogram = speech_embeddings(audio, return_spectrograms=True)
    assert spectrogram.shape == (1, 100, 32)
    assert embeddings.shape == (1, 4, 96)
    audio = torch.randn((23040,))
    embeddings, spectrogram = speech_embeddings(audio, return_spectrograms=True)
    assert spectrogram.shape == (1, 420, 32)
    assert embeddings.shape == (1, 16, 96)
