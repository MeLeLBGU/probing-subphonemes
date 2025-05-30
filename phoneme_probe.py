import numpy as np
from mdl_probe import get_compression_score


def phoneme_probe(
    phonemes: list[str],
    phoneme_embeddings: list[float],
    is_weighted: bool = True,
    oversample: int = 1,
) -> tuple[float, float]:
    """
    Calculate the compression score for a given phonemes for all subphonemes.

    `phonemes` is a list of tuples phonemes that are embedded in the model.
    `phoneme_embeddings` is a list of floats representing the phoneme embeddings.
    `subphoneme` is a phonological feature name like 'voi'.
    """
    subphoneme_score = {}
    subphoneme_score_control = {}

    for subphoneme in FT.names:
        y = [int(FT.word_fts(p)[0][subphoneme]) for p in phonemes if FT.word_fts(p)]
        score, score_control = get_compression_score(
            phoneme_embeddings, y, is_weighted, oversample
        )
        subphoneme_score[subphoneme] = score
        subphoneme_score_control[subphoneme] = score_control

    return subphoneme_score, subphoneme_score_control
