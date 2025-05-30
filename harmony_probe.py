from mdl_probe import get_compression_score
from collections import defaultdict
from phonological_features import (
    FeatureTable as FT,
)  # Assuming FT is defined in phonological_features

IPA_VOWELS_LIST = [
    "i",
    "y",
    "ɨ",
    "ʉ",
    "ɯ",
    "u",
    "ɪ",
    "ʏ",
    "ʊ",
    "e",
    "ø",
    "ɘ",
    "ɵ",
    "ɤ",
    "o",
    "ə",
    "ɛ",
    "œ",
    "ɜ",
    "ɞ",
    "ʌ",
    "ɔ",
    "æ",
    "ɐ",
    "a",
    "ɶ",
    "ä",
    "ɑ",
    "ɒ",
]  # taken from https://en.wikipedia.org/wiki/IPA_vowel_chart_with_audio


def is_character_vowel(ch: str) -> bool:
    for c in ch:
        if c in IPA_VOWELS_LIST:
            return True
    return False


def get_harmony_label(word, subphoneme, is_vowel) -> int:
    word_as_features = [
        x[subphoneme]
        for ph, x in zip(FT.ipa_segs(word), FT.word_fts(word))
        if is_character_vowel(ph) == is_vowel
    ]
    feature_set = set(word_as_features)
    if len(feature_set) > 1:
        return 0
    if len(feature_set) == 0:
        return 0
    return list(feature_set)[0]


def harmony_probe(
    words: list[str],
    words_embeddings: list[float],
    phoneme_vocab: list[str],
    is_weighted: bool = True,
    oversample: int = 1,
):
    """
    Calculate the compression score for given words for all subphonemes, for both consonant and vowel haromony.

    `words` (list): the lexicon that the model computed its embeddings.
    `words_embeddings` (list): a list of floats representing the words embeddings.
    `subphoneme` (str): is a phonological feature name like 'voi'.
    `phoneme_vocab` (list): A list of phonemes that are embedded in the model.
    """
    subphoneme_score = defaultdict(dict)
    subphoneme_score_control = defaultdict(dict)

    valid_features_cons = []
    valid_features_vow = []

    cons_vocab = [
        ph for ph in FT.ipa_segs("".join(phoneme_vocab)) if not is_character_vowel(ph)
    ]
    vowel_vocab = [
        ph for ph in FT.ipa_segs("".join(phoneme_vocab)) if is_character_vowel(ph)
    ]
    for f in FT.names:
        if (
            len([ph for ph in cons_vocab if FT.fts(ph)[f] == -1]) >= 2
            and len([ph for ph in cons_vocab if FT.fts(ph)[f] == 1]) >= 2
        ):
            valid_features_cons.append(f)
    for f in FT.names:
        if (
            len([ph for ph in vowel_vocab if FT.fts(ph)[f] == -1]) >= 2
            and len([ph for ph in vowel_vocab if FT.fts(ph)[f] == 1]) >= 2
        ):
            valid_features_vow.append(f)
    y = {
        "consonant": {
            f: [get_harmony_label(w, f, is_vowel=False) for w in words]
            for f in valid_features_cons
        },
        "vowel": {
            f: [get_harmony_label(w, f, is_vowel=True) for w in words]
            for f in valid_features_vow
        },
    }

    for harmony_type in ["vowel", "consonant"]:
        for subphoneme in FT.names:
            score, score_control = get_compression_score(
                words_embeddings, y[harmony_type], is_weighted, oversample
            )
            subphoneme_score[harmony_type][subphoneme] = score
            subphoneme_score_control[harmony_type][subphoneme] = score_control

    return subphoneme_score, subphoneme_score_control
