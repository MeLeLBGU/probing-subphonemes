import numpy as np
import random
from itertools import permutations
import numpy as np
import scipy.stats as stats
from panphon import sonority


def generate_lexicon(phoneme_vocab) -> list[str]:
    """
    Generate a lexicon of words based on the phoneme vocabulary provided.

    Taken the code from https://github.com/smuradoglu/phon-proc (Muradoglu & Hulden, ACL Findings 2023).
    We modified it to use the panphon sonority metric and removed diphtongs.

    Args:
        phoneme_vocab (list): A list of phonemes that are embedded in the model.
    """

    #

    random.seed(33)
    np.random.seed(33)

    CONSONANTS = [p for p in phoneme_vocab if ft.fts(p)["syl"] == -1]
    VOWELS = [p for p in phoneme_vocab if ft.fts(p)["syl"] == 1]
    S = sonority.Sonority()

    # %%
    # Generate syllable structure based weighted distribution of types of onsets, nuclei and codas
    def syll_struc():
        syll = ""
        onset = ["C", "", "CC"]
        n = ["V"]
        coda = ["C", "", "CC"]
        o = random.choices(onset, weights=(60, 30, 10), k=1)
        c = random.choices(coda, weights=(30, 60, 10), k=1)
        syll = o + n + c
        sylls = "".join(syll)
        return sylls

    # Calculate all possible permutations of 'CC'
    B = list(permutations(CONSONANTS, 2))

    # Convert tuple list to string list
    def CC(B):
        return ["".join(i) for i in permutations(CONSONANTS, 2)]

    J = CC(B)

    # %%# Replace phonemes with value according to Sonority Metric
    liste = []
    for p1, p2 in B:
        liste.append([S.sonority(p1), S.sonority(p2)])

    # %%#Calculate distance between two points/phonemes, allowing for direction
    def distance(p):
        x1 = p[0]
        x2 = p[1]
        return x1 - x2

    # Create list of distance values for all permutations of 'CC'
    sonor = []
    for item in liste:
        sonor.append(int(distance(item)))

    # Convert list of distance values to dictionary (i.e. distance value has mapping to 'CC')
    dictionary = dict(zip(J, sonor))

    # Filter for permutations with distance greater than 3 for coda
    d = dict((k, v) for k, v in dictionary.items() if v > 2)

    dictkeys_coda = list(d.keys())

    # Filter for permutations with distance less than -3 for onset
    e = dict((k, v) for k, v in dictionary.items() if v < -2)

    dictkeys_onset = list(e.keys())

    # %%##Replace syllable structure with phoneme inventory

    def syll_fill(syll):
        v = VOWELS
        c = CONSONANTS

        if "CC" in syll:
            cccount = syll.count("CC")
            ccinter = ["CC"] * cccount
            ccrand_c = random.choices(dictkeys_coda, k=cccount)
            ccrand_o = random.choices(dictkeys_onset, k=cccount)
            if syll.startswith("CC"):
                for i in range(cccount):
                    syll = syll.replace(ccinter[i], ccrand_o[i], 1)
            else:
                for i in range(cccount):
                    syll = syll.replace(ccinter[i], ccrand_c[i], 1)
            vcount = syll.count("V")
            ccount = syll.count("C")
            vinter = random.choices("V", k=vcount)
            cinter = random.choices("C", k=ccount)
            vrand = random.choices(v, k=vcount)
            crand = random.choices(c, k=ccount)
            drand = random.choices(d, k=dcount)
            for k in range(vcount):
                syll = syll.replace(vinter[k], vrand[k], 1)
            for j in range(ccount):
                syll = syll.replace(cinter[j], crand[j], 1)

        else:
            vcount = syll.count("V")
            ccount = syll.count("C")
            vinter = random.choices("V", k=vcount)
            cinter = random.choices("C", k=ccount)
            vrand = random.choices(v, k=vcount)
            crand = random.choices(c, k=ccount)
            for k in range(vcount):
                syll = syll.replace(vinter[k], vrand[k], 1)
            for j in range(ccount):
                syll = syll.replace(cinter[j], crand[j], 1)
        return syll

    # %%##Create words by joining syllable structures and replacing structures with phonemes, exclude invalid vowels: these would be treated as long vowels.
    def word():
        words = []
        invalidV = [char * 2 for char in VOWELS]

        for _ in range(100_000):
            num_sylls = np.random.randint(1, 10)
            s = ""
            for j in range(num_sylls):
                s += syll_struc()
            sf = syll_fill(s)
            if any(V in sf for V in invalidV):
                continue
            else:
                words.append(sf)
        return words

    w = word()

    # Convert to set & back to ensure no duplicate words
    WordSet = list(set(w))

    # Definte Gaussian
    def gaussian(x, mu=0, sig=1):
        return np.exp(-0.5 * ((x - mu) / (sig)) ** 2)

    # %%##Create list of word lengths
    LengthVector = []

    for word in WordSet:
        LengthVector.append(len(word))
    # start from 1 as there are no 0 length words
    x = np.linspace(1, np.max(LengthVector), 10000)

    # Estimate Kernel Density
    kernel = stats.gaussian_kde(LengthVector)
    LengthPDF = kernel(x)

    # %%##%% Define Sampling Distribution.
    # Mean 8 letters, std dev 4
    Sampling_PDF = gaussian(x, 8, 3)
    Norm = np.trapz(Sampling_PDF)
    # Normalize our gaussian so integral == 1
    Sampling_CDF = np.cumsum(Sampling_PDF / Norm)

    # %% Sample from Gaussian distribution
    # How many samples to take
    # NumberofTrials should be < WordSet so there are no duplicates

    NumberOfTrials = 50_000
    unVals = np.random.uniform(0, 1, NumberOfTrials)

    # Round sample length to nearest integer
    sampleLengths = []
    sampleIndex = []

    # Create copies of WordSet and LengthVector
    # This is because we will remove sampled words from these as we go
    LengthVectorCopy = LengthVector.copy()
    WordSetCopy = WordSet.copy()

    sampleset = []

    for unVal in unVals:
        SampleLength = np.round(x[np.argmin(np.abs(Sampling_CDF - unVal))])
        sampleLengths.append(SampleLength)

        # Find index of length vector where we match sample length
        idx = np.where(LengthVectorCopy == SampleLength)

        # We check if there are any more words of the required SampleLength
        if np.any(LengthVectorCopy == SampleLength):

            idx = np.random.choice(idx[0], 1).tolist()[0]
            # Get sampled word
            sampleset.append(WordSetCopy[idx])

            # remove the sampled word from length + WordSet
            WordSetCopy.pop(idx)
            LengthVectorCopy.pop(idx)

    # %%#
    # test sampling distribution

    SL = []

    for wugs in sampleset:
        SL.append(float(len(wugs)))

    return sampleset
