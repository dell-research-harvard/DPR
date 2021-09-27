import re
import pandas as pd
import os
import time
import pickle
from string import digits


def get_prop_non_allowed_chars(text):
    """
    Get proportion of non-allowed (non-basic-latin, non-general-punctuation)
    characters in a text.
    :param text: String for text to be checked.
    :return: Float proportion of non-allowed characters in text.
    """

    # set up non-allowed character counter
    non_allowed_chars = 0

    # establish unicode decimal ranges for allowed chars
    basic_latin_dec_lb, basic_latin_dec_ub = 0, 127
    gen_punct_dec_lb, gen_punct_dec_ub = 8192, 8303

    # assert text is not empty
    assert len(text) > 0

    # iterate through all characters in text
    for c in text:

        # get character
        unicode_dec = ord(c)

        # assess if allowed character; augment counter accordingly
        if basic_latin_dec_lb <= unicode_dec <= basic_latin_dec_ub or gen_punct_dec_lb <= unicode_dec <= gen_punct_dec_ub:
                pass
        else:
            non_allowed_chars += 1

    # get proportion of non allowed chars
    prop_non_allowed_chars = non_allowed_chars/len(text)

    return prop_non_allowed_chars


def load_lowercase_spell_dict(pickle_path):
    """
    Load a lowercase spell-checking dictionary from a pickled dictionary.
    :param pickle_path: String for path of pickled dictionary object.
    :return: Dict containing loaded spell-checking dictionary.
    """

    with open(pickle_path, 'rb') as fp:
        spell_dict = pickle.load(fp)
        return spell_dict


def get_prop_non_words(text, spell_dict, verbose=False):
    """
    Get proportion of misspelled words in a text.
    :param text: String for text to be checked.
    :param spell_dict: Dict for performing spell checking (where correctly spelled words
    are keys, all lower-cased).
    :param verbose: Bool for whether or not spell checking results at the word level should be printed.
    :return: Float proportion of incorrectly spelled words.
    """

    # lightly preprocess text
    text_lower = text.lower()
    text_lower_nonum = text_lower.translate(str.maketrans('', '', digits))
    text_lower_nopunct = text_lower_nonum.translate(str.maketrans('', '', r'!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~'))
    text_lower_nopunct_split = text_lower_nopunct.split()
    text_clean = text_lower_nopunct_split

    # return nwr of 1 for empty text
    if len(text_clean) == 0:
        return 1

    else:
        # perform spell checking
        dict_checked_words = [word in spell_dict.keys() for word in text_clean]

        # get proportion of correctly spelled words
        prop_correctly_spelled = sum(dict_checked_words)/len(text_clean)

        # print words found to be correctly or incorrectly spelled
        if verbose is True:
            df = pd.DataFrame({'word': text_clean, 'correctly_spelled': dict_checked_words})
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(df)

        return 1 - prop_correctly_spelled