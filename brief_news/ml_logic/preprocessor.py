import pandas as pd
import matplotlib.pyplot as plt

import re
import string
import contractions
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from brief_news.ml_logic.params import MAX_LEN_SUM, MAX_LEN_TEXT


def preprocessing(sentence: string, remove_stopwords=True) -> string:

    """Preprocessing text: lower case,
                        deleting punctuation,
                        replacing contructions with equivalent,
                        deleting stop words,
                        removing special characters """

    # Lowercase
    sentence = sentence.lower()

    # Remove return characters, url and html tags
    code_list = ['\n', '\S*(http|https)\S*', '\<a href', '&amp;', '<br />']
    for code in code_list:
        sentence = re.sub(code, ' ',sentence, flags=re.MULTILINE)

    # expand the shortened words (can't => can not)
    # after they will be deleted in stopwords
    expanded = []
    for word in sentence.split():
        expanded.append(contractions.fix(word, slang=False))

    expanded_sentence = ' '.join(expanded)

    # remove any parenthisis with text inside
    sentence = re.sub(r'\([^)]*\)', '', expanded_sentence)

    # Removing punctuation, url and html tags
    for punctuation in string.punctuation + '[\'\"]':
        sentence = sentence.replace(punctuation, ' ')

    # remove special characters
    sentence = re.sub("[^a-zA-Z]", " ", sentence)
    # remove extra spaces
    sentence = re.sub("\s+", ' ', sentence)

    # Removing whitespaces
    sentence = sentence.strip()

    if remove_stopwords:
        stop_words = set(stopwords.words('english')) ## defining stopwords
        sentence_list = [w for w in sentence.split() if not w in stop_words]
        sentence = (' '.join(sentence_list)).strip()

    return sentence


def cleaning(dataset: pd.Series, remove_stopwords=True) -> list:
    """
    This function creates a cleaned version of each dataset.
    Calls the preprocessing function.
    """

    rmv_stop = remove_stopwords
    clean = dataset.apply(preprocessing, remove_stopwords=rmv_stop)
    return clean



def preprocessing_target(dataset: pd.Series, remove_stopwords=False) -> pd.Series:
    """
    !!!Only for target sentences.
    Cleaning target sentences.
    Adding special tokens for the decoder only to target string
    """
    target_clean = cleaning(dataset, remove_stopwords=False)
    target_preproc = pd.Series(target_clean).apply(lambda x : '_START_ '+ x + ' _END_')

    return target_preproc
