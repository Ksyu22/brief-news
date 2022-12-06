from brief_news.ml_logic.preprocessor import cleaning, preprocessing_target, tokenizing

import numpy as np
from keras import backend as K
from tensorflow.keras import optimizers
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional, Attention
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings("ignore")



# X_train = cleaning(train.article)
# y_train = cleaning(train.highlights, remove_stopwords=False)

# X_val = cleaning(val.article)
# y_val = cleaning(val.highlights, remove_stopwords=False)

# X_test = cleaning(test.highlights, remove_stopwords=False)
# y_test = cleaning(test.article)

# y_train = adding_decoder_tokens(y_train)
# y_val = adding_decoder_tokens(y_val)

def adding_decoder_tokens(data: pd.Series) -> pd.Series:
    '''
    Adding special tokens for the decoder only to target string
    '''

    return pd.Series(data).apply(lambda x : '_START_ '+ x + ' _END_')

def model_layers(latent_dim, embedding_dim):
    encoder_inputs = Input(shape=(max_len_text,))
    enc_emb = Embedding(X_vocab, latent_dim,trainable=True)(encoder_inputs)

    # LSTM 1
    # first integer shown in the brackets is the "dimensionality of the output space". so, that would be the length of the output summary, right?
    # return_sequences = Boolean. Whether to return the last output in the output sequence, or the full sequence. Default: False.
    # return_state = Boolean. Whether to return the last state in addition to the output. Default: False.

    encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True)
    encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

    #LSTM 2
    encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True)
    encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

    #LSTM 3
    encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True)
    encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)

    decoder_inputs = Input(shape=(None,))
    dec_emb_layer = Embedding(y_vocab, latent_dim,trainable=True)
    dec_emb = dec_emb_layer(decoder_inputs)

    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

    attention = Attention(name='attention_layer')
    attn_out = attention([decoder_outputs, encoder_outputs])

    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

    decoder_dense = TimeDistributed(Dense(y_vocab, activation='softmax'))

    decoder_outputs = decoder_dense(decoder_concat_input)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model
