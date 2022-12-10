from brief_news.ml_logic.preprocessor import cleaning
from brief_news.data.big_query import get_bq_chunk

import numpy as np
from keras import backend as K
# from tensorflow.keras import optimizers
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Attention
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

import warnings

warnings.filterwarnings("ignore")


def tokenization(train, val=None):
    tok = Tokenizer()
    tok.fit_on_texts(list(train))
    train_tok = tok.texts_to_sequences(train)

    if len(val)<1:
        return train_tok, tok

    val_tok = tok.texts_to_sequences(val)

    return train_tok, val_tok, tok

def padder(train_tok, maxlen, val_tok=[]):
    train_pad = pad_sequences(train_tok, dtype='float32',
                                maxlen=maxlen, padding='post')
    if len(val_tok)<1:
        return train_pad

    val_pad = pad_sequences(val_tok, dtype='float32',
                              maxlen=maxlen, padding='post')

    return train_pad, val_pad

def adding_decoder_tokens(data: pd.Series) -> pd.Series:
    '''
    Adding special tokens for the decoder only to target string
    '''

    return pd.Series(data).apply(lambda x : '_START_ '+ x + ' _END_')

def model_summ(latent_dim, max_len_text, X_vocab, y_vocab):

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

    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

    model_vault = [encoder_inputs, encoder_outputs, state_h, state_c, dec_emb_layer,
                   decoder_lstm, decoder_dense, decoder_inputs]

    return model, model_vault

def model_fit(model, X_train_pad, X_val_pad, y_train_pad, y_val_pad, max_len_summary):

    history = model.fit(
        [X_train_pad,y_train_pad[:,:-1]],
        y_train_pad.reshape(len(y_train_pad), max_len_summary, 1)[:,1:],
        batch_size=30,
        epochs=10,
        #callbacks=[es],
        validation_data=([X_val_pad,y_val_pad[:,:-1]],
                         y_val_pad.reshape(y_val_pad.shape[0],
                                           y_val_pad.shape[1], 1)[:,1:])
        #validation_split=0.1,
        )

    return model

def setup_model(X_train, X_val, y_train, y_val,
                latent_dim, max_len_text, max_len_summary):

    X_train_tok, X_val_tok, X_tokenizer = tokenization(X_train, X_val)
    y_train_tok, y_val_tok, y_tokenizer = tokenization(y_train, y_val)

    X_vocab = len(X_tokenizer.word_index) + 1
    y_vocab = len(y_tokenizer.word_index) + 1

    reverse_target_word_index=y_tokenizer.index_word
    reverse_source_word_index=X_tokenizer.index_word
    target_word_index=y_tokenizer.word_index

    X_train_pad, X_val_pad = padder(X_train_tok, max_len_text, val_tok=X_val_tok)
    y_train_pad, y_val_pad = padder(y_train_tok, max_len_summary, val_tok=y_val_tok)

    data_pad = [X_train_pad, X_val_pad, y_train_pad, y_val_pad]

    model, model_vault = model_summ(latent_dim, max_len_text, X_vocab, y_vocab)
    model = model_fit(model, X_train_pad, X_val_pad,
                      y_train_pad, y_val_pad, max_len_summary)

    [encoder_inputs, encoder_outputs, state_h, state_c, dec_emb_layer,
    decoder_lstm, decoder_dense, decoder_inputs] = model_vault

    # encoder inference
    encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

    # decoder inference
    # Below tensors will hold the states of the previous time step
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_hidden_state_input = Input(shape=(max_len_text,latent_dim))

    # Get the embeddings of the decoder sequence
    dec_emb2= dec_emb_layer(decoder_inputs)

    # To predict the next word in the sequence, set the initial states to the states from the previous time step
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

    #attention inference
    attention = model.layers[8]
    attn_out_inf = attention([decoder_outputs2, decoder_hidden_state_input])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

    # A dense softmax layer to generate prob dist. over the target vocabulary
    #decoder_outputs2 = decoder_dense(decoder_inf_concat)
    decoder_outputs2 = decoder_dense(decoder_inf_concat)

    # Final decoder model
    decoder_model = Model([decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2])

    decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

    return encoder_model, decoder_model, target_word_index, \
           reverse_target_word_index, reverse_source_word_index, \
           max_len_summary, data_pad



def decode_sequence(input_seq, encoder_model, decoder_model, target_word_index,
                    reverse_target_word_index, max_len_summary):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))

    # Chose the 'start' word as the first word of the target sequence
    target_seq[0, 0] = target_word_index['start']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if(sampled_token!='end'):
            decoded_sentence += ' '+sampled_token

            # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'end' or len(decoded_sentence.split()) >= (max_len_summary-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence

def seq2summary(input_seq, target_word_index, reverse_target_word_index):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=target_word_index['start']) and i!=target_word_index['end']):
            newString=newString+reverse_target_word_index[i]+' '
    return newString

def seq2text(input_seq, reverse_source_word_index):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index[i]+' '
    return newString


if __name__ == '__main__':

    #data
    train = get_bq_chunk('train', 0, 100)
    val = get_bq_chunk('validation', 0, 100)
    test = get_bq_chunk('test', 0, 100)

    #initializing the variables
    latent_dim = 500
    max_len_text = 150
    max_len_summary = 10

    #preprocessing
    X_train = cleaning(train.article)
    y_train = cleaning(train.highlights, remove_stopwords=False)

    X_val = cleaning(val.article)
    y_val = cleaning(val.highlights, remove_stopwords=False)

    X_test = cleaning(test.highlights, remove_stopwords=False)
    y_test = cleaning(test.article)

    y_train = adding_decoder_tokens(y_train)
    y_val = adding_decoder_tokens(y_val)

    ## Setting up the model and all the environment
    encoder_model, decoder_model, target_word_index,\
    reverse_target_word_index, reverse_source_word_index,\
    max_len_summary, data_pad = setup_model(X_train, X_val, y_train, y_val,
                latent_dim, max_len_text, max_len_summary)

    [X_train_pad, X_val_pad, y_train_pad, y_val_pad] = data_pad

    #generating summary
    for i in range(len(X_val_pad)):
        print("Review:",seq2text(X_val_pad[i], reverse_source_word_index))
        print("Original summary:",seq2summary(y_val_pad[i], target_word_index, reverse_target_word_index))
        print("Predicted summary:",decode_sequence(X_val_pad[i].reshape(1,max_len_text),
                                                encoder_model, decoder_model,
                                                target_word_index, reverse_target_word_index,
                                                max_len_summary))
        print("\n")
