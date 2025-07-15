# Experiment 8: Dialogue Generation using LSTM with Attention

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Attention, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

input_texts = ['hello', 'how are you', 'what is your name', 'bye']
target_texts = ['<start> hi <end>', '<start> i am fine <end>', '<start> i am a bot <end>', '<start> goodbye <end>']

tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(input_texts + target_texts)
input_seqs = tokenizer.texts_to_sequences(input_texts)
target_seqs = tokenizer.texts_to_sequences(target_texts)
input_seqs = pad_sequences(input_seqs, padding='post')
target_seqs = pad_sequences(target_seqs, padding='post')

vocab_size = len(tokenizer.word_index) + 1
max_input_len = input_seqs.shape[1]
max_target_len = target_seqs.shape[1]

embedding_dim = 64
lstm_units = 128

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(vocab_size, embedding_dim)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(lstm_units, return_sequences=True, return_state=True)(enc_emb)

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb = Embedding(vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm_outputs, _, _ = LSTM(lstm_units, return_sequences=True, return_state=True)(dec_emb, initial_state=[state_h, state_c])

attention_layer = Attention()
context_vector = attention_layer([decoder_lstm_outputs, encoder_outputs])
combined = Concatenate(axis=-1)([context_vector, decoder_lstm_outputs])
decoder_outputs = Dense(vocab_size, activation='softmax')(combined)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

decoder_target_data = np.expand_dims(target_seqs, -1)
model.fit([input_seqs, target_seqs], decoder_target_data, batch_size=2, epochs=200, verbose=0)

print("Model trained for dialogue generation")
