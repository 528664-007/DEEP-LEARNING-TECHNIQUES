# Experiment 10: Machine Translation using Encoder-Decoder Model

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data
eng_sentences = ['hello', 'how are you', 'good morning']
fr_sentences = ['<start> bonjour <end>', '<start> comment ca va <end>', '<start> bonjour <end>']

# Tokenization
eng_tokenizer = Tokenizer()
eng_tokenizer.fit_on_texts(eng_sentences)
eng_seq = pad_sequences(eng_tokenizer.texts_to_sequences(eng_sentences), padding='post')
eng_vocab_size = len(eng_tokenizer.word_index) + 1

fr_tokenizer = Tokenizer(filters='')
fr_tokenizer.fit_on_texts(fr_sentences)
fr_seq = [fr_tokenizer.texts_to_sequences([s])[0] for s in fr_sentences]
fr_input_seq = pad_sequences([s[:-1] for s in fr_seq], padding='post')
fr_target_seq = pad_sequences([s[1:] for s in fr_seq], padding='post')
fr_target_seq = np.expand_dims(fr_target_seq, -1)
fr_vocab_size = len(fr_tokenizer.word_index) + 1

embedding_dim = 64
latent_dim = 128

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(eng_vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm, state_h, state_c = LSTM(latent_dim, return_state=True)(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb = Embedding(fr_vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm, _, _ = LSTM(latent_dim, return_sequences=True, return_state=True)(dec_emb, initial_state=encoder_states)
decoder_outputs = Dense(fr_vocab_size, activation='softmax')(decoder_lstm)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit([eng_seq, fr_input_seq], fr_target_seq, batch_size=2, epochs=300, verbose=0)
print("Translation model trained.")
