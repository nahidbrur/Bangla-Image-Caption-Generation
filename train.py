import os

import numpy as np
from numpy import array
from pickle import dump, load
from time import time
from keras.layers import LSTM, Embedding, Dense, Dropout
from keras.layers.merge import add
from keras.models import Model
from keras import Input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from config import Config
from utils import *
# data generator, intended to be used in a call to model.fit_generator()
def load_word_vector(file_path):
    embeddings_index = {} # empty dictionary
    f = open(file_path, encoding="utf-8")

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def get_embedding_matrix(word2vec_file_path, vocab_size, wordtoix, embedding_dim=300):
    embeddings_index = load_word_vector(word2vec_file_path)
    # Get 200-dim dense vector for each of the 10000 words in out vocabulary
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in wordtoix.items():
        #if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in the embedding index will be all zeros
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def load_features(pickle_file_path):
    features = pickle.load(open(pickle_file_path, "rb"))
    print('Photos: in pickle=%d' % len(features))
    return features

def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            n+=1
            # retrieve the photo feature
            photo = photos[key+'.jpg']
            for desc in desc_list:
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            # yield the batch data
            if n==num_photos_per_batch:
                yield [[array(X1), array(X2)], array(y)]
                X1, X2, y = list(), list(), list()
                n=0

def get_model(vocab_size, max_length, embedding_dim):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

def train():
    config = Config()
    
    train_descriptions = load_clean_descriptions(config.descriptions_file_path, config.train_file_path)
    print("Total discriptions : ", len(train_descriptions))
    vocab = get_vocab(train_descriptions)
    print("Total vocab : ", len(vocab))
    ixtoword, wordtoix = convert_ixtoword_and_wordtoix(vocab)
    vocab_size = len(ixtoword) + 1
    print("Vocab size : ", vocab_size)
    max_length = get_max_length(train_descriptions)
    print("Max caption length : ", max_length)
    print("Loading word vector...")
    
    embedding_matrix = get_embedding_matrix(config.word2vec_file_path, vocab_size, wordtoix, config.embedding_dim)
    print("Loading features...")
    
    train_features = load_features(config.train_features_file_path)
    print("Initializing model...")
    
    model = get_model(vocab_size, max_length, config.embedding_dim)
    model.layers[2].set_weights([embedding_matrix])
    model.layers[2].trainable = False
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    steps = len(train_descriptions)//config.batch_size

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    print("Model training...")
    for i in range(config.epochs):
        train_generator = data_generator(train_descriptions, train_features, wordtoix, max_length, config.batch_size)
        model.fit_generator(train_generator, epochs=1, steps_per_epoch=steps, verbose=1)
        model.save(os.path.join(config.checkpoint_dir, 'model_' + str(i) + '.h5'))
        if i == 50:
            model.optimizer.lr = 0.0001

if __name__ == '__main__':
    train()