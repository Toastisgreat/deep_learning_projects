#General imports:
import pandas as pd

#vocabulary_bank
from numpy import asarray, array
from math import log10

#dataset_to_tensor_and_split
from tensorflow import convert_to_tensor

#sequential_nn
from keras import models
from keras import layers
from torch import tensor
from tensorflow import string

#conv_nn
import tensorflow as tf
'''
TODO:
    Save and Load Models automatically
'''
class vocabulary_bank():



    def __init__(self, name: str):
        PAD_token = 0
        SOS_token = 1
        EOS_token = 2
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3
        self.num_sentences = 0
        self.longest_sentence = 0
        self.sentences = []
        self.weights = []

    def add_word(self, word):
        if word == '':
            pass
        elif word not in self.word2index:
            #if word not in index, create index value and increase number of words
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            #if word is in index, increase word count by 1
            self.word2count[word] += 1
            
    
    def add_sentence(self, sentence):
        #count the number of words in the sentence
        sentence_len = 0
        for word in sentence.split(' '):
            sentence_len += 1
            #for each word, index it in the bank
            self.add_word(word)
        #if its the longest sentence, update longest_sentence value
        if sentence_len > self.longest_sentence:
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1
        self.sentences.append(sentence)


    def to_word(self, index) -> str:
        return self.index2word[index]

    def to_index(self, word) -> int:
        return self.word2index[word]
    
    def td_frequency(self, word: str) -> int:
        doccount = 0
        for sentence in self.sentences:
            if word in sentence:
                doccount += 1
            else:
                pass
        return doccount

    def tf_idf(self) -> array:
        tf_idf = []
        total_words = sum(list(self.word2count.values()))
        doc_count = len(self.sentences)
        for word, count in self.word2count.items():
            tf = count / total_words
            idf = log10(doc_count/ (self.td_frequency(word) + 1))
            tf_idf.append(tf*idf)
        return asarray(tf_idf)



def dataset_to_tensor_and_split(dataset: pd.DataFrame, testsize: float) -> tensor:
    #dataset has two collumns: 'text' and 'label'
    X = dataset['text']
    y = dataset['label'].values

    #Split and turn into tensors
    X_valid = convert_to_tensor(X[int(len(X)*testsize):])
    X_train = convert_to_tensor(X[:int(len(X)*testsize)])
    y_train = convert_to_tensor(y[:int(len(y)*testsize)])
    y_valid = convert_to_tensor(y[int(len(y)*testsize):])
    
    return X_train, y_train, X_valid, y_valid


def sequential_nn(X_train: tensor, y_train: tensor, X_valid: tensor, y_valid: tensor, units: list, type: str, n_gram_no: int, vocab_bank = vocabulary_bank) -> models.Sequential:

    #Open Sequentional model
    seqNN = models.Sequential()

    #Model order: train_shape -> units[0]-> dropout -> units[1] -> dropout -> units[2] -> 1
    

    #preprocessing layers

    #tf-idf calculation
    vocab_words = tuple(vocab_bank.word2count.keys())
    tfidf = vocab_bank.tf_idf()
    Vectorizer = layers.TextVectorization(
        output_mode='tf_idf',
        ngrams=n_gram_no,
        standardize='lower_and_strip_punctuation',
        vocabulary=vocab_words,
        idf_weights= tfidf
    )
    seqNN.add(layers.Input(shape=(1,), dtype=string))
    seqNN.add(Vectorizer)

    #Input layer
    seqNN.add(layers.Dense(units[0], activation = 'relu', input_shape=X_train.shape[0:]))

    #Hidden Layers
    seqNN.add(layers.Dropout(0.5, noise_shape=None, seed=None))
    seqNN.add(layers.Dense(units[1], activation = "relu"))
    seqNN.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    seqNN.add(layers.Dense(units[2], activation = "relu"))

    #output layers
    seqNN.add(layers.Dense(1, activation = "sigmoid"))

    #Tensorboard activation
    #Compile NN
    seqNN.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    #train
    seqNN.fit(
        X_train, y_train,
        epochs=2,
        batch_size=32,
        validation_data=(X_valid, y_valid), 
        verbose= 0 #this makes it quiet
    )

    #print results
    print(f'{type} Sequational Neural Network \nTest Score: {round(seqNN.evaluate(X_train, y_train, verbose=0)[1], 2)} ; Validation Score: {round(seqNN.evaluate(X_valid, y_valid, verbose=0)[1],2)}')

    return seqNN

def convolutional_nn(X_train: tensor, y_train: tensor, X_valid: tensor, y_valid: tensor, vocab: vocabulary_bank, type: str, n_gram_no: int) -> models.Sequential:
    
    #Open the model
    conv_nn = models.Sequential()


    #input layers
    vocab_words = tuple(vocab.word2count.keys())
    tfidf = vocab.tf_idf()
    Vectorizer = layers.TextVectorization(
        output_mode='tf_idf',
        ngrams=n_gram_no,
        standardize='lower_and_strip_punctuation',
        vocabulary=vocab_words,
        idf_weights= tfidf
    )
    conv_nn.add(layers.Input(shape=(1,), dtype=string))
    conv_nn.add(Vectorizer)


    conv_nn.add(layers.Embedding(
        max(list(vocab.word2count.values())),
        100,
        input_length=vocab.longest_sentence
    ))

    #hidden layers
    conv_nn.add(layers.Conv1D(filters=32, kernel_size=8, activation='relu'))
    conv_nn.add(layers.MaxPooling1D(pool_size=2))
    conv_nn.add(layers.Conv1D(filters=32, kernel_size=8, activation='relu'))
    conv_nn.add(layers.MaxPooling1D(pool_size=2))
    conv_nn.add(layers.Conv1D(filters=32, kernel_size=8, activation='relu'))

    conv_nn.add(layers.Flatten())
    #output layers
    conv_nn.add(layers.Dense(10, activation='relu'))
    conv_nn.add(layers.Dense(1, activation='sigmoid'))

    #Compile
    conv_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #fit
    conv_nn.fit(
        X_train, y_train,
        epochs=2,
        validation_data=(X_valid, y_valid),
        verbose=0
    )

    print(f'{type} Convolution Neural Network \nTest score: {round(conv_nn.evaluate(X_train, y_train, verbose=0)[1], 2)} ; Validation score: {round(conv_nn.evaluate(X_valid, y_valid, verbose=0)[1],2)}')
    
    return conv_nn

def bi_lstm(X_train: tensor, y_train: tensor, X_valid: tensor, y_valid: tensor, vocab: vocabulary_bank, n_gram_no: int, type: str) -> models.Sequential:


    #Open model

    bi_lstm = models.Sequential()

    #input layers

    vocab_words = tuple(vocab.word2count.keys())
    tfidf = vocab.tf_idf()
    Vectorizer = layers.TextVectorization(
        output_mode='tf_idf',
        ngrams=n_gram_no,
        standardize='lower_and_strip_punctuation',
        vocabulary=vocab_words,
        idf_weights= tfidf
    )

    bi_lstm.add(layers.Input(shape=(1,), dtype=string))
    bi_lstm.add(Vectorizer)


    bi_lstm.add(layers.Embedding(
        max(list(vocab.word2count.values())),
        100,
        input_length=vocab.longest_sentence
    ))

    #hidden layers

    bi_lstm.add(layers.Dense(50, activation='relu'))
    bi_lstm.add(layers.Dropout(0.5))
    bi_lstm.add(layers.Bidirectional(layers.LSTM(64)))
    bi_lstm.add(layers.Dropout(0.5))
    bi_lstm.add(layers.Dense(1, activation='sigmoid'))

    #compile
    bi_lstm.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')

    #fit data
    bi_lstm.fit(
        X_train, y_train,
        epochs=2,
        validation_data=(X_valid, y_valid),
    )

    print(f'{type} Bi_lstm Neural Network \nTest score: {round(bi_lstm.evaluate(X_train, y_train, verbose=0)[1], 2)} ; Validation score: {round(bi_lstm.evaluate(X_valid, y_valid, verbose=0)[1],2)}')

    return bi_lstm

def conv_to_bilstm(X_train: tensor, y_train: tensor, X_valid: tensor, y_valid: tensor, vocab: vocabulary_bank, n_gram_no: int, type: str):
    
    
    bi_lstm = models.Sequential()

    #input layers

    vocab_words = tuple(vocab.word2count.keys())
    tfidf = vocab.tf_idf()
    Vectorizer = layers.TextVectorization(
        output_mode='tf_idf',
        ngrams=n_gram_no,
        standardize='lower_and_strip_punctuation',
        vocabulary=vocab_words,
        idf_weights= tfidf
    )

    bi_lstm.add(layers.Input(shape=(1,), dtype=string))
    bi_lstm.add(Vectorizer)


    bi_lstm.add(layers.Embedding(
        max(list(vocab.word2count.values())),
        100,
        input_length=vocab.longest_sentence
    ))

    #hidden layers
    
    bi_lstm.add(layers.Conv1D(filters=32, kernel_size=8, activation='relu'))
    bi_lstm.add(layers.MaxPooling1D(pool_size=2))
    bi_lstm.add(layers.Dropout(0.5))
    bi_lstm.add(layers.Dense(50, activation='relu'))
    bi_lstm.add(layers.Bidirectional(layers.LSTM(80)))
    bi_lstm.add(layers.Dropout(0.5))
    bi_lstm.add(layers.Dense(1, activation='sigmoid'))
    #compile
    bi_lstm.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')

    #fit data
    bi_lstm.fit(
        X_train, y_train,
        epochs=2,
        validation_data=(X_valid, y_valid),
    )

    print(f'{type} Bi_lstm Neural Network \nTest score: {round(bi_lstm.evaluate(X_train, y_train, verbose=0)[1], 2)} ; Validation score: {round(bi_lstm.evaluate(X_valid, y_valid, verbose=0)[1],2)}')

    return bi_lstm

