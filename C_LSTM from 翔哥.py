from __future__ import print_function
import os
import numpy as np
np.random.seed(1337)  # for reproducibility
import sys
import fileinput

from matplotlib import pyplot as plt

#from keras.preprocessing import sequence
#from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers import Embedding, Flatten
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
#from keras.models import Model
from keras.models import Sequential
from keras.layers import Merge, LSTM
#from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras import backend as K

# set parameters:
max_features = 5000
maxlen = 400

# LSTM
lstm_output_size = 70

filters = 64
kernel_size = 5
hidden_dims = 128

batch_size = 32
epochs = 15
margin = 0.6
theta = lambda t: (K.sign(t) + 1.) / 2.
loss = lambda y_true, y_pred: -(1 - theta(y_true - margin) * theta(y_pred - margin) - theta(1 - margin - y_true) * theta(1 - margin - y_pred)) * (y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))

# Load data.
print('Loading data...')

BASE_DIR = 'D:/资料/建模/数模课'
GLOVE_DIR = BASE_DIR + '/glove.6B'  
TRAIN_DATA_DIR = BASE_DIR + '/train'
TEST_DATA_DIR = BASE_DIR + '/test'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000000
EMBEDDING_DIM = 300  # The dimention of the matrix 
VALIDATION_SPLIT = 0.2

# Index the word vectors.
#print('Indexing word vectors.')
#embeddings_index = {}
#f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'), encoding='UTF-8')
#for line in f:
#    values = line.split()
#    word = values[0]
#    coefs = np.asarray(values[1:], dtype='float32')
#    embeddings_index[word] = coefs
#f.close()
#print('Found %s word vectors.' % len(embeddings_index))
#
## Process train text dataset.
#print('Processing train text dataset')
#
#texts = []  # list of text samples
#labels = []  # list of label ids
#train_in = open(os.path.join(TRAIN_DATA_DIR, 'train.in'), encoding='UTF-8')
#train_out = open(os.path.join(TRAIN_DATA_DIR, 'train.out'), encoding='UTF-8')
#for line in train_in:
#    texts.append(line)
#train_in.close()
#for line in train_out:
#    labels.append(line)
#train_out.close()
#print('Found %s texts.' % len(texts))
#
## Vectorize the train text samples into a 2D integer tensor
#tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
#tokenizer.fit_on_texts(texts)
#sequences = tokenizer.texts_to_sequences(texts)
#
#word_index = tokenizer.word_index
#print('Found %s unique tokens.' % len(word_index))
#
#data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
#labels = to_categorical(np.asarray(labels))
#
#indices=[]
#for k in range(data.shape[0]):
#    if labels[k][0]==0:
#        indices.append(k)
#        
#indices=np.asarray(indices)
#temp1=data[indices]
#temp2=labels[indices]
#for i in range(4):
#    data = np.concatenate((data, temp1))
#    labels = np.concatenate((labels, temp2))     
#        
#indices = np.arange(data.shape[0])
#np.random.shuffle(indices)
#data = data[indices]
#labels = labels[indices]
#
#print('Shape of data tensor:', data.shape)
#print('Shape of label tensor:', labels.shape)
#x_train = data
#y_train = labels
#
## Process test text dataset.
#print('Processing test text dataset')
#
#texts1 = []  # list of text samples
#labels1 = []  # list of label ids
#
#test_in = open(os.path.join(TEST_DATA_DIR, 'test.in'), encoding='UTF-8')
#test_out = open(os.path.join(TEST_DATA_DIR, 'test.out'), encoding='UTF-8')
#for line in test_in:
#    texts1.append(line)
#test_in.close()
#for line in test_out:
#    labels1.append(line)
#test_out.close()
#print('Found %s texts.' % len(texts1))
#
## finally, vectorize the text samples into a 2D integer tensor
#tokenizer1 = Tokenizer(nb_words=MAX_NB_WORDS)
#tokenizer1.fit_on_texts(texts1)
#sequences1 = tokenizer1.texts_to_sequences(texts1)
#
#word_index1 = tokenizer1.word_index
#print('Found %s unique tokens.' % len(word_index1))
#
#data1 = pad_sequences(sequences1, maxlen=MAX_SEQUENCE_LENGTH)
#labels1 = to_categorical(np.asarray(labels1))
#x_test = data1
#y_test = labels1
#
#print('Preparing embedding matrix.')
## prepare embedding matrix
#nb_words = min(MAX_NB_WORDS, len(word_index))
#embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
#for word, i in word_index.items():
#    if i > MAX_NB_WORDS:
#        continue
#    embedding_vector = embeddings_index.get(word)
#    if embedding_vector is not None:
#        # words not found in embedding index will be all-zeros.
#        embedding_matrix[i] = embedding_vector # word_index to word_embedding_vector ,<20000(nb_words)
#print('Prepared embedding matrix.')
#
## load pre-trained word embeddings into an Embedding layer
#embedding_layer = Embedding(nb_words + 1,
#                            EMBEDDING_DIM,
#                            input_length=MAX_SEQUENCE_LENGTH,
#                            weights=[embedding_matrix],
#                            trainable=False)
# note that we set trainable = False so as to keep the embeddings fixed

print('Training model.')

##left model
#model_left = Sequential()
##model.add(Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'))
#model_left.add(embedding_layer)
##model_left.add(Dropout(0.2))
#model_left.add(Conv1D(64, 5, activation='relu'))
#model_left.add(MaxPooling1D(5))
#model_left.add(Conv1D(64,5, activation='relu'))
#model_left.add(MaxPooling1D(5))
#model_left.add(Conv1D(64, 5, activation='relu'))
#model_left.add(MaxPooling1D(35))
##model_left.add(LSTM(lstm_output_size))
#model_left.add(Flatten())
#
##right model
#model_right = Sequential()
#model_right.add(embedding_layer)
##model_right.add(Dropout(0.2))
#model_right.add(Conv1D(64, 4, activation='relu'))
#model_right.add(MaxPooling1D(4))
#model_right.add(Conv1D(64, 4, activation='relu'))
#model_right.add(MaxPooling1D(4))
#model_right.add(Conv1D(64, 4, activation='relu'))
#model_right.add(MaxPooling1D(28))
##model_right.add(LSTM(lstm_output_size))
#model_right.add(Flatten())
#
##third model
#model_3 = Sequential()
#model_3.add(embedding_layer)
##model_3.add(Dropout(0.2))
#model_3.add(Conv1D(64, 6, activation='relu'))
#model_3.add(MaxPooling1D(3))
#model_3.add(Conv1D(64, 6, activation='relu'))
#model_3.add(MaxPooling1D(3))
#model_3.add(Conv1D(64, 6, activation='relu'))
#model_3.add(MaxPooling1D(30))
##model_3.add(LSTM(lstm_output_size))
#model_3.add(Flatten())
#
#
#merged = Merge([model_left, model_right,model_3], mode='concat') #merge
#model = Sequential()
#model.add(merged) # add merge
##model.add(Dense(128, activation='tanh'))
#model.add(Dense(hidden_dims))
#model.add(Dropout(0.25))
##model.add(Activation('relu'))
#model.add(Dense(len(labels_index), activation='sigmoid'))


model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.25))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=5))
model.add(LSTM(lstm_output_size))
model.add(Dense(2))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validataion loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)