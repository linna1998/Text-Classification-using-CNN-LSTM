from __future__ import print_function
import os
import numpy as np
from numpy.random import shuffle
from numpy import argmax
import random
np.random.seed(1337)  # for reproducibility
import numexpr as ne
from matplotlib import pyplot as plt
from heapq import heappush
from heapq import nsmallest
from sklearn.datasets import load_svmlight_file
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import hashlib
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Flatten
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
#from keras.models import Model
from keras.models import Sequential
from keras.layers import Merge, LSTM, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers.normalization import BatchNormalization
#from keras.layers.noise import GaussianNoise
from imblearn.over_sampling import SMOTE

# Load data.
print('Loading data...')

BASE_DIR = 'L:/学术/大三/数据挖掘2017/作业二'
GLOVE_DIR = BASE_DIR + '/glove.6B'  
TRAIN_DATA_DIR = BASE_DIR + '/train'
TEST_DATA_DIR = BASE_DIR + '/test'
RESULT_DATA_DIR=BASE_DIR+'/result'
MAX_SEQUENCE_LENGTH = 256 #Max length of text samples
MAX_NB_WORDS = 20000 #Max kinds of words
EMBEDDING_DIM = 300  # The dimention of the matrix
#VALIDATION_SPLIT = 0.3

# Index the word vectors.
print('Indexing word vectors.')
embeddings_index = {}
# Reduce the dim for debug.
f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'), encoding='UTF-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

# Process train text dataset.
print('Processing train text dataset')
texts1 = []  # list of text samples
labels1 = []  # list of label ids
train_in = open(os.path.join(TRAIN_DATA_DIR, 'train.in'), encoding='UTF-8')
train_out = open(os.path.join(TRAIN_DATA_DIR, 'train.out'), encoding='UTF-8')
for line in train_in:
    texts1.append(line)
train_in.close()
for line in train_out:
    # Change the labels into num 0/1
    labels1.append(int(line[0]))
train_out.close()
print('Found %s train texts.' % len(texts1))

# Process test text dataset.print('Processing test text dataset')
texts2 = []  # list of text samples
labels2 = []  # list of label ids
test_in = open(os.path.join(TEST_DATA_DIR, 'test.in'), encoding='UTF-8')
test_out = open(os.path.join(TEST_DATA_DIR, 'test.out'), encoding='UTF-8')
for line in test_in:
    texts2.append(line)
test_in.close()
for line in test_out:
    labels2.append(int(line[0]))
test_out.close()
print('Found %s test texts.' % len(texts2))

#Tokenize texts 分词
# 原来是分开的分词，现在是合成一个texts之后一起分词
texts = np.concatenate([texts1,texts2])
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences1 = tokenizer.texts_to_sequences(texts1)
sequences2 = tokenizer.texts_to_sequences(texts2)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data1 = pad_sequences(sequences1, maxlen=MAX_SEQUENCE_LENGTH)
data2 = pad_sequences(sequences2, maxlen=MAX_SEQUENCE_LENGTH)

# Add noise using SMOTE algorithm.
sm = SMOTE(ratio='auto')
data1,labels1 = sm.fit_sample(data1,labels1)

#Transform labels to 2-dim vectors
labels1 = to_categorical(np.asarray(labels1))
labels2 = to_categorical(np.asarray(labels2))

## Version 1.  Add noised samples to make samples balanced
#indices=[]
#for k in range(data1.shape[0]):
#    if labels1[k][0]==0:
#        indices.append(k)
        
#indices=np.asarray(indices)

#for i in range(4):
#    temp = data1[indices]
#    indices1 = np.random.randint(MAX_SEQUENCE_LENGTH, size=(1000,
#    int(MAX_SEQUENCE_LENGTH)))
#    for j in range(1000):
#        temp[j][indices1[j]] = temp[j][indices1[j]] *
#        np.random.uniform(low=0.9, high=1.1, size=(int(MAX_SEQUENCE_LENGTH)))
#    data1 = np.concatenate((data1, temp))
#    labels1 = np.concatenate((labels1, labels1[indices]))
for i in range(data1.shape[0]):
    for j in range(MAX_SEQUENCE_LENGTH):
        if data1[i][j] < 0:
            data1[i][j] = 0
        if data1[i][j] > len(word_index):
            data1[i][j] = len(word_index)
            
indices = np.arange(data1.shape[0])
np.random.shuffle(indices)
data1 = data1[indices]
labels1 = labels1[indices]

print('Shape of train data tensor:', data1.shape)
print('Shape of train label tensor:', labels1.shape)
print('Shape of test data tensor:', data2.shape)
print('Shape of train label tensor:', labels2.shape)

x_train = data1
y_train = labels1
x_test = data2
y_test = labels2

#delete some variables to save space
del texts
del indices
del data1
del labels1
del texts1
del data2
del labels2
del texts2

# prepare embedding matrix
print('Preparing embedding matrix.')
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector # word_index to word_embedding_vector ,<20000(nb_words)
print('Prepared embedding matrix.')

# load pre-trained word embeddings into an Embedding layer
embedding_layer = Embedding(nb_words + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH,
                            weights=[embedding_matrix],
                            trainable=False)
# Note that we set trainable = False so as to keep the embeddings fixed
print('Training model.')
# set parameters:
batch_size = 16
epochs = 15

model = Sequential()
model.add(embedding_layer)
#model.add(Dropout(0.25))
model.add(Conv1D(64, 5, padding='same', activation='relu'))
model.add(MaxPooling1D(4))
model.add(Conv1D(64, 4, padding='same', activation='relu'))
model.add(MaxPooling1D(4))
#model.add(Dropout(0.2))
#model.add(Flatten())
model.add(LSTM(64))
#model.add(Bidirectional(LSTM(32)))
#model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

predict_labels2=model.predict(x_test)
f1 = open(os.path.join(RESULT_DATA_DIR, 'file1.txt'),'w+')
f2 = open(os.path.join(RESULT_DATA_DIR, 'file2.txt'),'w+')
for i in range(predict_labels2.shape[0]):
    print(predict_labels2[i][1],file=f2)  # 为作者1的概率
    if predict_labels2[i][0]>0.5:
        print('0',file=f1)
    else:
        print('1',file=f1)

f1.close()
f2.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validataion loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model train vs validataion acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)