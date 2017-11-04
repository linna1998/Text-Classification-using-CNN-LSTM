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
from keras.layers import Conv1D, MaxPooling1D
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

# LSTM
lstm_output_size = 70  # 70个特征

filters = 64  # CNN输出，
kernel_size = 5  # 卷积大小。相邻5个卷积
hidden_dims = 250  # 最高层，投票神经网络的节点数。可以改小一点

batch_size = 32
epochs = 15
margin = 0.6
# theta = lambda t: (K.sign(t) + 1.) / 2.
# loss = lambda y_true, y_pred: -(1 - theta(y_true - margin) * theta(y_pred - margin) - theta(1 - margin - y_true) * theta(1 - margin - y_pred)) * (y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))

# Load data.
print('Loading data...')

BASE_DIR = 'L:/学术/大三/数据挖掘2017/作业二' 
GLOVE_DIR = BASE_DIR + '/glove.6B'  
TRAIN_DATA_DIR = BASE_DIR + '/train'
TEST_DATA_DIR = BASE_DIR + '/test'
MAX_SEQUENCE_LENGTH = 400  # 300*5000->400 ？？？
MAX_NB_WORDS = 20000000  # =max features， 只提取句子的前若干个词。我句子小的话，可以改小一点
# 维数，越大越好&越慢
EMBEDDING_DIM = 300  # The dimention of the matrix 
VALIDATION_SPLIT = 0.2  # 验证集切分

# Index the word vectors.
print('Indexing word vectors.')
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'), encoding='UTF-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

# Process test text dataset.
print('Processing test text dataset')

test_texts = []  # list of text samples
test_labels = []  # list of label ids
test_in = open(os.path.join(TEST_DATA_DIR, 'test.in'), encoding='UTF-8')
test_out = open(os.path.join(TEST_DATA_DIR, 'test.out'), encoding='UTF-8')
for line in test_in:
    test_texts.append(line)
test_in.close()
for line in test_out:
    test_labels.append(line)
test_out.close()
print('Found %s texts.' % len(test_texts))

# finally, vectorize the text samples into a 2D integer tensor
test_tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
test_tokenizer.fit_on_texts(test_texts)
test_sequences = test_tokenizer.texts_to_sequences(test_texts)

test_word_index = test_tokenizer.word_index
print('Found %s unique tokens.' % len(test_word_index))

test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_labels = to_categorical(np.asarray(test_labels))
x_test = test_data
y_test = test_labels

#def build_model(i):
# Process train1 text dataset.
print('Processing train1 text dataset')

texts1 = []  # list of text samples
labels1 = []  # list of label ids
# 1000:5000的训练集 train
# 做成1000:1000的训练集 chosen_train1
train1_in = open(os.path.join(TRAIN_DATA_DIR, 'chosen_train1.in'), encoding='UTF-8')
train1_out = open(os.path.join(TRAIN_DATA_DIR, 'chosen_train1.out'), encoding='UTF-8')
for line in train1_in:
    texts1.append(line)
train1_in.close()
for line in train1_out:
    labels1.append(line)
train1_out.close()
print('Found %s texts.' % len(texts1))
#对的还没有随机去做QAQ
# Vectorize the train text samples into a 2D integer tensor
tokenizer1 = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer1.fit_on_texts(texts1)
sequences1 = tokenizer1.texts_to_sequences(texts1)

word_index1 = tokenizer1.word_index
print('Found %s unique tokens.' % len(word_index1))

data1 = pad_sequences(sequences1, maxlen=MAX_SEQUENCE_LENGTH)
labels1 = to_categorical(np.asarray(labels1))
x_train1 = data1
y_train1 = labels1

print('Shape of data1 tensor:', data1.shape)
print('Shape of label1 tensor:', labels1.shape)

print('Preparing embedding matrix1.')
# prepare embedding matrix
nb_words1 = min(MAX_NB_WORDS, len(word_index1))
embedding_matrix1 = np.zeros((nb_words1 + 1, EMBEDDING_DIM))
for word, i in word_index1.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector1 = embeddings_index.get(word)
    if embedding_vector1 is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix1[i] = embedding_vector1 # word_index to word_embedding_vector1 ,<20000(nb_words)

print('Prepared embedding matrix1.')

# load pre-trained word embeddings into an Embedding layer
embedding_layer1 = Embedding(nb_words1 + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH,
                            weights=[embedding_matrix1],
                            trainable=False)
# note that we set trainable = False so as to keep the embeddings fixed
#    return train1_in, train1_out

# Process train2 text dataset.
print('Processing train2 text dataset')

texts2 = []  # list of text samples
labels2 = []  # list of label ids
# 1000:1000左右的训练集 chosen_train2
train2_in = open(os.path.join(TRAIN_DATA_DIR, 'chosen_train2.in'), encoding='UTF-8')
train2_out = open(os.path.join(TRAIN_DATA_DIR, 'chosen_train2.out'), encoding='UTF-8')
for line in train2_in:
    texts2.append(line)
train2_in.close()
for line in train2_out:
    labels2.append(line)
train2_out.close()
print('Found %s texts.' % len(texts2))

# Vectorize the train text samples into a 2D integer tensor
tokenizer2 = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer2.fit_on_texts(texts2)
sequences2 = tokenizer2.texts_to_sequences(texts2)

word_index2 = tokenizer2.word_index
print('Found %s unique tokens.' % len(word_index2))

data2 = pad_sequences(sequences2, maxlen=MAX_SEQUENCE_LENGTH)
labels2 = to_categorical(np.asarray(labels2))
x_train2 = data2
y_train2 = labels2

print('Shape of data2 tensor:', data2.shape)
print('Shape of label2 tensor:', labels2.shape)

print('Preparing embedding matrix2.')
# prepare embedding matrix
nb_words2 = min(MAX_NB_WORDS, len(word_index2))
embedding_matrix2 = np.zeros((nb_words2 + 1, EMBEDDING_DIM))
for word, i in word_index2.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector2 = embeddings_index.get(word)
    if embedding_vector2 is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix2[i] = embedding_vector2 # word_index to word_embedding_vector2 ,<20000(nb_words)

print('Prepared embedding matrix2.')

# load pre-trained word embeddings into an Embedding layer
embedding_layer2 = Embedding(nb_words2 + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH,
                            weights=[embedding_matrix2],
                            trainable=False)
# note that we set trainable = False so as to keep the embeddings fixed

## train a 1D convnet with global maxpoolinnb_wordsg

##left model
#model_left = Sequential()
##model.add(Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'))
#model_left.add(embedding_layer)
#model_left.add(Conv1D(64, 5, activation='relu'))
#model_left.add(MaxPooling1D(5))
#model_left.add(Conv1D(64,5, activation='relu'))
#model_left.add(MaxPooling1D(5))
#model_left.add(Conv1D(64, 5, activation='relu'))
##model_left.add(MaxPooling1D(35))
#model_left.add(LSTM(lstm_output_size))
##model_left.add(Flatten())

##right model
#model_right = Sequential()
#model_right.add(embedding_layer)
#model_right.add(Conv1D(64, 4, activation='relu'))
#model_right.add(MaxPooling1D(4))
#model_right.add(Conv1D(64, 4, activation='relu'))
#model_right.add(MaxPooling1D(4))
#model_right.add(Conv1D(64, 4, activation='relu'))
##model_right.add(MaxPooling1D(28))
#model_right.add(LSTM(lstm_output_size))
##model_right.add(Flatten())

##third model

#merged = Merge([model_left, model_right,model_3], mode='concat') #merge
#model = Sequential()
#model.add(merged) # add merge
#model = Sequential()
#model.add(embedding_layer)
#model.add(Embedding(nb_words+1,
#        EMBEDDING_DIM,
#        input_length=MAX_SEQUENCE_LENGTH))

## model.add(Dropout(0.2))
#model.add(Conv1D(64, 6, activation='relu'))
#model.add(MaxPooling1D(3))
#model.add(Conv1D(64, 6, activation='relu'))
#model.add(MaxPooling1D(3))
#model.add(Conv1D(64, 6, activation='relu'))
#model.add(MaxPooling1D(30))
#model.add(LSTM(lstm_output_size))
###model.add(Flatten())
###model.add(Dense(128, activation='tanh'))
#model.add(Dense(hidden_dims))
## model.add(Dropout(0.2))
##model.add(Activation('relu'))
##model.add(Dense(len(labels_index), activation='sigmoid'))
#model.add(Dense(2, activation='sigmoid'))  # len(labels_index)=2

print('Training model1.')
model1 = Sequential()
model1.add(embedding_layer1)
model1.add(Dropout(0.25))
model1.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model1.add(MaxPooling1D(pool_size=5))
model1.add(LSTM(lstm_output_size))
model1.add(Dense(2))
model1.add(Activation('sigmoid'))
model1.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history1 = model1.fit(x_train1, y_train1,
          batch_size=batch_size,
          nb_epoch=epochs,
          validation_data=(x_test, y_test))

print('Training model2.')
model2 = Sequential()
model2.add(embedding_layer2)
model2.add(Dropout(0.25))
model2.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model2.add(MaxPooling1D(pool_size=5))
model2.add(LSTM(lstm_output_size))
model2.add(Dense(2))
model2.add(Activation('sigmoid'))
model2.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history2 = model2.fit(x_train2, y_train2,
          batch_size=batch_size,
          nb_epoch=epochs,
          validation_data=(x_test, y_test))

print('Merging model.')
merged = Merge([model1, model2], mode='concat') #merge
model = Sequential()
model.add(merged) # add merge
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

## 现在history展示的是model1的训练
## 怎么做一个model1-model10完的投票？不知道。。。
## 能不能写成一个函数？不知道。。。
##可能是merge的时候要求层数维数相同
##而之前我天真的按照模5的余数在做，于是并没有能够merge到一起 QAQ
##决定再批量出来一堆正好1000：1000的数据集23333
#啊啊啊这里对合并之后的model要用什么去训练啊啊啊QAQ
history = model.fit(x_train1, y_train1,
          batch_size=batch_size,
          nb_epoch=epochs,
          validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

# 展示训练的过程 loss图
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validataion loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

# 展示训练的过程，accuracy图
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model train vs validataion acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
