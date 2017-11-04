from __future__ import print_function
import os
import numpy as np
from numpy.random import shuffle
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

# SMOTE code from
# https://github.com/MLevinson-OR/SMOTE/blob/master/src/SMOTE.py
# it can add neighbour noise to the dataset
# Error
def pca(X):
    '''reduce data'''
    pca = PCA(n_components=2)
    Xreduced = pca.fit_transform(X, y=None)
    return Xreduced

def partitionSamples(X,Y):
    minority_rows = []
    majority_rows = []
    for i,row in enumerate(Y):
        if(row == 1):
            minority_rows.append(i)
        else:
            majority_rows.append(i)
    return (X[minority_rows],X[majority_rows])

def chooseNeighbor(neighbors_index,numNeighbors,to_be_removed):
    indices = neighbors_index[0]
    index_list = indices.tolist()
    index_list.remove(to_be_removed)
    index_list_size = len(index_list)
    if(index_list_size < numNeighbors):
        raise Exception('the num of neighbors is less than the number of points in the cluster')
   
    elif(index_list_size == numNeighbors):    
        return index_list
    #remaining_rows = index_list.
    #create indices minus currRow
    else:
        listofselectedneighbors = []
        for i in range(numNeighbors):
            selected_index = random.choice(index_list)
            listofselectedneighbors.append(selected_index)
            index_list.remove(selected_index)
        return listofselectedneighbors

# DK觉得：nearestneigh 是初选中选的邻居数，可以选稍微大一些？
# numNeighbours是选择的邻居数目吧
def createSyntheticSamples(X,Y,nearestneigh,numNeighbors,majoritylabel,minoritylabel): 
    (Xminority,Xmajority) = partitionSamples(X,Y)
    numFeatures = Xminority.shape[1]
    Xreduced = pca(Xminority)  
    numOrigMinority = len(Xminority)
    #reducedMinoritykmeans = KMeans(init='k-means++',
    #max_iter=500,verbose=False,tol=1e-4,k=numCentroids, n_init=5,
    #n_neighbors=3).fit(Xreduced)
    reducedNN = NearestNeighbors(nearestneigh, algorithm='auto')
    reducedNN.fit(Xreduced)
    #Xsyn=np.array([numOrigMinority,numNeighbors*numFeatures])
    trylist = []
    #LOOPHERE for EACH (minority) point...
    for i,row in enumerate(Xreduced):
        #Expected 2D array, got 1D array instead:
        #array=[-1254.06656928   426.01329738].
        #Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
        #neighbor_index = reducedNN.kneighbors(row, return_distance=False) 

        neighbor_index = reducedNN.kneighbors(row.reshape(1,-1), return_distance=False) 

        closestPoints = Xminority[neighbor_index]
        #randomly choose one of the k nearest neighbors
        chosenNeighborsIndex = chooseNeighbor(neighbor_index,numNeighbors,i)
        chosenNeighbor = Xminority[chosenNeighborsIndex]
        #Calculate linear combination:
        #Take te difference between the orig minority sample and its selected
        #neighbor, where X[1,] is the orig point
        diff = Xminority[i,] - chosenNeighbor
        #Multiply this difference by a number between 0 and 1
        r = random.uniform(0,1)
        #Add it back to te orig minority vector and viola this is the synthetic
        #sample
        syth_sample = Xminority[i,:] + r * diff
        syth_sample2 = syth_sample.tolist()
        trylist.append(syth_sample2)
    Xsyn = np.asarray(trylist).reshape(numNeighbors * numOrigMinority,numFeatures)
    maj_col = majoritylabel * np.ones([Xmajority.shape[0],1])
    min_col = minoritylabel * np.ones([Xsyn.shape[0],1])
    syth_Y = np.concatenate((maj_col,min_col),axis=0)
    syth_X = np.concatenate((Xmajority,Xsyn),axis=0)
    if(syth_X.shape[0] != syth_Y.shape[0]):
        raise Exception("dim mismatch between features matrix and response matrix")
    return (syth_X, syth_Y)

# Load data.
print('Loading data...')

BASE_DIR = 'L:/学术/大三/数据挖掘2017/作业二'
GLOVE_DIR = BASE_DIR + '/glove.6B'  
TRAIN_DATA_DIR = BASE_DIR + '/train'
TEST_DATA_DIR = BASE_DIR + '/test'
MAX_SEQUENCE_LENGTH = 256 #Max length of text samples
MAX_NB_WORDS = 20000 #Max kinds of words
EMBEDDING_DIM = 200  # The dimention of the matrix
#VALIDATION_SPLIT = 0.3

# Index the word vectors.
print('Indexing word vectors.')
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.200d.txt'), encoding='UTF-8')
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

# Add noise to the data1 and label1 directly
# k is the number of centroids, num_neighbors is the number of neighbors per
# each minority sample
(data1,labels1) = createSyntheticSamples(data1,labels1,nearestneigh=10,
                                     numNeighbors=5,majoritylabel=0,minoritylabel=1) 

#Transform labels to 2-dim vectors
num1=0
num0=0
for i in range (10000):
    if (labels1[i]==1):
        num1=num1+1
    if (labels1[i]==0):
        num0=num0+1

labels1 = to_categorical(np.asarray(labels1))
labels2 = to_categorical(np.asarray(labels2))

#Add noised samples to make samples balanced

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
# CNN
# filters = 64 # The Output of CNN
# LSTM
lstm_output_size = 64  # 64 Characteristics
# Full-connection
hidden_dims = 128  

batch_size = 32
epochs = 10

model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.2))
model.add(Conv1D(128, 5, padding='same',strides=1, activation='relu'))
model.add(MaxPooling1D(4))
model.add(Conv1D(64, 4, padding='same',strides=1, activation='relu'))
model.add(MaxPooling1D(4))
model.add(BatchNormalization())
# model.add(Dropout(0.25))
# model.add(Flatten())
model.add(LSTM(lstm_output_size))
# model.add(Bidirectional(LSTM(128)))
# model.add(Dense(32, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])
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