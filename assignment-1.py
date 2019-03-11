#!/usr/bin/env python
# coding: utf-8

# In[1]:


#--------------------------importing data-----------------------------#
import nltk
from nltk.corpus import reuters
nltk.download('reuters')
from nltk.corpus import stopwords
nltk.download('stopwords')
import numpy as np


# In[2]:


print(">>> The reuters corpus has {} tags".format(len(reuters.categories())))
print(">>> The reuters corpus has {} documents".format(len(reuters.fileids())))


# In[3]:


#--------------------------exploring data-----------------------------#
import pandas as pd# create counter to summarize
categories = []
file_count = []

# count each tag's number of documents
for i in reuters.categories():
    """print("$ There are {} documents included in topic \"{}\""
          .format(len(reuters.fileids(i)), i))"""
    file_count.append(len(reuters.fileids(i)))
    categories.append(i)

# create a dataframe out of the counts
df = pd.DataFrame(
    {'categories': categories, "file_count": file_count}) \
    .sort_values('file_count', ascending=False)
print(df)


# In[4]:


#--------------------------Separating train and test data-----------------------------#
doc_list = np.array(reuters.fileids())

test_doc = doc_list[['test' in x for x in doc_list]]
print(">>> test_doc is created with following document names: {} ...".format(test_doc[0:5]))
train_doc = doc_list[['training' in x for x in doc_list]]
print(">>> train_doc is created with following document names: {} ...".format(train_doc[0:5]))

test_corpus = [" ".join([t for t in reuters.words(test_doc[t])])
               for t in range(len(test_doc))]
print(">>> test_corpus is created, the first line is: {} ...".format(test_corpus[0][:100]))
train_corpus = [" ".join([t for t in reuters.words(train_doc[t])])
                for t in range(len(train_doc))]
print(">>> train_corpus is created, the first line is: {} ...".format(train_corpus[0][:100]))


# In[5]:


#--------------------------cleaning data-----------------------------#
import text_clean as tc

# create clean corpus for word2vec approach
test_clean_string = tc.clean_corpus(test_corpus)
train_clean_string = tc.clean_corpus(train_corpus)
test_corpus=[]
for row in test_clean_string:
    row1=row.split()
    test_corpus.append(row1)

train_corpus=[]
for row in train_clean_string:
    row1=row.split()
    train_corpus.append(row1)


# In[6]:



import numpy as np
import re
from collections import defaultdict

#--- CONSTANTS ----------------------------------------------------------------+


class word2vec():
    def __init__ (self):
        self.n = settings['n']
        self.eta = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window_size']
        pass
    
    
    # GENERATE TRAINING DATA
    def generate_training_data(self, settings, corpus):

        # GENERATE WORD COUNTS
        self.word_counts = defaultdict(int)
        for row in corpus:
            for word in row:
                self.word_counts[word] += 1

        self.v_count = len(self.word_counts.keys())

        # GENERATE LOOKUP DICTIONARIES
        self.words_list = sorted(list(self.word_counts.keys()),reverse=False)
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))

        training_data = []
        # CYCLE THROUGH EACH SENTENCE IN CORPUS
        for sentence in corpus:
            sent_len = len(sentence)

            # CYCLE THROUGH EACH WORD IN SENTENCE
            for i, word in enumerate(sentence):
                
                #w_target  = sentence[i]
                w_target = self.word2onehot(sentence[i])

                # CYCLE THROUGH CONTEXT WINDOW
                w_context = []
                for j in range(i-self.window, i+self.window+1):
                    if j!=i and j<=sent_len-1 and j>=0:
                        w_context.append(self.word2onehot(sentence[j]))
                training_data.append([w_target, w_context])
        return np.array(training_data)


    # SOFTMAX ACTIVATION FUNCTION
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)


    # CONVERT WORD TO ONE HOT ENCODING
    def word2onehot(self, word):
        word_vec = [0 for i in range(0, self.v_count)]
        word_index = self.word_index[word]
        word_vec[word_index] = 1
        return word_vec


    # FORWARD PASS
    def forward_pass(self, x):
        h = np.dot(self.w1.T, x)
        u = np.dot(self.w2.T, h)
        y_c = self.softmax(u)
        return y_c, h, u
                

    # BACKPROPAGATION
    def backprop(self, e, h, x):
        dl_dw2 = np.outer(h, e)  
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))

        # UPDATE WEIGHTS
        self.w1 = self.w1 - (self.eta * dl_dw1)
        self.w2 = self.w2 - (self.eta * dl_dw2)
        pass


    # TRAIN W2V model
    def train(self, training_data):
        # INITIALIZE WEIGHT MATRICES
        self.w1 = np.random.uniform(-0.8, 0.8, (self.v_count, self.n))     # embedding matrix
        self.w2 = np.random.uniform(-0.8, 0.8, (self.n, self.v_count))     # context matrix
        
        # CYCLE THROUGH EACH EPOCH
        epoch=[]
        loss=[]
        for i in range(0, self.epochs):

            self.loss = 0

            # CYCLE THROUGH EACH TRAINING SAMPLE
            for w_t, w_c in training_data:

                # FORWARD PASS
                y_pred, h, u = self.forward_pass(w_t)
                
                # CALCULATE ERROR
                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)

                # BACKPROPAGATION
                self.backprop(EI, h, w_t)

                # CALCULATE LOSS
                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
                #self.loss += -2*np.log(len(w_c)) -np.sum([u[word.index(1)] for word in w_c]) + (len(w_c) * np.log(np.sum(np.exp(u))))
                epoch.append(i)
                loss.append(self.loss)
            print('EPOCH:',i, 'LOSS:', self.loss)
            plt.figure()
            plt.plot(epoch,loss)   
        
        
        pass


    # input a word, returns a vector (if available)
    def word_vec(self, word):
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w


    # input a vector, returns nearest word(s)
    def vec_sim(self, vec, top_n):

        # CYCLE THROUGH VOCAB
        word_sim = {}
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(vec, v_w2)
            theta_den = np.linalg.norm(vec) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda word,sim:sim, reverse=True)

        for word, sim in words_sorted[:top_n]:
            print(word, sim)
            
        pass

    # input word, returns top [n] most similar words
    def word_sim(self, word, top_n):
        
        w1_index = self.word_index[word]
        v_w1 = self.w1[w1_index]

        # CYCLE THROUGH VOCAB
        word_sim = {}
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda word,sim:sim, reverse=True)

        for word, sim in words_sorted[:top_n]:
            print(word, sim)
            
        pass

#--- EXAMPLE RUN --------------------------------------------------------------+

settings = {}
settings['n'] = 300                   # dimension of word embeddings
settings['window_size'] = 2         # context window +/- center word
settings['min_count'] = 0           # minimum word count
settings['epochs'] = 2000           # number of training epochs
settings['neg_samp'] = 10           # number of negative words to use during training
settings['learning_rate'] = 0.01    # learning rate
np.random.seed(0)                   # set the seed for reproducibility


# INITIALIZE W2V MODEL
w2v = word2vec()


# In[11]:


word='week'
w2v.word_sim(word,10)


# In[159]:


total_train_words


# In[7]:



training_data = w2v.generate_training_data(settings, train_corpus[:4000])


# In[10]:


w2v.train(training_data)


# In[12]:


#--------------------------code for calculationg cosine similarity-----------------------------#
from scipy import spatial
sim=spatial.distance.cosine


# In[13]:


#--------------------------getting all words in training corpus in a list-----------------------------#
total_train_words=[]
for i in range(0,50):
    total_train_words=total_train_words+train_corpus[i]


# In[14]:


#--------------------------extracting words from simlex-999 file -----------------------------#
words=[]
with open ("SimLex-999/SimLex-999.txt","r") as f:
    for line in f:
        words.append([line.split()[0],line.split()[1],line.split()[3]])    


# In[15]:


#----------------------Calculating similarity of words present in corpus-----------------------------#
words[0].append('cosine')
for word in words[1:]:
    if(word[0] in total_train_words and word[1] in total_train_words):
        result=1 - sim(w2v.word_vec(word[0]), w2v.word_vec(word[1]))
        word.append(result)
    else:
        word.append(0)

df = pd.DataFrame(words,columns=words.pop(0))


# In[16]:


from sklearn import preprocessing
# Create x, where x the 'scores' column's values as floats
x = df[['SimLex999']].values.astype(float)

# Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor
x_scaled = min_max_scaler.fit_transform(x)

# Run the normalizer on the dataframe
df_normalized = pd.DataFrame(x_scaled)


# In[17]:


df[['SimLex999']]=df_normalized


# In[18]:


df_score=df


# In[19]:


df_score=df_score.loc[df_score['cosine'] != 0]
df_score.corr()


# In[20]:


df_score


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


#--------------------------code to get words:their embeddings-----------------------------#
embeddings=defaultdict()
for row in train_corpus[:20]:
    for word in row:
        embeddings[word]=list(w2v.word_vec(word))


# In[22]:


embeddings.keys()


# In[262]:


#--------------------------error is coming, dont look into it noe-----------------------------#

import math
class TableForNegativeSamples:
    def __init__(self, words_list):
        power = 0.75
        norm = sum([math.pow(word_counts[word], power) for word in words_list]) # Normalizing constants

        table_size = 1e8
        table = np.zeros(table_size)

        p = 0 # Cumulative probability
        i = 0
        for j, word in enumerate(word_counts):
            p += float(math.pow(w2v.word_counts[word], power))/norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]


# In[263]:


#--------------------------error is coming, dont look into it now-----------------------------#



ng=TableForNegativeSamples(w2v.words_list)


# In[23]:


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# In[24]:


viz_words = 500
tsne = TSNE()
embed_tsne_w1 = tsne.fit_transform(w2v.w1[:viz_words, :])
embed_tsne_w2 = tsne.fit_transform(w2v.w1[:viz_words, :])


# In[25]:


fig, ax = plt.subplots(figsize=(14, 14))
for idx in range(viz_words):
    plt.scatter(*embed_tsne_w1[idx, :])
    plt.annotate(w2v.index_word[idx], (embed_tsne_w1[idx, 0], embed_tsne_w1[idx, 1]))


# In[27]:


fig, ax = plt.subplots(figsize=(14, 14))
for idx in range(viz_words):
    plt.scatter(*embed_tsne_w2[idx, :])
    plt.annotate(w2v.index_word[idx], (embed_tsne_w2[idx, 0], embed_tsne_w2[idx, 1]), alpha=0.7)


# In[70]:


import time
import tensorflow as tf
import numpy as np
import pickle


# In[138]:


word2index = w2v.word_index
index2word = w2v.index_word
VOCAB_SIZE =  w2v.v_count
avl_questions = []
not_avl_questions = 0
questions_words = open("questions-words.txt","r")
for line in questions_words:
    if line.startswith(":"):
        continue
    words = line.strip().lower().split(" ")
    ids = [word2index.get(word) for word in words]
    if None in ids or len(ids)!= 4:
        not_avl_questions += 1
    else:
        avl_questions.append(ids)
print("Questions Skipped:{}".format(not_avl_questions))
print("Number of analogy questions:{}".format(len(avl_questions)))
questions_words.close()
avl_questions = np.array(questions, dtype=np.int32)

avl_questions_=[]
for q in avl_questions:
    qs=[]
    for i in q:
        qs.append(index2word[i])
    avl_questions_.append(qs)
avl_questions_

