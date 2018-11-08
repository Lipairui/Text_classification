# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import jieba
import time
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation, metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from gensim.models.lsimodel import LsiModel
from gensim.models.ldamodel import LdaModel
from gensim.models.tfidfmodel import TfidfModel
from gensim.models import Word2Vec,KeyedVectors
from gensim import corpora
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.core import *
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout,Activation
from keras.models import model_from_yaml
from keras.models import Sequential
from keras.optimizers import *
from keras.preprocessing.sequence import pad_sequences

def LogInfo(stri):
    '''
     Funciton: 
         print log information
     Input:
         stri: string
     Output: 
         print time+string
     '''
    
    print(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+'  '+stri)

def getColName(colNum, stri):
    '''
     Funciton: 
         generate columns names
     Input: 
         colNum: number of columns
         stri: string
     Output:
         list of columns names   
    '''
    LogInfo(' '+str(colNum)+','+stri)
    colName = []
    for i in range(colNum):
        colName.append(stri + str(i))
    return colName
    
def get_tfidf_feature(documents):
    '''
     Funciton:
         generate tfidf features 
     Input:
         data: list of preprocessed sentences    
     Output:
         tfidf features(DataFrame format)
    '''   
    LogInfo('Generate TFIDF features...')
    tfidf = TfidfVectorizer()
    res = tfidf.fit_transform(documents).toarray()
    dim = len(tfidf.get_feature_names())
    colName = getColName(dim, "tfidf")
    tfidf_features = pd.DataFrame(res,columns = colName)
    return tfidf_features
    
  def getLsiFeature(documents, topicNum):
    '''
     Funciton:
         generate lsi features by training lsi model
     Input:
         documents: list of preprocessed sentences
         topicNum: output vector dimension
     Output:
         lsi features(DataFrame format)
    '''
    LogInfo('Generate LSI features...')
    # get corpus
    texts = [[word for word in document.split(' ')] for document in documents]
    dictionary = corpora.Dictionary(texts)
    corpusD = [dictionary.doc2bow(text) for text in texts]
    
    # train lsi model
    tfidf = TfidfModel(corpusD)
    corpus_tfidf = tfidf[corpusD]
    model = LsiModel(corpusD, num_topics=topicNum, chunksize=8000, extra_samples = 100)#, distributed=True)#, sample = 1e-5, iter = 10,seed = 1)

    # generate lsi features
    LogInfo('Generate LSI features...')
    lsiFeature = np.zeros((len(texts), topicNum))
    i = 0
    for doc in corpusD:
        topic = model[doc]
        for t in topic:
             lsiFeature[i, t[0]] = round(t[1],5)
        i = i + 1
    colName = getColName(topicNum, "qlsi")
    lsiFeature = pd.DataFrame(lsiFeature, columns = colName)
    return lsiFeature
  
 def getLdaFeature(documents, topicNum):
    '''
     Funciton:
         generate lda features by training lda model
     Input:
         documents: list of preprocessed sentences
         topicNum: output vector dimension
     Output:
         lda features(DataFrame format)
    '''
    LogInfo('Generate LDA features...')
    # get corpus
    texts = [[word for word in document.split(' ')] for document in documents]
    dictionary = corpora.Dictionary(texts)    
    corpusD = [dictionary.doc2bow(text) for text in texts]

    # train lda model
    tfidf = TfidfModel(corpusD)
    corpus_tfidf = tfidf[corpusD]
#     ldaModel = gensim.models.ldamulticore.LdaMulticore(corpus_tfidf, workers = 8, num_topics=topicNum, chunksize=8000, passes=10, random_state = 12)
    ldaModel = LdaModel(corpus_tfidf, num_topics=topicNum, chunksize=8000, passes=10, random_state = 12)
    # generate lda features
    ldaFeature = np.zeros((len(texts), topicNum))
    i = 0
    for doc in corpus_tfidf:
        topic = ldaModel.get_document_topics(doc, minimum_probability = 0.01)
        for t in topic:
             ldaFeature[i, t[0]] = round(t[1],5)
        i = i + 1
    colName = getColName(topicNum, "qlda")
    ldaFeature = pd.DataFrame(ldaFeature, columns = colName)
    return ldaFeature
    
def getWord2VecFeature(documents):
    '''
    Function:
        compute the mean of word vectors
    Input:
        documents: list of preprocessed sentences
    Output:
        doc vector 
    '''
    LogInfo('Generate Word2Vec features...')
    model_path = '../model/cn.cbow.bin'
    # load word2vec model  
    LogInfo('Load word2vec model...')
    model = KeyedVectors.load_word2vec_format(model_path,binary=True,unicode_errors='ignore')
    # normalize vectors
    model.init_sims(replace=True)
    # remove out-of-vocab words
    word2vecFeature = np.zeros((len(documents),300))
    errors = 0
    for i,doc in enumerate(documents):   
        doc = [word for word in doc.split(' ') if word in model.vocab]
        if len(doc)!=0:
            word2vecFeature[i,:] = np.mean(model[doc],axis=0)
        else:
            errors += 1
    LogInfo('Errors: %d'%errors)
    colName = getColName(300, "w2v")
    word2vecFeature = pd.DataFrame(word2vecFeature, columns = colName)    
    return word2vecFeature
    
def getKeyWordFeature(documents,corpus_path):
    '''
    Function:
        compute keyword frequency in every document
    Input:
        documents: list of preprocessed sentences
        corpus_path: path of keyword corpus file(one word per line)
    Output:
        keyword frequency feature (DataFrame format)    
    '''
    LogInfo('Generate key word features...')
    corpus = [w.strip() for w in codecs.open(corpus_path, 'r',encoding='utf-8').readlines()]
    feature = []
    for i,doc in enumerate(documents):
        key_words_num = sum([1 for word in doc.split(' ') if word in corpus])     
        feature.append(key_words_num)     
    feature = pd.DataFrame(feature,columns=['key_words_num'])
    return feature
    
def getLstmFeature(X_train,y_train,X_test):
    '''
    Function:
        geneate LSTM 2-classification probability features by cross validation
    Input:
        X_train: list of preprocessed sentences of traning data
        y_train: list of labels of training data
        X_test:  list of preprocessed sentences of testing data
    Output:
        lstm_train: LSTM 2-classification probability features of training data (DataFrame format)
        lstm_test:  LSTM 2-classification probability features of testing data (DataFrame format)
     '''
    LogInfo('Generate LSTM featrues...')
    # parameters
    max_length = 40
    dim = 300
    batch_size = 32
    n_epoch = 20
    # load word2vec model  
    LogInfo('Load word2vec model...')
    model_path = '../model/cn.cbow.bin'   
    model = KeyedVectors.load_word2vec_format(model_path,binary=True,unicode_errors='ignore')
    # Get vocab
    documents = X_train+X_test
    documents = [doc.split(' ') for doc in documents]
    # filter if not in word2vec
    vocab = []
    docs = []
    for doc in documents:
        s = []
        for word in doc:     
            if word in model.vocab:
                s.append(word)
                vocab.append(word)
        docs.append(s)   
    vocab = set(vocab)

    # Encode documents
    wordindex = dict()
    embedding_matrix = np.zeros((len(vocab),dim))
    for i, word in enumerate(vocab):
        wordindex[word] = i
        embedding_matrix[i] = model[word]
    encoded_docs = []
    for doc in docs:
        encoded_doc = []
        for word in doc:
            encoded_doc.append(wordindex[word])
        encoded_docs.append(encoded_doc)

    # Pad sentences
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    p_train = padded_docs[:len(y_train)]
    p_test = padded_docs[len(y_train):]

    # lstm model structure
    model = Sequential()
    embedding = Embedding(
        input_dim=len(vocab), 
        output_dim=dim, 
        mask_zero=True,
        weights=[embedding_matrix], 
        input_length=max_length,
        trainable=False)
    model.add(embedding)
    model.add(LSTM(units=50, activation='sigmoid', recurrent_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    # Compile and train the model
    LogInfo('Compiling the Model...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])

    LogInfo("Train and predict...")
    skf = StratifiedKFold(y_train, n_folds=3, shuffle=True)
    new_train = np.zeros((len(p_train),1))
    new_test = np.zeros((len(p_test),1))
    for i,(trainid,valid) in enumerate(skf):
        print('fold' + str(i))
        train_x = p_train[trainid]
        train_y = y_train[trainid]
        val_x = p_train[valid]
        model.fit(train_x, train_y,
              batch_size=batch_size,
              epochs=n_epoch,verbose=1)
        new_train[valid] = model.predict_proba(val_x)
        new_test += model.predict_proba(p_test)

    new_test /= 3
    stacks = []
    stacks_name = []
    stack = np.vstack([new_train,new_test])
    stacks.append(stack)
    stacks = np.hstack(stacks)
    clf_stacks = pd.DataFrame(data=stacks,columns=['lstm'])
    lstm_train = clf_stacks.iloc[:len(X_train)]
    lstm_test = clf_stacks.iloc[len(X_train):].reset_index(drop=True)
    return lstm_train, lstm_test    

  
  
  
  
  
  
  
  
  
  
  
