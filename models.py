# -*- coding: utf-8 -*-
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
    
def getLstmFeature(X_train,X_test,y_train):

  
  
  
  
  
  
  
  
  
  
  
