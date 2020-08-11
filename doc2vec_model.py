import pandas as pd
import numpy as np
import re 
import seaborn as sns 
import matplotlib.pyplot as plt 
import plotly.express as px
import datetime
from matplotlib.pyplot import *
import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec 
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn import utils
import multiprocessing
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from etl import *


def create_df(csv, trainSize):
    
    print("Preprocessing Dataset....")
    
    df = format_data(csv)
    df['Tokens'] = df['Quote'].apply(lambda x: nltk.word_tokenize(x))
    df = df.sample(frac=1)
    
    df_train = df[-int(len(df)*trainSize):]
    df_test = df[int(len(df)*trainSize):]
    
    print("Preprocessing Complete....")
    
    return df_train, df_test



def tag_values(df_train, df_test):
    
    train_tagged = df_train.apply(lambda r: TaggedDocument(words=r['Tokens'], tags=r['Label']), axis=1)
    test_tagged = df_test.apply(lambda r: TaggedDocument(words=r['Tokens'], tags=r['Label']), axis=1)
    
    print("Tagged Training and Testing Datasets....")
    
    return train_tagged, test_tagged
    
    
def vec_for_learning(model, tagged_docs):
    
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    
    return targets, regressors

def vocab_training_helper(model, train_tagged, test_tagged, epochs, alpha):
    
    model.build_vocab([x for x in tqdm(train_tagged.values)])

    for epoch in range(epochs):
        model.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
        model.alpha -= alpha
        model.min_alpha = model.alpha
        
    y_train, X_train = vec_for_learning(model, train_tagged)
    y_test, X_test = vec_for_learning(model, test_tagged)
    
    return y_train, X_train, y_test, X_test


def fit_model_logit(docModel, train_tagged, test_tagged, epochs, alpha):

    print("Fitting and Training Model....\n")
    
    cores = multiprocessing.cpu_count()
    
    if(docModel == 'model_dbow'):
        
        model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
        y_train, X_train, y_test, X_test = vocab_training_helper(model_dbow, train_tagged, test_tagged, epochs, alpha)
        print("Model: model_dbow")
    
    
    elif(docModel == 'model_dmm'):
        
        model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=10, negative=5, min_count=1, workers=cores, alpha=0.065, min_alpha=0.065)
        y_train, X_train, y_test, X_test = vocab_training_helper(model_dmm, train_tagged, test_tagged, epochs, alpha)
        print("Model: model_dmm")
        
    else:
        
        print("INPUT ERROR: @param docModel")


    logreg = LogisticRegression(n_jobs=1, C=1e5)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    
    labels = ['true', 'false']
    #cf_matrix = confusion_matrix(y_test, y_pred, labels)
    
    print('\n\nTesting accuracy %s' % accuracy_score(y_test, y_pred))
    print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
    print('Confusion_matrix\n')
    #print(cf_matrix)

    
df_train, df_test = create_df('politifact_data.csv', .8)
train_tagged, test_tagged = tag_values(df_train, df_test)
fit_model_logit('model_dmm', train_tagged, test_tagged, 30, 0.002)