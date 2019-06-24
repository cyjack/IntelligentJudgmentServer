# -*- coding: utf-8 -*-
import os
import sys
import getopt
from PIL import Image
import numpy as np
import jieba
import pickle
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer
from utils.path import main_path

def cut_words(corpus):
    '''
    cut word. return a list in which contains all cut word of a sentence in each list item
    :param corpus: a list type, which contains all sentences in corpus
    '''
    res = list()
    for sentence in corpus:
        sen_cut = jieba.cut(sentence, cut_all=False)
        res.append(' '.join(sen_cut))
    return res

def train_tf_idf_model(train_corpus, save_path = None, save_name=None):
    '''
    train a feature extractor of tf-idf
    :param train_corpus: list type, each item is a sentence with cut words
    :param save_path: str type, path of saving model
    :param save_name: str type, name of saving model
    '''
    vectorizer = CountVectorizer(ngram_range = (1,1), min_df = 0.008)
    transformer = TfidfTransformer()
    tf = vectorizer.fit_transform(train_corpus)
    tfidf = transformer.fit_transform(tf)
    #a = tfidf.toarray()
    #print(a.shape)
    if save_path != None:
        pickle.dump((vectorizer,transformer),open(save_path + save_name + '.pkl','wb'))
    else:
        pickle.dump((vectorizer,transformer),open(main_path()+'/tfidf_generator.pkl','wb'))

def get_tf_idf_model(model_path = None, model_name=None):
    '''
    get model extracting tf-idf feature
    :param model_path: str type, path of model
    :param model_name: str type, name of model
    '''
    if model_path != None:
        model = pickle.load(open(model_path + model_name,'rb'))
    else:
        model = pickle.dump(open(main_path()+'/tfidf_generator.pkl','wb'))
    return model
    
def tf_idf_generator(data, model):
    '''
    generate tf-idf feature for data according to current tf_idf generator model
    :param data: list type, each item is a sentence with cut words
    :param model: object, tf_idf generator model
    '''
    data_corpus = cut_words(data)
    vectorizer, transformer = model
    test_tf = vectorizer.transform(data_corpus)
    tf_idf = transformer.transform(test_tf)
    return tf_idf.toarray()   