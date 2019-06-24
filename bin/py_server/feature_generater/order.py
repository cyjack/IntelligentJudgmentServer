'''
Created on 2017年5月11日

@author: mage
'''
import jieba
from keras.preprocessing import sequence
from feature_generater.tf_idf import cut_words
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

def order_fea_generator(data, model=None, feature_len=10):
    '''
    generate order features of data
    :param data: list, a list of sentence with cut word
    :param model: int, length of feature
    :param save_path: str, save path of feature file
    :param save_name: str, save name of feature file
    :param feature_generator: path of feature generator
    '''
    corpus = cut_words(data)
    sequences = model.texts_to_sequences(corpus)
    sequences = sequence.pad_sequences(sequences,feature_len)
    return sequences