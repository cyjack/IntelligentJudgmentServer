# -*- coding: utf-8 -*-
import os
import sys
import getopt
from PIL import Image
import numpy as np
import jieba
import json
import pickle
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer
# from utils.path import main_path
import tensorflow as tf
import logging
import time
import numpy as np
import os
import json
import tensorflow.contrib as tc
import xlrd
import jieba
import pickle
from sklearn.metrics import classification_report,confusion_matrix
from collections import Counter
import os
import sys
import getopt
# from PIL import Image
import numpy as np
import jieba
import pickle
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.word2vec import Word2Vec
# from _license import join
import json
from sklearn.externals import joblib
def DeleteStopWords(data, stopWords):
    wordList = []

    # 先分一下词
    cutWords = jieba.cut(data)
    for item in cutWords:
        if item.encode('utf-8') not in stopWords:  # 分词编码要和停用词编码一致
            wordList.append(item)
            # print(item)
    return wordList
#############################读取json文件##########################################
def read_json(name):
    xx = ""
    x = []
    y = []

    with open(name, 'r', encoding="utf-8") as f:
        data = json.load(f)
        for item in range(len(data['RECORDS'])):
            word = DeleteStopWords(data['RECORDS'][item]['answer_1'], ['，'])
            # print(word)
            xx = ' '.join(word)

            # if ('没' in xx or '不' in xx):
            #     continue
            # else:
            x.append(xx)
            y.append(data['RECORDS'][item]['score_1'])
        x, y = d_repeat(x, y)
        # key_word = ['光合作用', '淀粉']
        # y_number = self.keyword_number_feature(key_word, x)
        # for i in range(len(y_number)):
        #     if (y_number[i] == 0):
        #         continue
        #     else:
        #         x_1.append(x[i])
        #         y_1.append(y[i])
        # 去掉不含有关键词的数据
        return x, y


#############################去除重复数据##########################################
def d_repeat( x, y):
    x_1 = []
    y_1 = []
    for j in range(len(x)):
        if (x[j] not in x_1):
            x_1.append(x[j]) 
            y_1.append(y[j])
            #            print(x_1)
            #     print(len(x_1))
    return x_1, y_1
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
def get_one_hot_model(model_path = None, model_name=None):
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
def one_hot_generator(data, model):
    '''
    generate tf-idf feature for data according to current tf_idf generator model
    :param data: list type, each item is a sentence with cut words
    :param model: object, tf_idf generator model
    '''
    data_corpus = cut_words(data)
    vectorizer = model
    onehot = vectorizer.transform(data_corpus)
    return onehot.toarray()
def get_W2V_model():
    '''
    get model extracting W2V feature
    :param model_path: str type, path of model
    :param model_name: str type, name of model
    '''
    model = Word2Vec.load('/root/下载/IntelligentJudgmentServer/src/py_server/feature_generater/vector/sougou_tune.model')
    return model
def W2V_sentence(x,model):
        x = x.split()
        i=0
        j=1
        sentence =np.full((1,400),0)
        for item in x :
            try:
             sentence =sentence+model.wv[item]
#              *j/len(x)
             i=i+1
             j = j + 1
            except KeyError:
                i = i - 1
                j = j + 1
                continue
        x = sentence.tolist()
        if(i==0):
            i=1
        sentence_list=[]

        for item in x[0]:
            if(item !=0):
                item =(item/i)

            sentence_list.append(item)
        # print(sentence_list)
        return sentence_list
def Dcut_Words(data):

    wordList = []

    # 先分一下词
    cutWords = jieba.cut(data)
    for item in cutWords: # 分词编码要和停用词编码一致
            wordList.append(item)
            # print(item)
    word = " ".join(wordList)
    return word
def W2V_generator(data, model):
    '''
    generate tf-idf feature for data according to current tf_idf generator model
    :param data: list type, each item is a sentence with cut words
    :param model: object, tf_idf generator model
    '''
    data = Dcut_Words(data)
    print(data)
    vectorizer = model
    sentence = W2V_sentence(data,vectorizer)
    return sentence
def normal_leven(str1, str2):
        len_str1 = len(str1) + 1
        len_str2 = len(str2) + 1
        # create matrix
        matrix = [0 for n in range(len_str1 * len_str2)]
        # init x axis
        for i in range(len_str1):
            matrix[i] = i
        # init y axis
        for j in range(0, len(matrix), len_str1):
            if j % len_str1 == 0:
                matrix[j] = j // len_str1

        for i in range(1, len_str1):
            for j in range(1, len_str2):
                if str1[i - 1] == str2[j - 1]:
                    cost = 0
                else:
                    cost = 1
                matrix[j * len_str1 + i] = min(matrix[(j - 1) * len_str1 + i] + 1,
                                               matrix[j * len_str1 + (i - 1)] + 1,
                                               matrix[(j - 1) * len_str1 + (i - 1)] + cost)

        return matrix[-1]

def lw_generator(s1):
    # s1 = '淀粉 产生 光合作用'
    x, z = read_json('/root/下载/IntelligentJudgmentServer/src/py_server/feature_generater/json/love_YX_1_v2.json')
    temp = []
    for i in x:
        a = normal_leven(s1, i)
        temp.append(a)
    return temp
def lw_vector_pro(data):
    s1 =lw_generator(data)
    s2 =W2V_generator(data,get_W2V_model())
    clf_lw = joblib.load('/root/下载/IntelligentJudgmentServer/src/model/2017_11_25_ph_1_lw.svm.cf.m')#load模型lw
    clf_w2v = joblib.load('/root/下载/IntelligentJudgmentServer/src/model/2017_11_25_ph_1_vector.svm.cf.m')#load模型w2v
    clf_all = joblib.load('/root/下载/IntelligentJudgmentServer/src/model/2017_11_25_ph_1.svm.cf.m')#load模型拼接后的
    w2v_p = clf_w2v.predict_proba(s2)
#     print(w2v_p)
    lw_p = clf_lw.predict_proba(s1)
    a = lw_p.tolist()[0]
    a[0:0] = w2v_p.tolist()[0]
#     print(a)
    return clf_all.predict(a)

def bigram_generate(x):
    x = Dcut_Words(x)
    vectorizer = joblib.load('/root/下载/IntelligentJudgmentServer/src/model/2017_11_25_ph_3.bigram.fes.pkl')
    list_dql= []
    list_dql.append(x)
    list_1 = vectorizer.transform(list_dql)
     
#     print(list_1.toarray().tolist())
    
    return list_1.toarray().tolist()

def bigram_vector_pro(question_content):
    s1 = bigram_generate(question_content)
    s2 = W2V_generator(question_content, get_W2V_model())
    clf_bigram = joblib.load('/root/下载/IntelligentJudgmentServer/src/model/2017_11_25_ph_3.svm.bigram.cf.m')
    #
    clf_vector = joblib.load('/root/下载/IntelligentJudgmentServer/src/model/2017_11_25_ph_3.vector.svm.cf.m')
    clf_all = joblib.load('/root/下载/IntelligentJudgmentServer/src/model/2017_11_25_ph_3.svm.cf.m')
    joblib
    print(s1)
     
    bigram_p = clf_bigram.predict_proba(s1[0])

   
    vector_p = clf_vector.predict_proba(s2)
#     print(s2)
#     print(  vector_p )
#     print(  bigram_p )
    a = bigram_p.tolist()[0]
    a[0:0] = vector_p.tolist()[0]
    print(a)  
#     print(clf_all.predict_proba(a))
     
    return clf_all.predict(a)
     
     
    pass
def response_q(question_id,class_s,user_id):
    dignosis=' '
    score=0
    if(question_id == '2017_11_25_ph_1'):
        if (class_s == '0'):
            score = 0
            dignosis = '你的回答不正确。本题解答要点有两点：(1) ' \
                       '番茄苗在阳光下放置一段时间，可以进行光合作用；' \
                       '(2)对番茄叶片进行题干中的实验处理后，叶片变蓝，根' \
                       '据淀粉遇碘变蓝，说明该叶片中含有淀粉；综合两个要点，' \
                       '说明番茄苗在光下进行光合作用产生淀粉。请回顾光合作用的原' \
                       '料与产物相关知识点。'
        elif (class_s == '1'):
            score = 1
            dignosis = '你的回答不完整，本题解答要点有两点：(1) 番茄苗在阳' \
                       '光下放置一段时间，可以进行光合作用；(2)对番茄叶片进行题' \
                       '干中的实验处理后，叶片变蓝，根据淀粉遇碘变蓝，说明该叶片中' \
                       '含有淀粉；综合两个要点，说明番茄苗在光下进行光合作用产生淀粉' \
                       '。请检查你的答案是否缺少要点（2）或者描述是否准确，继续加油哦!。'
        elif (class_s == '2'):
            score = 1
            dignosis = '你的回答不完整，本题解答要点有两点：(1) 番茄苗在阳光' \
                       '下放置一段时间，可以进行光合作用；(2)对番茄叶片进行题干中' \
                       '的实验处理后，叶片变蓝，根据淀粉遇碘变蓝，说明该叶片中含有淀粉' \
                       '；综合两个要点，说明番茄苗在光下进行光合作用产生淀粉。请检查你的答' \
                       '案是否缺少要点（1）或者描述是否准确，继续加油哦!'
        elif (class_s == '3'):
            score = 2
            dignosis = '恭喜你，回答完全正确，很棒！' \
                       '本题是一道实验题，需要根据实验现象得出结论' \
                       '。解答此类题目，应注意以下两点：(1) 明确实验目' \
                       '的；(2) 正确理解实验步骤，根据题干中的实验现象，' \
                       '推理正确结论。'

        elif (class_s == '4'):
            score = 0
            dignosis = '你的回答不正确，你对实验的目的理解有误。' \
                       '本次实验是为了验证“光合作用的产物是淀粉”，' \
                       '而非为了验证“光合作用的条件、原料或场所”。下' \
                       '次审题时请仔细正确理解实验目的。解答此类题目，应注' \
                       '意以下两点：(1) 明确实验目的；(2) 正确理解实验步骤，' \
                       '根据题干中的实验现象，推理正确结论。'
        gold = '光合作用可以产生淀粉'

    if(question_id == '2017_11_21_ph_2'):
        if(class_s == '0'):
            score =0
            dignosis = '你的回答不正确。回答本题可以从以下两点入手' \
                       '：(1)阐述生物学原理：在一定范围内，玻璃罩内二氧' \
                       '化碳的浓度越高，番茄光合作用产生的有机物越多，长势' \
                       '就越好。(2)结合题干信息，分析推理得出结论：甲-1装置' \
                       '中的碳酸氢钠溶液相较于其他装置能提高（增加）环境中二' \
                       '氧化碳的浓度，所以甲-1装置中番茄长势最好。'
        elif(class_s == '3'):
            score = 2
            dignosis ='你的回答不完整，你未答出在一定范围内，' \
                      '玻璃罩内二氧化碳的浓度越高，番茄光合作用产' \
                      '生的有机物越多，长势就越好。请回顾知识点光合' \
                      '作用的原料与产物。'
        elif(class_s == '4'):
            score = 3
            dignosis = '恭喜你，回答完全正确，很棒！解答此类题目时，' \
                       '应注意从以下几点进行阐述：(1) 阐述生物学原理：在' \
                       '一定范围内，玻璃罩内二氧化碳的浓度越高，番茄光合作用' \
                       '产生的有机物越多，长势就越好。(2) 结合题干信息，分析推理得' \
                       '出结论：甲-1装置中的碳酸氢钠溶液相较于其他装置能提高（增加）' \
                       '环境中二氧化碳的浓度，所以甲-1装置中番茄长势最好。'

        gold ='在一定范围内，玻璃罩内二氧化碳的浓度越高，番茄光合' \
              '作用产生的有机物越多，长势就越好；甲-1装置中的碳酸氢钠' \
              '溶液相较于其他装置能提高（增加）环境中二氧化碳的浓度，所以' \
              '甲-1装置中番茄长势最好。'

    if (question_id == '2017_11_25_ph_3'):
            if (class_s == '0'):

                score = 0
                dignosis = '你的回答不正确。回答本题可以从以下三点' \
                           '入手：(1)正确解读曲线，获取直观信息：从图中' \
                           '的曲线变化可以看出，A-B段二氧化碳浓度逐渐降低，B点' \
                           '时玻璃罩内二氧化碳浓度最低；(2)根据直观信息，推理相' \
                           '关的隐含信息：说明B点时光合作用吸收的二氧化碳最多；(3)' \
                           '利用所学生物知识，进行科学解释：光合作用吸收二氧化碳合成有' \
                           '机物；综合三个要点，故B点时积累有机物最多。请回顾光合作用的' \
                           '原料与产物，外界条件对光合作用的影响知识点。'
            elif (class_s == '1'):
                score = 1
                dignosis = '你的回答不完整，缺少以下要点中的一个，本题解答要点有三点：(1)正确解读曲线，获取直观信息：从图中' \
                           '的曲线变化可以看出，A-B段二氧化碳浓度逐渐降低，B点时玻璃罩内二氧' \
                           '化碳浓度最低；(2)根据直观信息，推理相关的隐含信息：说明B点时光合' \
                           '作用吸收的二氧化碳最多；(3)利用所学生物知识，进行科学解释：光合' \
                           '作用吸收二氧化碳合成有机物；综合三个要点，故B点时积累有机物最多。'
            elif (class_s == '2'):
                score = 1
                dignosis ='你的回答不完整，缺少以下要点中的两个。本题解答要点有三点：(1)正确解' \
                           '读曲线，获取直观信息：从图中的曲线变化可以看出，A-B段二氧化碳浓' \
                           '度逐渐降低，B点时玻璃罩内二氧化碳浓度最低；(2)根据直观信息，推理' \
                           '相关的隐含信息：说明B点时光合作用吸收的二氧化碳最多；(3)利用所学' \
                           '生物知识，进行科学解释：光合作用吸收二氧化碳合成有机物；综合三个' \
                           '要点，故B点时积累有机物最多。'
            elif (class_s == '3'):
                score = 3
                dignosis = '恭喜你，回答完全正确，很棒！解答此类题目时，应' \
                           '注意从以下几点进行阐述：(1)正确解读曲线，获取直观' \
                           '信息；(2)根据直观信息，推理相关的隐含信息；(3)利用' \
                           '所学生物知识，进行科学解释。'
            
            gold = '从图中的曲线变化可以看出，A-B段二氧化碳浓度逐渐降低，' \
                   'B点时玻璃罩内二氧化碳浓度最低，说明此时光合作用吸收的二氧' \
                   '化碳最多，光合作用吸收二氧化碳合成有机物，故此时积累有机物最多。'
    if (question_id == 'math_001'):
        if (class_s == '0'):
                score=0
                dignosis='你的答案中包含与正确答案矛盾的部分，请注意审题。'
        elif (class_s == '1' or class_s == '2'):
                score = 1.5
                dignosis = '你的答案到部分正确，但距离正确答案还有一定的差距，请仔细审题。'
        elif (class_s == '3'):
                score = 2
                dignosis = '回答的非常不错，再接再厉。'

            
        gold = '四条边相等的四边形是菱形；菱形的对角线互相垂直平分..'
        str_h=json_generate_h(question_id,score,gold,dignosis,user_id)
        
    if (question_id == 'math_002'):
        if (class_s == '0'):
                score=0
                dignosis='你的答案中包含与正确答案矛盾的部分，请注意审题。'
        elif (class_s == '1'):
                score = 1
                dignosis = '你的答案到部分正确，但距离正确答案还有一定的差距，请仔细审题。'
        elif (class_s == '2'):
                score = 2
                dignosis = '回答的非常不错，再接再厉。'

            
        gold = '老师可以在155~165的身高范围内挑选队员.因为在此范围内，人数最为集中，且大家的身高相对接近.'
        str_h=json_generate_h(question_id,score,gold,dignosis,user_id)
    if (question_id == 'math_003'):
        if (class_s == '0'):
                score=0
                dignosis='你的答案中包含与正确答案矛盾的部分，请注意审题。'
        elif (class_s == '1' or class_s == '2'):
                score = 1.5
                dignosis = '你的答案到部分正确，但距离正确答案还有一定的差距，请仔细审题。'
        elif (class_s == '3'):
                score = 2
                dignosis = '回答的非常不错，再接再厉。'

            
        gold = '同旁内角互补，两直线平行；两直线平行，同旁内角互补'
        str_h=json_generate_h(question_id,score,gold,dignosis,user_id)

    if (question_id == 'math_004'):
        if (class_s == '0'):
                score = 0
                dignosis = '你的答案中关键概念错误'
        elif (class_s == '1'):
                score = 1
                dignosis = '你的答案到部分正确，但距离正确答案还有一定的差距，请仔细审题。'
        elif (class_s == '2'):
                score = 2
                dignosis = '回答的非常不错，再接再厉。'
            
        gold = '两点之间线段最短；直线外一点到这条直线上所有点连结的线段中，垂线段最短.(或垂线段最短)'

        str_h=json_generate_h(question_id,score,gold,dignosis,user_id)

    return str_h


def json_generate_h(question_id,score,gold,dignosis, user_id):
    str1 = '{"answer_id":"' + str(question_id) + '",' + \
          '"score":"' + str(score) + '",' + \
          '"gold":"' + str(gold) + '",' + \
          '"dignosis":"' + str(dignosis) + '",' + \
         '"userid":"' + str( user_id)  + '"}'
    return str1 

if __name__ == '__main__':
    for i in range(20):
       s = predict_score("math_002","选择155-165，人数集中。")
       print (s) 