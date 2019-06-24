# -*- coding: utf-8 -*-
import os
import sys
import getopt
from PIL import Image
import numpy as np
import pickle
import IntelligentJudgment
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from sklearn.externals import joblib
from sklearn.feature_extraction.tests.test_text import test_tf_idf_smoothing
from feature_generater.tf_idf import tf_idf_generator
from feature_generater.order import order_fea_generator
from keras.models import load_model
from utils.my_dict import MyDict
from utils.path import main_path

class MyHandler:
    
    def __init__(self):
        """
        constructor
        """
        self.model = MyDict() # key is question id, value is model
        #self.model_path = '/Volumes/E/workspace/IntelligentJudgmentModel/src/com/aic/pij/model/'
        self.model_path = main_path().rsplit('/',1)[0]+'/model/'
        model_name = self.eachFile(self.model_path)
        print('Model loading ...')
        for ele in model_name:
            #ele looks like '2017_01_qm_11_1.svm.cf.m'
            print('  ', ele)
            ele_list = ele.split('.')
            question_model = self.load_model(self.model_path+ele, ele_list[-1])
            self.model.add(key     = ele_list[0],
                           sub_key = ele_list[1],
                           value   = [question_model, ele_list[2], ele_list[-1]])
        print('Model loading completed ...')
              
    def judgment(self, question_id, question_content):
        '''
        Client calls the model cumpute and get a result from this funtion
        :param question_id: ID of question in paper
        :param question_content: answer of question ID
        ''' 
        # generator tf-idf feature
        # adfd$dhdha$kdkajh
        # content = question_content.split('$')
        tfidf_X = tf_idf_generator(data=[question_content], # list() method can't be used to convert question_content to a list
                                   model = self.model.get(question_id, 'tfidf')[0])
        
        order_X = order_fea_generator(data=[question_content], 
                                      model= self.model.get(question_id, 'order')[0], 
                                      feature_len=78)

        # Note: the returned type must be the predefined type, this problem cost my a half day 
        if question_id not in self.model.keys():
            res = question_id+question_content
            return res
        elif question_id == '2017_01_qm_luwang':
#             sub_models = self.model.get_sub_item(question_id)
#             res = 0
#             for key in sub_models.keys():
#                 if sub_models.get(key)[2] == 'h5':
#                     res += sub_models.get(key)[0].predict(tfidf_X)
#                 elif sub_models.get(key)[2] == 'm':
#                     res += sub_models.get(key)[0].predict_proba(tfidf_X)
            rf     = self.model.get(question_id, 'rf')[0]
            logreg = self.model.get(question_id, 'logreg')[0]
            lstm   = self.model.get(question_id, 'lstm')[0]
            res = rf.predict_proba(tfidf_X) + logreg.predict_proba(tfidf_X) + lstm.predict(order_X)
            res = np.argmax(res)
        else:
            svm = self.model.get(question_id, 'svm')[0]
            res = svm(tfidf_X)
        print(res)
        return str(res)
    
    def load_model(self, model, model_type):
        """
        load the model according to model path 'model'
        :param model: model to be loaded
        :param model_type: the type of model. optional value 'm','h5',‘pkl’
        """
        if model_type == 'm': 
            m = joblib.load(model)
            return m
        elif model_type == 'h5':
            h5 = load_model(model)
            return h5
        elif model_type == 'pkl':
            f = open(model, 'rb')
            pkl = pickle.load(f)
            f.close()
            return pkl

    def eachFile(self, file_path):
        """
        traverse all files in path 'file_path'
        :param file_path: str type, path of traverse a directory
        """
        pathDir =  os.listdir(file_path)
        file_names = list()
        for allDir in pathDir:
            child = os.path.join('%s%s' % (file_path, allDir))
            file_names.append(str(child).split('/')[-1])
        if '.DS_Store' in file_names:
            file_names.remove('.DS_Store')
        return file_names
        
if __name__ == '__main__':
#     myHandler=MyHandler()
#     answer = ["因为小鸟不喜欢吃肉喝酒。",
#               "鲁王喝醉了",
#             "因为他没有按照小鸟的生活安排。",
#             "人们有自己的规律，而鸟也有鸟的规律。",
#             "因为鲁国国王没有符合自然的规律，鸟类应该让它栖息在深山老林，在陆地沙洲游玩，在江河湖海飞翔。"];
#     q_id = "2017_01_qm_luwang";
#     for i in np.arange(len(answer)):
#         q_content = answer[i];
#         myHandler.judgment(q_id, q_content)

    myHandler=MyHandler()
    processor = IntelligentJudgment.Processor(myHandler)
    transport = TSocket.TServerSocket("127.0.0.1", 8989)
    #transport way，using buffer
    tfactory = TTransport.TBufferedTransportFactory()
    #data type of transport：binary
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    #create a thrift service
    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
    print('Starting thrift server in python...')
    server.serve()