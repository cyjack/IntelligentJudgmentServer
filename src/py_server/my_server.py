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
from feature_generater.tf_idf import tf_idf_generator
from feature_generater import *
from feature_generater.tf_idf import response_q
from feature_generater.att_grading_model import *
from feature_generater.order import order_fea_generator
from keras.models import load_model
from utils.my_dict import MyDict
from utils.path import main_path
from filterNoise import judgeOrNot

# from keras.backend.cntk_backend import one_hot


class MyHandler:
    
    def __init__(self):
        """
        constructor
        """
        self.i = 228
#         self.model = MyDict() # key is question id, value is model,
#         #self.model_path = '/Volumes/E/workspace/IntelligentJudgmentModel/src/com/aic/pij/model/'
#         self.model_path = main_path().rsplit('/',1)[0]+'/model/'
#         model_name = self.eachFile(self.model_path)
#         print('Model loading ...')
#         for ele in model_name:
#             #ele looks like '2017_01_qm_11_1.svm.cf.m'
#             print('  ', ele)
#             ele_list = ele.split('.')
#             question_model = self.load_model(self.model_path+ele, ele_list[-1])
#             self.model.add(key     = ele_list[0],
#                            sub_key = ele_list[1],
#                            value   = [question_model, ele_list[2], ele_list[-1]])
        #加载词
        
        self.vocab_002 = get_vovab('math_002')
        self.vocab_004 = get_vovab('math_004')
        self.vocab_001 = get_vovab('math_001')
        self.vocab_003 = get_vovab('math_003')
        #load vocab 
        
        self.sess_002 = get_sess('math_002') 
        self.sess_004 = get_sess('math_004')
        self.sess_001 = get_sess('math_001')
        self.sess_003 = get_sess('math_003')
        #load sesss
           
          
        #加载词
       

        print('Model loading completed ...')
    def fileout(self,string):
        fout = open("question"+'.txt','a',encoding ='utf-8')
        fout.write(string)
        fout.close()
    def judgment(self, question_id, question_content):
        '''
        Client calls the model cumpute and get a result from this funtion
        :param question_id: ID of question in paper
        :param question_content: answer of question ID
        ''' 
        # generator tf-idf feature
        # adfd$dhdha$kdkajh
        # content = question_content.split('$')
#         tfidf_X = tf_idf_generator(data=[question_content], # list() method can't be used to convert question_content to a list
#                                    model = self.model.get(question_id, 'tfidf')[0])
#         order_X = order_fea_generator(data=[question_content], 
#                                       model= self.model.get(question_id, 'order')[0], 
#                                       feature_len=78)
        # Note: the returned type must be the predefined type, this problem cost my a half day 
#         if question_id not in self.model.keys():
#             self.fileout(question_id+","+question_content+'\n')
#             res = question_id+question_content
#             return res
        
        if question_id == '2017_01_qm_luwang':
#             sub_models = self.model.get_sub_item(question_id)
#             res = 0
#             for key in sub_models.keys():
#                 if sub_models.get(key)[2] == 'h5':
#                     res += sub_models.get(key)[0].predict(tfidf_X)
#                 elif sub_models.get(key)[2] == 'm':
#                     res += sub_models.get(key)[0].predict_proba(tfidf_X)
            tfidf_X = tf_idf_generator(data=[question_content], # list() method can't be used to convert question_content to a list
                                   model = self.model.get(question_id, 'tfidf')[0])
            order_X = order_fea_generator(data=[question_content], 
                                      model= self.model.get(question_id, 'order')[0], 
                                      feature_len=78)
            rf     = self.model.get(question_id, 'rf')[0]
            logreg = self.model.get(question_id, 'logreg')[0]
            lstm   = self.model.get(question_id, 'lstm')[0]
            res = rf.predict_proba(tfidf_X) + logreg.predict_proba(tfidf_X) + lstm.predict(order_X)
            res = np.argmax(res)
           
        elif question_id == '2017_11_21_ph_2': 
            svm = self.model.get(question_id, 'svm')[0]
            res = svm.predict(onehot_X)
            res = response_q(question_id,str(res[0]))
                
        elif question_id == '2017_11_25_ph_1':
            if(question_content.find(',')):
                temp = judgeOrNot('2017_11_25_ph_1',question_content)
                if(temp!=''):
                    if(temp !=True):
                        res = lw_vector_pro(question_content)
                        self.i= self.i+1
                        self.fileout(str(self.i)+","+question_content+','+str(res[0])+'\n')
                        res = response_q(question_id,str(res[0]),str(self.i))
                    else:
                        res = response_q(question_id,'3',str(self.i))
                else:
                    res = response_q(question_id,'0',str(self.i))
            else:
                self.fileout(x+'\n')
                
        elif question_id == '2017_11_25_ph_3':
            if(question_content.find(',')):
                temp = judgeOrNot('2017_11_25_ph_3',question_content)
                if(temp == True):
                    self.i= self.i+1
                    res = response_q(question_id,'3',str(self.i))
                else:
                    res = bigram_vector_pro(question_content)
                    res = response_q(question_id,str(res[0]),str(self.i))
            else:
                self.fileout(x+'\n')
        elif question_id == 'math_002':
#             print(question_content)
              #加载模型
             
            res = mpredict_score(self.sess_002,self.vocab_002,question_id,question_content)
            
            print(res)
            res = response_q(question_id,str(res),str(self.i))
            
        elif question_id == 'math_004':
            print(question_content)
            #加载模型
                     
            res = mpredict_score(self.sess_004,self.vocab_004,question_id,question_content)
            
            print(res)
            res = response_q(question_id,str(res),str(self.i))
        elif question_id == 'math_001':
            print(question_content)
            #加载模型
                     
            res = mpredict_score(self.sess_001,self.vocab_001,question_id,question_content)
            
            print(res)
            res = response_q(question_id,str(res),str(self.i))
        elif question_id == 'math_003':
            print(question_content)
            #加载模型
                     
            res = mpredict_score(self.sess_003,self.vocab_003,question_id,question_content)
            
            print(res)
            res = response_q(question_id,str(res),str(self.i))
            
                # i record which was we need
#             else:
#                 svm = self.model.get(question_id, 'svm')[0]
#                 self.fileout(x+'\n')
                
#         data,stuRate = changeData(list_h)
#         scoreList,knowledgeList = calculateScore(data,stuRate)
#         writeHTML(data,scoreList,knowledgeList,stuRate)
#         print("hahahaha")
#         list_h=[]

               
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
    q_content = '同旁内角互补，两直线平行；两直线平行，同旁内角互补';
    q_id = "math_002";
    
    print(myHandler.judgment(q_id, q_content))

#     myHandler=MyHandler()
#     processor = IntelligentJudgment.Processor(myHandler)
#     transport = TSocket.TServerSocket("172.18.136.139", 8989)
#     #transport way，using buffer
#     tfactory = TTransport.TBufferedTransportFactory()
#     #data type of transport：binary
#     pfactory = TBinaryProtocol.TBinaryProtocolFactory()
#     #create a thrift service
#     server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
#     print('Starting thrift server in python...') 
#     server.serve()