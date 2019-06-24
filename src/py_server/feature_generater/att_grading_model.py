# -*- coding:utf-8 -*-
import tensorflow as tf
import jieba
import os
import sys
import pickle,re
from tensorflow.python.platform import gfile
import gc
sys.path.append("..")
#Get path from high level file
from utils.path import main_path
def getWord_Tokens(sentence):
    """
    :param sentence:需要进行分词的句子
    :return: 返回以空格隔开的string类型
    """
    jieba.load_userdict(main_path()+'/data/jiebaDict.txt')
    print ("我执行了这个加载")
    answer = sentence
    answer = answer.replace('cm', '')
    answer = answer.replace('米', '')
    answer = answer.replace('x', '')
    answer = answer.replace('≤', '-')
    answer = answer.replace('!', '-')
    answer = answer.replace('`', '-')
    answer = re.sub('[≤|<].*?[≤|<]', '-', answer)
    answer = answer.replace('155~160~165', '155-165')
    answer = answer.replace('150-155-160', '150-160')
    answer = answer.replace('~', '到')
    answer = answer.replace('-', '到')
    seg = jieba.cut(answer)

    seg = jieba.cut(sentence)
   
    return " ".join(seg).split(" ")
def get_sentence_index(sentence,vocab):
    """
    将序列转化为id
    :param sentence:
    :param vocab:
    :return:
    """
    print(getWord_Tokens(sentence))
    pad_id = vocab.get_id(vocab.pad_token)
    L = vocab.convert_to_ids(getWord_Tokens(sentence))
    L = (L+ [pad_id] * (60 - len(L)))[: 60]
    return [L]
def get_vovab(question_id):
    """

    :param question_id:
    :return:
    """
#     with open(os.path.join("G:/工作任务2019/合作相关/合作成果/IntelligentJudgmentServer/src/py_server/feature_generater/vocab", str(question_id) + "/", 'vocab.data'), 'rb') as fin:
#         vocab = pickle.load(fin)
   
    with open(main_path()+"/vocab/"+str(question_id) + "/"+'vocab.data', 'rb') as fin:
        vocab = pickle.load(fin)

    return vocab
# def get_model(question_id,student_answer):
#     sess = tf.Session()
# #     pb_file_path = "G:/工作任务2019/合作相关/合作成果/IntelligentJudgmentServer/src/py_server/feature_generater/models/"+question_id+"/model.pb"
#     pb_file_path = main_path()+"/models/" + question_id + "/model.pb"
#     print('我被调用了，我是获取模型')
#     with gfile.FastGFile(pb_file_path, 'rb') as f:
#         graph_def = tf.GraphDef() 
#         graph_def.ParseFromString(f.read())
#         sess.graph.as_default()
#         tf.import_graph_def(graph_def, name='')  # 导入计算图
# 
#     # 需要有一个初始化的过程
#     sess.run(tf.global_variables_initializer())
# 
#     # 输入
#     input_reference = sess.graph.get_tensor_by_name('standard_answer:0')
#     input_student = sess.graph.get_tensor_by_name('student_answer:0')
#     input_reference_length = sess.graph.get_tensor_by_name('standard_answer_length:0')
#     input_student_length = sess.graph.get_tensor_by_name('student_answer_length:0')
# 
#     op = sess.graph.get_tensor_by_name('y_pred_cls:0')
# 
#     reference_dict = {"math_002": "老师可以在155-165的身高范围内挑选队员.因为在此范围内，人数最为集中，且大家的身高相对接近.",
#                       "math_004": "两点之间线段最短；直线外一点到这条直线上所有点连结的线段中，垂线段最短.(或垂线段最短)",
#                       "CRCC_1": "《红岩》是题材独特的红色经典小说。小说记叙了1948年在国民党统治下革命志士在监狱中同敌人英勇斗争的事迹，再现了中国共产党的坚强意志和高尚品格。",
#                       "CRCC_2": "海底世界充满异国风情和浓厚的浪漫主义色彩以及曲折的情节和对海洋知识的介绍都深深吸引我。",
#                       "CRCC_3": "扁鹊直接指出蔡桓公的病，不被采纳，而邹忌通过讽谏的方式，而齐王也纳谏了。这说明，我们在人际交往中，要注意说话的方式、语气、分寸，做到既表达实际，又要让对方容易接受。", }
# 
#     reference = reference_dict[question_id]
# 
#     r = get_sentence_index(reference, vocab)
#     s = get_sentence_index(student_answer, vocab)
#     score = sess.run(op, feed_dict={input_reference: r,
#                                     input_student: s,
#                                     input_reference_length: [50],
#                                     input_student_length: [55],
#                                     # dropout:0.9
#                                     }
#                      )
# 
#     return score
def get_sess(question_id):
    """
    加载 对应question_id 的模型,返回sess
    :param question_id:
    :return:
    """
    tf.reset_default_graph()
    sess = tf.Session()
    # 模型文件地址
#     pb_file_path = "G:/工作任务2019/合作相关/合作成果/IntelligentJudgmentServer/src/py_server/feature_generater/models/" + question_id + "/model.pb"
    pb_file_path = main_path()+"/models/" + question_id + "/model.pb"
    print('我被调用了，我是获取session')
    with gfile.FastGFile(pb_file_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')  # 导入计算图

    # 需要有一个初始化的过程
    sess.run(tf.global_variables_initializer())

    return sess
def get_score(sess,vocab,question_id,student_answer):
    """
    预测分值
    :param sess: 模型sess
    :param question_id: 试题id
    :param student_answer: 学生答案
    :return: 返回分值
    """
    # 输入
    input_reference = sess.graph.get_tensor_by_name('standard_answer:0')
    input_student = sess.graph.get_tensor_by_name('student_answer:0')
    input_reference_length = sess.graph.get_tensor_by_name('standard_answer_length:0')
    input_student_length = sess.graph.get_tensor_by_name('student_answer_length:0')

    op = sess.graph.get_tensor_by_name('y_pred_cls:0')

    reference_dict = {"math_002":"老师可以在155-165的身高范围内挑选队员.因为在此范围内，人数最为集中，且大家的身高相对接近.",
                      "math_004":"两点之间线段最短；直线外一点到这条直线上所有点连结的线段中，垂线段最短.(或垂线段最短)",
                      "CRCC_1": "《红岩》是题材独特的红色经典小说。小说记叙了1948年在国民党统治下革命志士在监狱中同敌人英勇斗争的事迹，再现了中国共产党的坚强意志和高尚品格。",
                      "CRCC_2": "海底世界充满异国风情和浓厚的浪漫主义色彩以及曲折的情节和对海洋知识的介绍都深深吸引我。",
                      "CRCC_3": "扁鹊直接指出蔡桓公的病，不被采纳，而邹忌通过讽谏的方式，而齐王也纳谏了。这说明，我们在人际交往中，要注意说话的方式、语气、分寸，做到既表达实际，又要让对方容易接受。",
                      "math_001": "四条边相等的四边形是菱形；菱形的对角线互相垂直平分.",
                       "math_003": "同旁内角互补两直线平行,两直线平行同旁内角互补",}

    reference = reference_dict[question_id]

    r = get_sentence_index(reference, vocab)
    s = get_sentence_index(student_answer, vocab)
    score = sess.run(op, feed_dict={input_reference: r,
                                    input_student: s,
                                    input_reference_length: [50],
                                    input_student_length: [55],
                                    # dropout:0.9
                                    }
                     )
    score = score[0]
    if question_id == "CRCC_3":
        score = score/2.0
    return score
def feedback(question_id,score):
    """
    根据分值给出反馈意见
    :param question_id:试题id
    :param score: 分值
    :return: 返回反馈意见
    """
    Feedback = ""
def checkRepeat(student_answer):
    # """
    # 检测答案中的重复片段
    # :param student_answer:学生答案
    # :return: 返回重复片段概率及重复片段
    # """
    # word_list = getWord_Tokens(student_answer)
    # length = len(word_list)
    # def getstr(word_list):
    #     str = ""
    #     for word in word_list:
    #         str += word
    #     return str
    # # 重复词长度
    # max_overlap_length = 0
    # # 重复词所占比率
    # p = 0.0
    # overlap_str= ""
    # for i in range(2,int(length/2)):
    #     word_i_list = []
    #     for j in range(0,length):
    #         word_i_list.append(getstr(word_list[j:j+i]))
    #     for k in range(len(word_i_list)):
    #         if word_i_list[k] in word_i_list[k+1:]:
    #             max_overlap_length = i
    #             overlap_str = word_i_list[k]
    # # 重复词所占比率
    # p = (max_overlap_length*1.0)/length
    # # print (overlap_str)
    # if max_overlap_length != 0 and overlap_str != "垂线段最短":
    #     print('发现重复现象')
    #     return True
    # else:
        return False
def checkIllegal(vocab,student_answer):
    """
    检测学生答案中的无关词语
    :param vocab: 词表
    :param student_answer:学生答案
    :return: 返回学生答案中的无关词语所占概率及无关词语
    """
    count = 0
    unknow_list = []
    word_list = getWord_Tokens(student_answer)
    length = len(word_list)
    for word in word_list:
        if word not in vocab.token2id:
            if word in ['，','。','、','；','！','.']:
                continue
            count += 1
            unknow_list.append(word)
    p = (count * 1.0) / length
    # print (unknow_list,p)
    if p > 0.15:
        print('发现非法词')
        return True
    else:
        return False
def checkSecquence(question_id,student_answer):
    """
    判断是否存在词序错误，若存在返回True，否则返回False
    :param question_id:题号
    :param student_answer:学生答案
    :return:
    """
    # 文件路径
    # path = "/root/workspace/IntelligentJudgmentServer/src/py_server/feature_generater/data/" + question_id + "/2gram.txt"
    # tempQuery = str(student_answer)
    # sequence_word = ''
    # tempQuery = ",".join(jieba.lcut(student_answer))
    # arr = tempQuery.split(",")
    # ngrams = []
    # for i in range(len(arr) - 1):
    #     ngrams.append(arr[i:i + 2])
    # f1 = open(path, "r", encoding='utf-8')
    # f1 = f1.readlines()
    # for r in ngrams:
    #     r = str(r).replace(",", "").replace("'", "").replace("[", "").replace("]", "")
    #     rlist = r.split(" ")
    #     for i in f1:
    #         ii = i.replace("\n", "")
    #         iilist = ii.split(" ")
    #         if r == ii:
    #             break
    #         elif (rlist[0] == iilist[1] and rlist[1] == iilist[0]):
    #             sequence_word = r
    #             # return (r, "请注意这两个词的书写顺序")
    #             # print (r)
    # if sequence_word != '':
    #     print('我的顺序有问题')
    #     return True
    # else:
    #     return False
    return False
def mpredict_score(sess, vocab,question_id,question_content):
    """
    :param question_id:试题id
    :param question_contet
      """
    #检测无关词语
    IllegalTorF = checkIllegal(vocab=vocab,student_answer=question_content)
    # print ("IllegalTorF", IllegalTorF)
    #检测重复语段
    RepeatTorF= checkRepeat(student_answer=question_content)
    # print ("RepeatTorF", RepeatTorF)
    #检测单词词序
    SecquenceTorF = checkSecquence(question_id,question_content)
    # print ("SecquenceTorF", SecquenceTorF)
    if IllegalTorF or RepeatTorF or SecquenceTorF:
        return 0
    #预测分值
   
    score = get_score(sess,vocab,question_id,str(question_content))

    return score
if __name__ == '__main__':
    print ("--------Main----------")
    # print (get_score(get_model("math_002",sess),"math_002","选择155-165，因为身高集中。"))
    # print(get_model("math_002","选择155-165，因为身高集中。"))
    # print(get_model("math_002","选择155-165，因为身高集中。"))
    # print(get_model("math_002","选择155-165，因为身高集中。"))
    print(main_path())
    question_id = "math_002"
    student_answer = "选择，因为身高集中。"
    #加载词
    vocab = get_vovab(question_id)
    #加载模型
    sess = get_sess(question_id)
#     #预测分值
#     score = get_score(sess,vocab,question_id,student_answer)
#     print (score)
#     score = get_score(sess, vocab, question_id, student_answer)
#     print (score)
#     #结束
#     sess.close()
# 
#     question_id = "math_004"
#     student_answer = "两点之间线段最短。"
#     # 加载词
#     vocab = get_vovab(question_id)
#     # 加载模型
#     sess = get_sess(question_id)
#     # 预测分值
#     score = get_score(sess, vocab, question_id, student_answer)
#     print (score)
#     score = get_score(sess, vocab, question_id, student_answer)
#     print (score)
    # 结束
    print(mpredict_score(sess, vocab,question_id,student_answer))
    
  