import jieba 
import jieba.posseg as pos
from sklearn.externals import joblib
from feature_generater.tf_idf import read_json
from nltk.corpus import stopwords
from _ast import If
# from dask.tests.test_cache import flag

wordList = ['番茄苗','碳酸氢钠','二氧化碳','碘液','淀粉','光合作用','产生']
for i in wordList:
	jieba.add_word(i)
item = '某生物兴趣小组选用番茄苗做实验材料开展了一系列探究活动。先将番茄苗置于黑暗处一昼夜，然后盖上透明玻璃罩并用凡士林密封，实验装置见图甲（碳酸氢钠可增加水中及周围环境中二氧化碳浓度） 将装置甲在室外阳光下放置一段时间。取一片番茄叶片，酒精隔水加热后，用清水清洗后滴加碘液，片刻后洗去，叶片变蓝，该现象可说明'
def constructWordDict(proItem):
	itemWords = []
	itemWords = jieba.cut(proItem,cut_all = False)
	return itemWords

def filter(answer,dictionaryAll):
	if len(answer) < 4:
		return False
	else:
		itemWords = []
		itemWords = constructWordDict(item)
		answerWords = []
		answerTemp = []
		answerWords = jieba.cut(answer, cut_all=False)
		#过滤在题目中的词
		for word in answerWords:
			if word not in itemWords:
				answerTemp.append(word)
			
		wordScore = 0.0
		dictAll = dictionaryAll.token2id.keys()
		#这个是获取的词表
# 		print(dictAll)
		answer_t = ''.join(answerTemp)
		if answer_t.find(',')!=-1:
			answer_s = answer.split(',')
			for i in range(len(answer_s)):
				print(answer_s)
				z = jieba.cut(answer_s[i],cut_all = False)
				temp = []
				for word in z:
					temp.append(word)
				for word in temp:
					if word in dictAll:
						wordScore += 1.0
						wordScore = wordScore/(len(temp)*1.0+0.1)
						
				if wordScore < 0.8:
					print(wordScore)
					print('发现分句的错误')
					return False
		for word in answerTemp:
			if word in dictAll:
				wordScore += 1.0
		wordScore = wordScore/(len(answerTemp)*1.0+0.1)
# 		answerWord = pos.cut(''.join(answerTemp))
# 		for w in answerWord:
# 			 str1= str+str(w.flag)
		stopword = []
		for a in open('/root/下载/IntelligentJudgmentServer/src/py_server/feature_generater/json/停用词.txt','r',encoding = 'utf-8'):
			stopword.append(a.split('\n')[0])
# 			print(a)
		for item1 in stopword:
			if(answer.find(item1) !=-1):
# 				print(answer)
# 				print(item1)
				return False
			
#判断一下是否含有不该有的词

		if wordScore < 0.6 or (''.join(answerTemp).find('淀粉') == -1 and ''.join(answerTemp).find('光合作用')== -1 ):
# 			print(wordScore)
			return False
		else:
			return True
#
def coverdata(qustion_content,name):
	 x,z = read_json(name)
	 for item in x:
	 	if(item == qustion_content):
	 		return True 
	 	else:
	 		return False
	 	

	 	
	 	
	 	
	 	
def judgeOrNot(question_id,answer):
    if question_id == '2017_11_25_ph_1' :
        dictionaryAll = joblib.load('/root/下载/IntelligentJudgmentServer/src/py_server/feature_generater/vector/2017_11_25_ph_1.dic.cf.m')
        if filter(answer,dictionaryAll) == True:
            if coverdata(answer,'/root/下载/IntelligentJudgmentServer/src/py_server/feature_generater/json/love_YX_1_v2.json')==True:
                return True
            else:
                return answer
        else:
            return ''
    elif (question_id == '2017_11_25_ph_3'):
        if(coverdata(answer,'/root/下载/IntelligentJudgmentServer/src/py_server/feature_generater/json/2017_11_25.json')==True):
            return True
        else:
            return answer
	  		 

	




