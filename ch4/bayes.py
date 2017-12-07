from numpy import *
from math import log
import re
import random

'''
def load_data_set():
	posting_list = [['my','dog','has','flea','problems','help','please'],\
	['maybe','not','take','him','to','dog','park','stupid'],\
	['my','dalmation','is','so','cute','I','love','him'],\
	['stop','posting','stupid','worthless','garbage'],\
	['mr','licks','ate','my','steak','how','to','stop','him'],\
	['quit','buying','worthless','dog','food','stupid']]
	class_vec = [0,1,0,1,0,1]
	return posting_list,class_vec
'''

#返回词汇(所有出现过得词)表
def create_vocab_list(data_set):
	vocab_set = set([])
	for document in data_set:
		vocab_set = vocab_set | set(document)
	return list(vocab_set)

#文档词袋模型，某个词出现的次数可能不止一次。
def bag_words_to_vec(vocab_list,input_set):
	res_vec = [0]*len(vocab_list)
	for word in input_set:
		if word in vocab_list:
			res_vec[vocab_list.index(word)] += 1   #也可以是等于1
		else:
			print(error)
			return
	return res_vec
	

#针对一条向量，给出表示每个词是否出现的向量。ex：[1,0,1,0]代表词1和词3出现，词2，词4未出现。
def set_words_to_vec(vocab_list,input_set):
	res_vec = [0]*len(vocab_list)
	for word in input_set:
		if word in vocab_list:
			res_vec[vocab_list.index(word)] += 1
		else:
			print('error')
			return
	return res_vec

#需要求的是:
#一、p_bad(p(c1),该字段具有侮辱性的概率)。
#二、p0_vec(p(x=X|c1),在已知该字段具有侮辱性的情况下，某个词向量出现的概率，这个概率本身不好求
#但是根据朴素贝叶斯的假设，各(对于不同词汇)后验概率之间相互独立，因此可以转化为p(x1=X1|c1)*p(x2=X2|c1)
#.....p(xn=Xn|c1)，于是这里用一个向量表示所有的词汇的后验概率(也就是p0_vec[i]=p(xi=Xi|c1)).
#三、p1_vec与二中解释同理。
def train_bayes(train_matrix,train_class):
	n_train_docs = len(train_matrix)
	n_words = len(train_matrix[0])
	p_bad = sum(train_class)/float(n_train_docs)
	p1_num_vect = ones(n_words)
	p0_num_vect = ones(n_words)
	p1_denom,p0_denom = 2.0,2.0
	for i in range(n_train_docs):
		if train_class[i] == 1:
			p1_num_vect += train_matrix[i]
			p1_denom += sum(train_matrix[i])
		else:
			p0_num_vect += train_matrix[i]
			p0_denom += sum(train_matrix[i])
	p1_vec = array([log(x) for x in (p1_num_vect/p1_denom)])
	p0_vec = array([log(x) for x in (p0_num_vect/p0_denom)])
	return p0_vec,p1_vec,p_bad


#每次分类，测试数据是一个概率“向量”。(要用set_to_vec)
def classify(vec_to_classify,p0_vec,p1_vec,p_class):
	p1 = sum(vec_to_classify * p1_vec) + log(p_class)
	p0 = sum(vec_to_classify * p0_vec) + log(1.0 - p_class)
	if p1 > p0:
		return 1
	else:
		return 0
'''
def test_bayes():
	posts,classes = load_data_set()
	vocab_list = create_vocab_list(posts)
	train_matrix = []
	for x in posts:
		train_matrix.append(set_words_to_vec(vocab_list,x))
	p0_vec,p1_vec,p_bad = train_bayes(train_matrix,classes)
	test_input = ['love','my','dalmation']
	test_vec = array(set_words_to_vec(vocab_list,test_input))
	print('the class of your input is',classify(test_vec,p0_vec,p1_vec,p_bad))
	test_input_1 = ['stupid','garbage']
	test_vec_1 = array(set_words_to_vec(vocab_list,test_input_1))
	print('the class of your input is',classify(test_vec_1,p0_vec,p1_vec,p_bad))
'''

#用来预处理文本文档，核心是正则表达式。
#每次处理一个文档，或者说每次处理一个语句。
def pre_txt(filename):
	fr = open(filename)
	text = fr.read()
	rec = re.compile(r'\\W*')
	return [x.lower() for x in rec.split(text) if len(x) > 2]

#spam或者ham中的第6个文件有问题。
def test_spam():
	all_words = []
	docs_input = []
	classes = []
	for i in range(1,26):
		mid = pre_txt('D:\\Documents\\Desktop\\mechinelearning\\ch4\\spam\\%d.txt' %i)
		docs_input.append(mid)
		all_words.extend(mid)
		classes.append(1)
		mid = pre_txt('D:\\Documents\\Desktop\\mechinelearning\\ch4\\ham\\%d.txt' %i)
		docs_input.append(mid)
		all_words.extend(mid)
		classes.append(0)
	vocab_list = list(set(all_words))
	range_all = list(range(50)) 
	test_set = []
	test_classes = []
	train_set = []
	classes_train = []
	#选择测试数据，往分类器里输入,这里随机选择10个数据。
	for i in range(10):
		rand_index = int(random.uniform(0,len(range_all)))
		test_set.append(docs_input[rand_index])
		test_classes.append(classes[rand_index])
		del(range_all[rand_index])
	#获取剩余的数据作为训练集。因为是训练集，所以要转化成频数向量。
	for i in range_all:
		train_set.append(set_words_to_vec(vocab_list,docs_input[i]))	
		classes_train.append(classes[i])
	#把训练集输入进训练函数，计算各种概率
	p0_vec,p1_vec,p_bad = train_bayes(train_set,classes_train)
	error_count = 0	
	for i in range(10):
		res = classify(array(set_words_to_vec(vocab_list,test_set[i])),p0_vec,p1_vec,p_bad)
		if res != test_classes[i]:
			error_count += 1
	error_rate = float(error_count/len(test_set))
	#print('the error rate of the bayes model is',error_rate)		
	return error_rate

def test(x1,x2):
	count = 0.0
	for i in range(2000):
		count += test_local_words(x1,x2)
	print(count/2000)	

#计算所有词汇的词频，并按降序排序，注意排序用到的语句，好好记住。
def cal_freq(vocab_list,all_words):
	import operator
	word_freq = {}
	for word in vocab_list:
		word_freq[word] = all_words.count(word)
	#这条排序语句应当好好记住。
	sorted_freq = sorted(word_freq.items(),key=operator.itemgetter(1),reverse=True)


def text_parse(string):
	rec = re.compile(r'\\W*')
	return [x.lower() for x in re.split('\\W*',string) if len(x)>2]

#第二个例子，删除词频前7的词汇。ny ==== 1,sf====0
def test_local_words(feed_1,feed_0):
	all_words = []
	docs_input = []
	classes = []
	for i in range(25):
		mid_1 = feed_1['entries'][i]['summary_detail']['value']+ feed_1['entries'][i]['title_detail']['value']
		mid = text_parse(mid_1)
		docs_input.append(mid)
		all_words.extend(mid)
		classes.append(1)
		mid_1 = feed_0['entries'][i]['summary_detail']['value']+ feed_1['entries'][i]['title_detail']['value']
		mid = text_parse(mid_1)
		docs_input.append(mid)
		all_words.extend(mid)
		classes.append(0)
	vocab_list = list(set(all_words))
	range_all = list(range(50)) 
	test_set = []
	test_classes = []
	train_set = []
	classes_train = []
	#选择测试数据，往分类器里输入,这里随机选择10个数据。
	for i in range(10):
		rand_index = int(random.uniform(0,len(range_all)))
		test_set.append(docs_input[rand_index])
		test_classes.append(classes[rand_index])
		del(range_all[rand_index])
	#获取剩余的数据作为训练集。因为是训练集，所以要转化成频数向量。
	for i in range_all:
		train_set.append(set_words_to_vec(vocab_list,docs_input[i]))	
		classes_train.append(classes[i])
	#把训练集输入进训练函数，计算各种概率
	p0_vec,p1_vec,p_bad = train_bayes(train_set,classes_train)
	error_count = 0	
	for i in range(10):
		res = classify(array(set_words_to_vec(vocab_list,test_set[i])),p0_vec,p1_vec,p_bad)
		if res != test_classes[i]:
			error_count += 1
	error_rate = float(error_count/len(test_set))
	#print('the error rate of the bayes model is',error_rate)		
	return error_rate,p0_vec,p1_vec,vocab_list

def top_used_words(x1,x2):
	error_rate,p0_vec,p1_vec,vocab_list = test_local_words(x1,x2)
	sf_top = []
	ny_top = []
	for i in range(len(vocab_list)):
		if p0_vec[i] > -6.0:
			sf_top.append((p0_vec[i],vocab_list[i]))
		if p1_vec[i] > -6.0:
			ny_top.append((p1_vec[i],vocab_list[i]))
		#按照词频大小排序
	k1 = sorted(sf_top,key=lambda x:x[0],reverse = True)
	k2 = sorted(ny_top,key=lambda x:x[0],reverse = True)
	print('SF_SF_SF_SF_SF_SF_SF_SF_SF_SF_SF_SF')
	for i in range(10,20):
		print(k1[i])
	print('NY_NY_NY_NY_NY_NY_NY_NY_NY_NY_NY_NY')
	for i in range(10,20):
		print(k2[i])
	print('the error_rate is',error_rate)
