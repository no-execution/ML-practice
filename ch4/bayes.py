from numpy import *
from math import log


def load_data_set():
	posting_list = [['my','dog','has','flea','problems','help','please'],\
	['maybe','not','take','him','to','dog','park','stupid'],\
	['my','dalmation','is','so','cute','I','love','him'],\
	['stop','posting','stupid','worthless','garbage'],\
	['mr','licks','ate','my','steak','how','to','stop','him'],\
	['quit','buying','worthless','dog','food','stupid']]
	class_vec = [0,1,0,1,0,1]
	return posting_list,class_vec

#返回词汇(所有出现过得词)表
def create_vocab_list(data_set):
	vocab_set = set([])
	for document in data_set:
		vocab_set = vocab_set | set(document)
	return list(vocab_set)


#针对一条向量，给出表示每个词是否出现的向量。ex：[1,0,1,0]代表词1和词3出现，词2，词4未出现。
def set_words_to_vec(vocab_list,input_set):
	res_vec = [0]*len(vocab_list)
	for word in input_set:
		if word in vocab_list:
			res_vec[vocab_list.index(word)] = 1
		else:
			print('error')
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



def classify(vec_to_classify,p0_vec,p1_vec,p_class):
	p1 = sum(vec_to_classify * p1_vec) + log(p_class)
	p0 = sum(vec_to_classify * p0_vec) + log(1.0 - p_class)
	if p1 > p0:
		return 1
	else:
		return 0

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