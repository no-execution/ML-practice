from math import log
import operator

#首先明白给出的数据是怎样的
#[0,1,'no'] 首先整个决策树是判断某个量属于哪一类。
#以这个例子为例，只有两个类，1.是鱼2.不是鱼。但是对于其他的情况，有可能出现很多类比如1.男2.女3.gay
#给出的数据中用0和1代替“是”，“否”，其实可以有更多的分支，比如胖，瘦，正好。

#计算给出数据集的香农熵
def cal_shannon_ent(dataset):
	n_nums = len(dataset)
	label_count = {}
	#创建以类为key的dict，items为该类包含的信息条数(用于进行线性组合)
	for x in dataset:
		label = x[-1]
		if not label in label_count.keys():
			label_count[label] = 0
		label_count[label] += 1
	ent = 0.0
	#计算香农熵，prob小于1，虽然ent是-=，但是因此ent实际上是递增的
	#计算公式∑pi*log2(pi) 
	for key in label_count:
		prob = float(label_count[key])/n_nums
		ent -= prob*log(prob,2)
	return ent


def split_data(dataset,axis,value):
	ret = []
	for x in dataset:
		if x[axis] == value:
			reduced_vec = x[:axis]
			reduced_vec = reduced_vec+x[axis+1:]
			ret.append(reduced_vec)
	return ret

#要重点认识data_set的含义，data——set(数据和集合)
def choose_best(data_set):
	num_features = len(data_set[0])-1   #特征数或者说特征向量的长度。
	base_ent = cal_shannon_ent(data_set)  #先算出整个数据集的基础香农熵
	best_info_gain = 0.0
	best_feature = -1
	for i in range(num_features):		  #第一个for针对特征向量中每个值(每个i对应一个特征)
		feat_list = [x[i] for x in data_set]  #抽取所有数据的第i个特征
		unique_val = set(feat_list)       #表示i个特征的所有分类(有房没房之类的)
		new_ent = 0.0
		for value in unique_val:
			sub_set = split_data(data_set,i,value)  #按第i个特征的所有可能取值选取数据集 
			prob = len(sub_set)/float(len(data_set))
			new_ent += prob*cal_shannon_ent(sub_set)  #对len(set)个不同取值计算熵
		info_gain = base_ent - new_ent				  #每个特征的熵:对该特征按其所有取值分类得到的集合	
		if info_gain > best_info_gain:				  #单独求熵，然后按比例线性组合求和。
			best_info_gain = info_gain    #取特征向量中所有值中信息增益最大的那个作为最好的特征。
			best_feature = i              #输出的是特征对应的序号	
	return best_feature


#多数表决，一个分类中肯定不可能全都是同一类型的信息，因此找一个符合大多数信息的特征作为分类。
def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if not vote in classCount.keys():
			classCount[vote] = 0
		classCount[vote] += 1
	return max(classCount.items())[0]
			
#递归的基底:特征数贼小（每次labels都缩短）
def create_tree(dataset,labels):
	#先判断是否满足停止条件.
	classList = [x[-1] for x in dataset]   #收集所有的类:[[1],[2],[3]] → [1,2,3]方便运算
	if classList.count(classList[0]) == len(dataset):  #所有数据条目都是同类的。
		return classList[0]
	if len(dataset[0]) == 1:     #只剩下类了，那就多数表决
		return majorityCnt(classList)
	best_feat = choose_best(dataset)  #返回特征的编号
	bestFeatLabel = labels[best_feat]
	myTree = {bestFeatLabel:{}}
	del(labels[best_feat])   #将该特征从标签中删除
	featValues = [x[best_feat] for x in dataset]
	unique_vals = set(featValues)
	for x in unique_vals:
		subLabels = labels[:]
		myTree[bestFeatLabel][x] = create_tree(split_data(dataset,best_feat,x),subLabels)
	return myTree


def classify(input_tree,labels,test_vec):
	first_str = list(input_tree.keys())[0]
	second_dict = input_tree[first_str]
	feat_index = labels.index(first_str)
	for key in second_dict.keys():
		if test_vec[feat_index] == key:
			if type(second_dict[key]).__name__ == 'dict':
				class_label = classify(second_dict[key],labels,test_vec)
			else:
				class_label = second_dict[key] 
	return class_label


def creat_set():
	dataset = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
	labels = ['no surfacing','flippers']
	return dataset,labels

def save_tree(input_tree,filename):
	import pickle
	fw = open(filename,'wb')     #注意加b,因为存储的是二进制文件		
	pickle.dump(input_tree,fw)   #把input_tree这个饺子下到fw里。
	fw.close()

def get_tree(filename):
	import pickle
	fr = open(filename,'rb')  #注意加b,因为读取的是二进制序列
	return pickle.load(fr)    #把fr装载出来


def initial_dataset(filename):
	fr = open(filename)
	info = fr.readlines()
	data_set = []
	for line in info:
		mid = line.split()
		if len(mid) == 5:
			data_set.append(mid)
		elif len(mid) == 6:
			mid[4] = mid[4]+' '+mid[5]
			data_set.append(mid[:5])
	train_set = []
	mid_set = []
	test_set = []
	s = len(data_set)-1
	while s >= 0:
		if s%2 == 1:
			train_set.append(data_set[s])
		else:
			mid_set.append(data_set[s])
		s -= 1
	s = len(mid_set) - 1
	while s>=0:
		if s%2 == 0:
			train_set.append(data_set[s])
		else:
			test_set.append(data_set[s])
		s -= 1
	labels = ['age','prescript','astigmatic','tearRate']
	return train_set,test_set,labels

