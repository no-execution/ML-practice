from numpy import *
from pandas import *

class tree(object):
	def __init__(self):
		self.left = None
		self.right = None
		self.judge = 0
		self.x = None
		self.y = None
		self.c = None
		self.frame = None
		self.fea = None
		self.cla = None
		self.res = None   #这个属性只有叶节点才配有

def find_best_s(x,y):
	#x,y为array或者Series类型，one-dimension
	m = shape(x)[0]     
	min_loss = 0
	frame = DataFrame(x,columns=['x'])
	frame['y'] = y
	for i in range(m):
		r1,r2 = frame[frame['x']<=frame['x'][i]],frame[frame['x']>frame['x'][i]]
		c1,c2 = float(mean(r1['y'])),float(mean(r2['y']))
		mid_loss = float(sum((r1['y']-c1)**2))+float(sum((r2['y']-c2)**2))   #通过这种方式求平方和
		if i == 0:
			min_loss = mid_loss
			s,j = frame['x'][i],i
		if mid_loss < min_loss:
			min_loss = mid_loss
			s,j = frame['x'][i],i
	return float(s),j

def generate_reg_tree(root):
	if shape(root.x)[0] <= 30:    #递归的baseline
		root.c = float(mean(root.y))
		return root
	root.c = float(mean(root.y))
	s,j = find_best_s(root.x,root.y)  #找到每组数据的最佳分类值s，传入的x与y为array类型或者Series类型
	root.judge = s                  #把分类的标准传给节点，方便分类器使用
	left,right = tree(),tree()    #初始化左右儿子
	frame_out = DataFrame(root.x,columns=['x'])
	frame_out['y'] = root.y
	r1,r2 = frame_out[frame_out['x']<=s],frame_out[frame_out['x']>s]
	left.x,left.y = array(r1['x']),array(r1['y'])  #把归为类别1的数据的x,y存到左儿子里，保证数据为列向量
	right.x,right.y = array(r2['x']),array(r2['y']) #把归为类别2的数据的x,y存到右儿子里，保证数据为列向量
	root.left = generate_tree(left)
	root.right = generate_tree(right)
	return root
	
def plot_tree(root):
	pass
	

#这里假设data为一条数据向量，第一个分量为x，第二个分量为y,返回该数据向量所属的集合的c
def classify(root,data):
	base = root
	while base.left and base.right:
		if data[0] <= base.judge:
			base = base.left
		else:
			base = base.right
	print('the c of this data is %f'%base.c)
	print('the loss of this data is %f'%(data[1]-base.c)**2)

#--------------------------------------------------------------------------------		
def load_data_cla(filename):
	f = open(filename)
	s = f.readlines()
	m = len(s)
	data = zeros((15,5))
	for i in range(m):
		data[i,:] = s[i].split()
	frame = DataFrame(data,columns=['age','worked','have house','loan situation','yes or no'])
	return frame

def init_root(frame):
	root = tree()
	root.frame = frame
	return root
#生成一个决策树：找到最佳分类特征---→把数据分为两类----
#直到gini指数很小，或者集合中的数据个数小于预定个数时，
#还有可能是没有特征可用时(该条件分的最细，最容易过拟合)，停止递归。
#初始的root需要一个frame
def generate_cla_tree(root):
	if len(root.frame.index) <= 5:
		root.res = value_counts(root.frame[root.frame.columns[-1]]).index[0]
		return root
	if len(value_counts(root.frame[root.frame.columns[-1]]).index)==1:
		root.res = value_counts(root.frame[root.frame.columns[-1]]).index[0]
		return root		
	root.cla,root.fea = find_best_feature(root.frame)
	left,right = tree(),tree()
	left.frame = root.frame[root.frame[root.cla]==root.fea]
	right.frame = root.frame[root.frame[root.cla]!=root.fea]
	root.left = generate_cla_tree(left)
	root.right = generate_cla_tree(right)
	return root

def find_best_feature(frame):
	best_cla,best_fea = None,None
	min_gini = 10
	for cla in frame.columns[:-1]:
		for fea in value_counts(frame[cla]).index:
			mid_gini = calc_gini(cla,fea,frame)
			if  mid_gini < min_gini:
				best_cla,best_fea = cla,fea
				min_gini = mid_gini
	return best_cla,best_fea

#计算gini指数，gini指数的计算需要指定一个类别的一个特征
def calc_gini(cla,fea,frame):
	print(cla,fea)
	print(frame)
	zonghe = sum(value_counts(frame[frame.columns[-1]]))
	f1 = frame[frame[cla] == fea] 
	f2 = frame[frame[cla] != fea]
	count1 = value_counts(f1[f1.columns[-1]])      #真正分类的那一个Series
	if len(count1) == 0:
		p1,c1 = 0,0
	else:	
		p1 = count1[count1.index[0]]/(sum(count1))
		c1 = sum(count1)/zonghe
	count2 = value_counts(f2[f2.columns[-1]])
	if len(count2)  == 0:
		p2,c2 = 0,0
	else:
		p2 = count2[count2.index[0]]/(sum(count2))
		c2 = sum(count2)/zonghe
	return 2*p1*(1-p1)*c1 + 2*p2*(1-p2)*c2
#---------------------------------------------------------------------------------
def load_data_reg(filename):
	f = open(filename)
	s = f.readlines()
	m = len(s)
	x,y = zeros(m),zeros(m)
	for i in range(m):
		mid = s[i].split()
		x[i],y[i] = float(mid[0]),float(mid[1])
	return x,y			

def gen(x,y):
	#统一规定：tree类型的x和y都采用array或者Series类型，以便进一步操作
	root = tree()
	root.x = x
	root.y = y
	return root

def main(filename):
	x,y = load_data(filename)
	root = gen(x,y)
	return generate_tree(root)

#该函数依次打印一个二叉树的所有叶节点
def get_reg_leaf(root):
	if not root.left and not root.right:
		print('--------------fengeline-----------------')
		'''
		print('the c is :',root.c)
		print('the judge value is:',root.judge)
		print('the x data is:',root.x)
		print('the y data is:',root.y)
		'''
		print('the class is',root.res)
		print('the frame is',root.frame)
		return
	get_reg_leaf(root.left)
	get_reg_leaf(root.right)
