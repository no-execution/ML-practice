from numpy import *
from pandas import *

class tree(object):
	def __init__(self):
		self.left = None
		self.right = None
		self.val = 0
		self.x = None
		self.y = None

def find_best_s(x,y):
	#x,y为array或者Series类型，one-dimension
	m = shape(x)[0]      #架构不对
	min_loss = 0
    frame = DataFrame(x,columns=['x'])
    fram['y'] = y
	for i in range(m):
		r1,r2 = frame[frame['x']<=frame['x'][i]],frame[frame['x']>frame['x'][i]]
		c1,c2 = float(mean(r1['y'])),float(mean(['y']))
		mid_loss = (r1-c1).T*(r1-c1)+(r2-c2).T*(r2-c2)   #通过这种方式求平方和
		if i == 0:
			min_loss = mid_loss
			s,j = x[i],i
		if mid_loss < min_loss:
			min_loss = mid_loss
			s,j = x[i],i
		ss = ss + 1
	return float(s),j

def generate_tree(root):
	if shape(root.x)[0] <= 10:    #递归的baseline
		return
	s,j = find_best_s(root.x,root.y)  #找到每组数据的最佳分类值s，传入的x与y为array类型或者Series类型
	root.val = s                  #把分类的标准传给节点，方便分类器使用
	left,right = tree(),tree()    #初始化左右儿子
	left.x,left.y = root.x[root.x<=s].T,root.y[root.y<=s].T  #把归为类别1的数据的x,y存到左儿子里，保证数据为列向量
	right.x,right.y = root.x[root.x>s].T,root.y[root.y>s].T  #把归为类别2的数据的x,y存到右儿子里，保证数据为列向量
	root.left = left
	root.right = right
	generate_tree(root.left)
	generate_tree(root.right)
	return root

def load_data(filename):
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