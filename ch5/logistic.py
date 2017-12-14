from math import *
from numpy import *
import random

def load_dataset():
	data_mat,label_mat = [],[]
	fr = open('testSet.txt')
	for line in fr.readlines():
		info = line.split()
		data_mat.append([1.0,float(info[0]),float(info[1])])  #[1.0,特征1,特征2]
		label_mat.append(int(info[2]))
	return data_mat,label_mat

def sigmoid(input_x):
	return 1.0/(1+exp(-input_x))

def grad_ascent(pre_data_mat,pre_label_mat):
	data_mat = mat(pre_data_mat)
	label_mat = mat(pre_label_mat)
	m,n = shape(data_mat)
	alpha = 0.001
	max_cycles = 500
	weights = ones((n,1))    #权重的初始值为1,1,1，在这里n=3.
	for k in range(max_cycles):
		h = sigmoid(data_mat*weights)   #把当前的线性组合代入sigmoid函数，算分类
		error = (label_mat.transpose() - h)         #计算得到的值与实际分类的偏差,有正有负。
		weights = weights + alpha*data_mat.transpose() * error  #error代表方向
	return weights

def grad_ascent_simply(pre_data_mat,pre_label_mat):
	data_mat,label_mat = mat(pre_data_mat),mat(pre_label_mat).transpose()
	m,n = shape(data_mat)
	weights = ones((n,1))  #列向量，3*1
	alpha = 0.01
	for i in range(m):
		h = sigmoid(float(data_mat[i] * weights))
		error = float(label_mat[i] - h)
		weights = weights + alpha*error*data_mat[i].transpose()
	return weights

def grad_ascent_simply_random(pre_data_mat,pre_label_mat):
	data_mat,label_mat = mat(pre_data_mat),mat(pre_label_mat).transpose()
	m,n = shape(data_mat)
	weights = ones((n,1))
	alpha = 0.01
	for i in range(2000):
		mid = int(random.uniform(0,m))
		h = sigmoid(float(data_mat[mid] * weights))
		error = float(label_mat[mid] - h)
		weights += alpha*error*data_mat[mid].transpose()
	return weights


def plot_best_fit(weights):
	import matplotlib.pyplot as plt
	data_mat,label_mat = load_dataset()
	data_arr = array(data_mat)
	n = shape(data_arr)[0]
	x_cord1,y_cord1,x_cord2,y_cord2 = [],[],[],[]
	#把分类不同的两种点的坐标，分别加入不同的集合，以便进行不同的画图。
	for i in range(n):
		if int(label_mat[i]) == 1:
			x_cord1.append(data_mat[i][1])
			y_cord1.append(data_mat[i][2])
		if int(label_mat[i]) == 0:	
			x_cord2.append(data_mat[i][1])
			y_cord2.append(data_mat[i][2])
	#添加一张画布，一般用fig变量表示。
	fig = plt.figure()
	#将画布分块，代表分成一块，并且使用这一块。
	ax = fig.add_subplot(1,1,1)
	#分别指定两类点的外观。
	ax.scatter(x_cord1,y_cord1,s=30,c='red',marker='s')
	ax.scatter(x_cord2,y_cord2,s=30,c='green')
	#代表横坐标取-3到3每0.1个距离取一个点，一共60个点(目的是绘制曲线。)
	x = arange(-3.0,3.0,0.1)
	print(type(x))
	y = (-weights[0]-weights[1]*x)/weights[2]
	print(y,shape(y),type(y))
	ax.plot(x,y.transpose())
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.show()