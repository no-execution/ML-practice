from numpy import *
from pandas import *

#如何让数据集具有分类的标志？：额外创建一个label向量用于聚类？
#通过calc_dist函数返回的值，来改变label？
#分类的标签用1,2,3,4....k来表示
def k_mean(data_mat,k,iter_steps=20):
	m,n = shape(data_mat)	
	frame = DataFrame(data_mat)
	label = mat(zeros((m,1)))
	frame['label'] = label
	init_point = []
	#该while循环随机选出k个点作为k个集合的初始质心进行第一步计算。
	while len(init_point) < k:
		mid = random.randint(m)
		if len(init_point) == 0:
			init_point.append(mid)
		elif mid != init_point[-1]:
			init_point.append(mid)
		else:
			continue
	cent_points = mat(zeros((k,n)))
	for z in range(k):
		cent_points[z,:] = data_mat[init_point[z],:]
	#########以上步骤完成了kmean算法的初始设定，接下来要进行迭代#########
	#迭代的停止条件可以用迭代步数或者两步之间质心的偏移量的大小来进行判定
	iters = 0
	while iters < iter_steps: #or 偏移量>alpha:
		#先分类
		for j in range(m):
			frame['label'][j] = find_best_class(data_mat[j,:],cent_points) 
		#再算cent_point,有k个集合，也就有k个质心。
		for i in range(k):
			mid_class = frame[frame['label']==(i+1)]
			cent_points[i,:] = mean(mid_class[mid_class.columns[:-1]])
		iters += 1
	return frame


#这个函数的功能是：计算某“一个”点到k个簇的距离，并返回簇的标签
def find_best_class(data_arr,cent_points):	
	#假设进来的是列向量	
	min_dist = 100000
	best_class = 0
	k = shape(cent_points)[0]  #各个簇的质心，每个质心代表一个向量，m为k
	for i in range(k):
		mid = data_arr - cent_points[i,:]
		dist = float(mid * mid.T)
		if dist < min_dist:
			min_dist = dist
			best_class = i+1
	return best_class

def load_data(filename):
	f = open(filename)
	s = f.readlines()
	data_mat = mat(zeros((len(s),len(s[0].split()))))
	for i in range(shape(data_mat)[0]):
		data_mat[i,:] = s[i].split()
	return data_mat

def plot_cluster(frame):
	import matplotlib.pyplot as plt
	colors = ['r','g','b']
	fig = plt.figure()
	mid = value_counts(frame['label'])
	k = len(mid)
	for i in range(k):
		d = frame[frame['label']==mid.index[i]]
		plt.scatter(d[0],d[1],color=colors[i])
	plt.show()
