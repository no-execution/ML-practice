from numpy import *
from math import log


####寻找最佳阈值,暂定为在两数中间，如(1.5,2.5等。。)
#定义分类为：大于阈值=-1，小于阈值=1或者相反，目的是使加权损失最小。
def find_yuzhi(y,w):
	flag = 0.5
	m = int(shape(y)[0])
	min_error = 1
	w_error = 0
	pos = 0
	while flag < m:
		w_error = 0
		for i in range(m):
			if i > flag and y[i] != -1:
				w_error += w[i]
			if i < flag and y[i] != 1:
				w_error += w[i]
		if w_error < min_error:
			min_error = w_error
			pos = flag
		flag += 1
	flag_re = 0.5
	min_error_re = 1
	w_error_re = 0
	pos_re = 0
	while flag_re < m:
		w_error_re = 0
		for i in range(m):
			if i < flag_re and y[i] != -1:
				w_error_re += w[i]
			if i > flag_re and y[i] != 1:
				w_error_re += w[i]
		if w_error_re < min_error_re:
			min_error_re = w_error_re
			pos_re = flag_re
		flag_re += 1
	if min_error_re < min_error:
		return -pos_re,min_error_re
	else:
		return pos,min_error	

#分类并返回一个数组
def classify(x,pos):
	m = int(shape(x)[0])
	res = zeros(m)
	if pos > 0:
		for i in range(m):
			if x[i] > pos:
				res[i] = -1
			else:
				res[i] = 1
		return res
	else:
		for j in range(m):
			if x[j] > -pos:
				res[j] = 1
			else:
				res[j] = -1
		return res

#更新alpha，w，输出pos用来构造最终的分类器		
def ada(x,y,w):
	m = int(shape(w)[0])
	pos,e= find_yuzhi(y,w)
	g_arr = classify(x,pos)
	error = 0
	alpha = 0.5*log((1-e)/e)
	z = 0
	for i in range(m):
		z += w[i]*exp(-alpha*y[i]*g_arr[i])
	for i in range(m):
		w[i] = (w[i]*exp(-alpha*y[i]*g_arr[i]))/z
	return alpha,w,pos 



def main():
	x,y = array([0,1,2,3,4,5,6,7,8,9]),array([1,1,1,-1,-1,-1,1,1,1,-1])
	m = shape(y)[0]
	w = ones(10)/float(m)
	alphas,poses = [],[]
	g = zeros(10)
	for i in range(3):
		alpha,w,pos = ada(x,y,w)
		alphas.append(alpha)
		poses.append(pos)
	for i in range(3):
		g += abs(alphas[i]) * array(classify(x,poses[i]))
	for i in range(int(shape(g)[0])):
		if g[i] > 0:
			g[i] = 1
		else:
			g[i] = -1
	print(g)

	