import random
from numpy import *

def load_dataset(filename):
	data_mat,label_mat = [],[]
	fr = open(filename)
	for line in fr.readlines():
		line_arr = line.strip().split('\t')
		data_mat.append([float(line_arr[0]),float(line_arr[1])])
		label_mat.append(float(line_arr[2]))
	return data_mat,label_mat

#找到两个相异的下标，选择对应α作为smo的两个未知量
def select_jrand(i,m):
	j = i
	while j==i :
		j = int(random.uniform(0,m))
	return j

def clip_alpha(aj,h,l):
	if aj > h:
		aj = h
	if aj < l:
		aj = l
	return aj


def smo_simple(pre_data_mat,pre_label_mat,const,error_toler,max_iter):
	data_mat,label_mat = mat(pre_data_mat),mat(pre_label_mat).transpose()     
	b =0     
	m,n = shape(data_mat)     
	alphas = mat(zeros((m,1)))     
	iter = 0 
	while iter < max_iter:         
		alpha_pair_changed = 0         
		for i in range(m):
		#这个fxi是计算WX+b             
			fxi = float(multiply(alphas,label_mat).T *(data_mat*data_mat[i,:].T)) + b 
		#ei是计算结果和实际类别的误差
			ei = fxi - float(label_mat[i])
		#也就是说误差超出了可以接受的范围，并且α满足约束条件
			if ((label_mat[i]*ei < -error_toler) and (alphas[i] < const)) or ((label_mat[i]*ei > error_toler) and (alphas[i]>0)):
				j = select_jrand(i,m)
				fxj = float(multiply(alphas,label_mat).T *(data_mat*data_mat[j,:].T)) + b
				ej = fxj - float(label_mat[j])
				alpha_i_old = alphas[i].copy()
				alpha_j_old = alphas[j].copy()
				#接下来的l和h是上限与下限
			if label_mat[i] != label_mat[j]:
				# alpha[i] - alpha[j] = 常数 如果 = const，则不行
				l = max(0,alphas[j] - alphas[i])
				h = min(const,const+alphas[j] - alphas[i])
			else:
				#alpha[i] + alpha[j] = 常数  如果=const，则不行
				l = max(0,alphas[j] + alphas[i] - const)
				h = min(const,alphas[j] + alphas[i])
			if l == h:
				print('l=h')
				continue
			eta = 2.0 *data_mat[i,:]*data_mat[j,:].T - data_mat[i,:]*data_mat[i,:].T \
				- data_mat[j,:]*data_mat[j,:].T
			if eta >= 0:
				print('eta > 0')
				continue
			alphas[j] -= label_mat[j]*(ei-ej)/eta
			alphas[j] = clip_alpha(alphas[j],h,l)
			if (abs(alphas[j] - alpha_j_old) < 0.00001):
				print('not enough')
				continue
			alphas[i] += label_mat[j] * label_mat[i] * (alpha_j_old - alphas[j])

			b1 = b - ei - label_mat[i]*(alphas[i] - alpha_i_old)*data_mat[i,:]*data_mat[i,:].T \
				- label_mat[j]*(alphas[j]-alpha_j_old)*data_mat[i,:]*data_mat[j,:].T
			b2 = b - ej - label_mat[i]*(alphas[i] - alpha_i_old)*data_mat[i,:]*data_mat[j,:].T \
				- label_mat[j] * (alphas[j] - alpha_j_old) * data_mat[j,:] * data_mat[j,:].T
			if alphas[i] > 0 and alphas[i] < const:
				b = b1
			elif alphas[j] > 0 and alphas[j] < const:
				b = b2
			else:
				b = (b1+b2)/2
			alpha_pair_changed += 1
			print("iter: %d i: %d,alpha_pair_changed %d" %(iter,i,alpha_pair_changed))
		if alpha_pair_changed == 0:
			iter += 0
		else:
			iter += 1
			print("iteration number: %d" %iter)
	return b,alphas

def plot_res(data,label,u,b):
	import matplotlib.pyplot as plt
	fig = plt.figure()
	ax1 = fig.add_subplot(1,1,1)
	for i in range(len(label)):
		if label[i] == 1:
			ax1.scatter(data[i][0],data[i][1],color='blue')
		else:
			ax1.scatter(data[i][0],data[i][1],color='red')
	x = arange(10)
	ax1.plot(x,(-u[0,0]/u[0,1])*x-b/u[0,1],color='green')
	plt.show()