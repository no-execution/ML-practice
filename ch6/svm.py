import random

def load_dataset(filename):
	data_mat,label_mat = [],[]
	fr = open(filename)
	for line in fr.readlines():
		line_arr = line.strip().split('\t')
		data_mat.append([float(line_arr[0]),float(line_arr[1])])
		label_mat.append(float(line_arr[2]))
	return data_mat,label_mat

def select_jrand(i,m):
	j = i
	while j==i :
		j = int(random.uniform(0,m))
	return j

def clip_alpha(aj,h,l):
	if aj > h:
		aj = h
	if aj < L:
		aj = L
	return aj


def smo_simple(pre_data_mat,pre_label_mat,const,error_toler,max_iter):
	data_mat,label_mat = mat(pre_data_mat),mat(pre_label_mat).transpose()
	b = 0
	m,n = shape(data_mat)