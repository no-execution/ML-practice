from numpy import *
import operator
from numpy import array
import matplotlib
import matplotlib.pyplot as plt
from os import listdir


def file2matrix(filename):
	fr = open(filename)   #存在同一个文件夹下。
	arrayOFLines = fr.readlines()
	numberOFLines = len(arrayOFLines)
	returnMat = mat(zeros((numberOFLines,3)))  #zeros()的参数是一个tuple数组(行，列)。返回一个用0
	classLabelVector = []					   #填充的数组。	
	index = 0
	for line in arrayOFLines:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index = index+1
	return returnMat,classLabelVector

def autoNorm(dataSet):
	min_val = dataSet.min(0) #0代表列1代表行,什么都没有代表所有。
	max_val = dataSet.max(0)
	ranges = max_val-min_val #是一个行向量
	norm_mat = zeros(shape(dataSet))     #矩阵类型不能直接赋值？
	m = dataSet.shape[0]
	norm_mat = dataSet - tile(min_val,(m,1))  #减去最小值，再除以极值间的长度
	norm_mat = norm_mat/tile(ranges,(m,1))
	return norm_mat,ranges,min_val

def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group,labels

def classify0(inX,dataSet,labels,k):
	#这里把输入的参数明确一下。
	#1.inX表示待判断的项目。这是一个向量，包括该对象的各项参数(比如亲吻次数，打斗次数等等)
	#2.dataSet代表数据集，应该是带有对应标签的，用来判断inX的类别。
	#3.labels就是dataSet对应的标签向量
	#3.k就是knn里的k。
	#dataSetSize代表列的数量，#diffMat代表inX与每个标准#数据对应项目的差，以方便计算距离。
	#diffMat**2将矩阵中每个元素平方化，再求和取平方根就是距离了~~~。
	#求出距离后排序，distances.argsort()。
	#argsort()这个函数需要注意，他不返回原队列排序后的结果，而是将原队列从小到大排序，然后返回对
	#应的下标。(这样其实方便去对应标签。)
	#a.get(tag,0)如果字典中有tag则代表tag的值，否则创建key为tag的键值对，初始值为0.
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX,(dataSetSize,1)) - dataSet 
	sqDiffMat = array(diffMat)**2	   			
	sqDistances = sqDiffMat.sum(axis=1)	  #axis = 1代表行,axis=1代表列。对矩阵专用。
	distances = sqDistances**0.5
	sortedDistIndicie = distances.argsort() #一定要注意maxtrix和array的区别！！！否则会出偏差！！！
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicie[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1  #统计标签出现的次数。
	sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	#classCount.iteritems()在python3中改成了items()，他的作用是把键值对一起返回，因为如果不用
	#items()的话，返回的只有键的值，这样就无法取到标签x的频数了。key=operator.itemgetter(1)代表用
	#classCount.items()的第2个量排序(也就是用频数排序。),reverse就是降序。
	return sortedClassCount[0][0]  #取频数最高的标签。

def datingClassTest():
	hoRatio = 0.10
	datingDataMat,datingDataLabels = file2matrix('datingTestSet2.txt')
	norm_mat,ranges,min_val = autoNorm(datingDataMat)
	m = norm_mat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classfierResult = classify0(norm_mat[i,:],norm_mat[numTestVecs:m,:],\
			datingDataLabels[numTestVecs:m],3)
		print('the result is %d'%classfierResult,'the right answer is %d'%datingDataLabels[i])
		if (classfierResult!=datingDataLabels[i]):
			errorCount = errorCount+1.0
	print('the error rate is %f'%(errorCount/float(numTestVecs)))

def get_mr_right():
	a0 = float(input('每年坐飞机的里程数'))
	a1 = float(input('玩游戏时间百分比'))
	a2 = float(input('冰激凌公升数'))
	q = file2matrix('datingTestSet2.txt')
	se = q[0]
	labels = q[1]
	norm_mat,fenmu,min_val = autoNorm(se)
	k = classify0((array([a0,a1,a2])-min_val)/fenmu,norm_mat,labels,3)
	print([None,'屌丝','小有魅力','极有魅力'][int(k)])

#################

def img2vector(filename):
	returnVect = zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0,32*i+j] = int(lineStr[j])
	return returnVect

def handwritingClassTest():
	hwLabels = []
	trainingFileList = listdir('trainingDigits')
	m = len(trainingFileList)
	trainingMat = zeros((m,1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0]) #9_79代表数字9的第79个测试数据
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2vector('trainingDigits/%s'%fileNameStr)
	testFileList = listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = fileStr.split('_')[0]
		vectorUnderTest = img2vector('testDigits/%s'%fileNameStr)
		classfierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
		print('the result is %s'%classfierResult,'the right answer is %s'%classNumStr)
		if int(classfierResult)!=int(classNumStr):
			errorCount = errorCount+1.0
	print('error rate is %f'%(errorCount/float(mTest)))