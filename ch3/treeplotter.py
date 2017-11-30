import matplotlib.pyplot as plt

decision_node = dict(boxstyle='sawtooth',fc='0.8')
leaf_node = dict(boxstyle='round4',fc='0.8')
arrow_args = dict(arrowstyle='<-')

def plot_node(nodeTxt,centerPt,parentPt,node_type):
	createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext = centerPt,\
		textcoords = 'axes fraction',va = 'center',ha = 'center',bbox = node_type,arrowprops = arrow_args)

def createPlot():
	fig = plt.figure(1,facecolor='white')
	fig.clf()
	createPlot.ax1 = plt.subplot(111,frameon=False)
	plot_node('决策节点',(0.5,0.1),(0.1,0.5),decision_node) #(方框中的文本,箭头线的终点，文本的位置，方框的样式)
	plot_node('叶节点',(0.8,0.1),(0.3,0.8),leaf_node)
	plt.show()	

def get_num_leaf(myTree):
	num_leaf = 0
	first_str = list(myTree.keys())[0]
	second_dict = myTree[first_str]
	for key in second_dict.keys():
		if type(second_dict[key]).__name__ == 'dict':
			num_leaf += get_num_leaf(second_dict[key])
		else:
			num_leaf += 1
	return num_leaf

def get_tree_depth(myTree):
	max_depth = 0
	first_str = myTree.keys()[0]
	second_dict = myTree[first_str]
	for key in second_dict.keys():
		if type(second_dict[key]).__name__ == 'dict':
			this_depth = 1+get_tree_depth(second_dict[key])
		else:
			this_depth = 1
		if this_depth > max_depth:
			max_depth = this_depth
	return max_depth