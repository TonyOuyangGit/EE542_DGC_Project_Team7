# copyright: yueshi@usc.edu
# modified: yutongou@usc.edu shujiayy@usc.edu xiangchl@usc.edu
import pandas as pd 
import hashlib
import os 
from utils import logger
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from scipy import interp
from sklearn.preprocessing import label_binarize
from sklearn.feature_selection import SelectFromModel
from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from utils import logger
#def lassoSelection(X,y,)

def lassoSelection(X_train, y_train, n):
	'''
	Lasso feature selection.  Select n features. 
	'''
	#lasso feature selection
	#print (X_train)
	clf = LassoCV()
	sfm = SelectFromModel(clf, threshold=0)
	sfm.fit(X_train, y_train)
	X_transform = sfm.transform(X_train)
	n_features = X_transform.shape[1]
	
	#print(n_features)
	while n_features > n:
		sfm.threshold += 0.01
		X_transform = sfm.transform(X_train)
		n_features = X_transform.shape[1]
	features = [index for index,value in enumerate(sfm.get_support()) if value == True]
	logger.info("selected features are {}".format(features))
	return features


def specificity_score(y_true, y_predict):
	'''
	true_negative rate
	'''
	true_negative = len([index for index,pair in enumerate(zip(y_true,y_predict)) if pair[0]==pair[1] and pair[0]==0 ])
	real_negative = len(y_true) - sum(y_true)
	return true_negative / real_negative 

def model_fit_predict(X_train,X_test,y_train,y_test):

	np.random.seed(2018)
	from sklearn.linear_model import LogisticRegression
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.ensemble import GradientBoostingClassifier
	from sklearn.ensemble import ExtraTreesClassifier
	from sklearn.svm import SVC
	from sklearn.metrics import precision_score
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import f1_score
	from sklearn.metrics import recall_score
	models = {
		'LogisticRegression': LogisticRegression(),
		# 'ExtraTreesClassifier': ExtraTreesClassifier(),
		# 'RandomForestClassifier': RandomForestClassifier(),
  #   	'AdaBoostClassifier': AdaBoostClassifier(),
  #   	'GradientBoostingClassifier': GradientBoostingClassifier(),
  #   	'SVC': SVC()
	}
	tuned_parameters = {
		'LogisticRegression':{'random_state': 0,
								'solver' : 'lbfgs',
								'multi_class' : 'multinomial'
								}
		# 'ExtraTreesClassifier': { 'n_estimators': [16, 32] },
		# 'RandomForestClassifier': { 'n_estimators': [16, 32] },
  #   	'AdaBoostClassifier': { 'n_estimators': [16, 32] },
  #   	'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] },
  #   	'SVC': {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
	}
	scores= {}
	# for key in models:
		# clf = GridSearchCV(models[key], tuned_parameters[key], scoring=None,  refit=True, cv=10)
	clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
	# clf.fit(X_train,y_train)
	y_test_predict = clf.predict(X_test)
	y_score = clf.decision_function(X_test)

	precision_micro = precision_score(y_test, y_test_predict, average = "micro")
	precision_marco = precision_score(y_test, y_test_predict, average = "macro")
	precision_weighted = precision_score(y_test, y_test_predict, average = "weighted")
	print("precision_micro: " + str(precision_micro))
	print("precision_marco: " + str(precision_marco))
	print("precision_weighted: " + str(precision_weighted))
	accuracy = accuracy_score(y_test, y_test_predict)
	print("accuracy: " + str(accuracy))
	f1_micro = f1_score(y_test, y_test_predict, average = "micro")
	f1_macro = f1_score(y_test, y_test_predict, average = "macro")
	f1_weighted = f1_score(y_test, y_test_predict, average = "weighted")
	print("f1_micro: " + str(f1_micro))
	print("f1_macro: " + str(f1_macro))
	print("f1_weighted: " + str(f1_weighted))
	recall_micro = recall_score(y_test, y_test_predict, average = "micro")
	recall_macro = recall_score(y_test, y_test_predict, average = "macro")
	recall_weighted = recall_score(y_test, y_test_predict, average = "weighted")
	print("recall_micro: " + str(recall_micro))
	print("recall_macro: " + str(recall_macro))
	print("recall_weighted: " + str(recall_weighted))
	# specificity = specificity_score(y_test, y_test_predict)
	scores = [precision_marco,accuracy,f1_macro,recall_macro]
	# scores = [0.8813116656993616,0.8544249290249643,0.8447784315662945,0.8797101098582462]
	# print(scores)
	return scores



def draw(scores):
	'''
	draw scores.
	'''
	import matplotlib.pyplot as plt
	logger.info("scores are {}".format(scores))
	ax = plt.subplot(111)
	precisions = []
	accuracies =[]
	f1_scores = []
	recalls = []
	# categories = []
	# specificities = []
	N = len(scores)
	ind = 0  # set the x locations for the groups
	width = 0.1        # the width of the bars
	# for key in scores:
	# categories.append(key)
	precisions.append(scores[0])
	accuracies.append(scores[1])
	f1_scores.append(scores[2])
	recalls.append(scores[3])
		# specificities.append(scores[key][4])

	precision_bar = ax.bar(ind, precisions,width=0.1,color='b',align='center')
	accuracy_bar = ax.bar(ind+2*width, accuracies,width=0.1,color='g',align='center')
	f1_bar = ax.bar(ind+4*width, f1_scores,width=0.1,color='r',align='center')
	recall_bar = ax.bar(ind+6*width, recalls,width=0.1,color='y',align='center')
	# specificity_bar = ax.bar(ind+4*width,specificities,width=0.1,color='purple',align='center')

	# print(categories)
	# ax.set_xticks(np.arrage(N))
	# ax.set_xticklabels(categories)
	# ax.legend((precision_bar[0], accuracy_bar[0],f1_bar[0],recall_bar[0],specificity_bar[0]), ('precision', 'accuracy','f1','sensitivity','specificity'))
	ax.legend((precision_bar[0], accuracy_bar[0],f1_bar[0],recall_bar[0]), ('precision', 'accuracy','f1','sensitivity'))
	ax.grid()
	plt.show()

if __name__ == '__main__':


	data_dir ="/Users/Tony/Desktop/"

	data_file = data_dir + "miRNA_matrix_label.csv"

	df = pd.read_csv(data_file)
	# print(df)
	y_data = df.pop('label').values

	df.pop('file_id')

	columns =df.columns
	#print (columns)
	X_data = df.values
	
	# split the data to train and test set
	X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=0)
	

	# standardize the data.
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)

	
	n = 100
	feaures_columns = lassoSelection(X_train, y_train, n)
	# feaures_columns = [49, 217, 240, 285, 287, 514, 1860]
	# feaures_columns = [9, 13, 49, 54, 71, 74, 78, 96, 107, 
	# 144, 149, 163, 179, 180, 185, 187, 191, 194, 195, 203, 204, 
	# 205, 217, 226, 240, 242, 248, 252, 253, 254, 269, 276, 282, 
	# 284, 285, 287, 301, 309, 316, 325, 328, 338, 342, 353, 356,
	# 464, 487, 490, 495, 498, 500, 505, 514, 518, 538, 588, 594, 
	# 631, 714, 764, 767, 768, 795, 856, 894, 897, 957, 967, 991, 
	# 1006, 1056, 1066, 1101, 1106, 1121, 1125, 1214, 1285, 1299, 
	# 1309, 1330, 1342, 1371, 1388, 1462, 1527, 1530, 1577, 1632, 
	# 1637, 1655, 1689, 1722, 1742, 1768, 1791, 1834, 1838, 1848, 1860]
	# print(X_train[1:,feaures_columns])
	# print(X_test[1:,feaures_columns])
	# scores = [0.8813116656993616,0.8544249290249643,0.8447784315662945,0.8797101098582462]

	scores = model_fit_predict(X_train[:,feaures_columns],X_test[:,feaures_columns],y_train,y_test)
	# scores = model_fit_predict(X_train, X_test, y_train, y_test)
	draw(scores)
	#lasso cross validation
	# lassoreg = Lasso(random_state=0)
	# alphas = np.logspace(-4, -0.5, 30)
	# tuned_parameters = [{'alpha': alphas}]
	# n_fold = 10
	# clf = GridSearchCV(lassoreg,tuned_parameters,cv=10, refit = False)
	# clf.fit(X_train,y_train)




 




