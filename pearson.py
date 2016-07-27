import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
	import scipy.stats

from utils import root_mean_squared_error as rmse, mean_absolute_deviation as mad
import utils

# Rows 531 through 536 belong to dataset 123, which has (for unknown reason) infinite Kurtosis 
# try to find out why you have to delete 530~534 instead
data = utils.load_file('metadata.txt', rows_to_ignore=[0,*range(530,535)], columns_to_ignore=[0, 1, *range(46,72,3), *range(50,66,3)], seperator=",")#*range(46,78)],seperator=",")#
column_names = utils.load_file('metadata.txt', rows_to_ignore=[*range(1,736)], columns_to_ignore=[0,1, *range(46,72,3), *range(50,66,3)], seperator=",")#*range(46,78)],seperator=",")#
data = data.astype(dtype=float)

target_columns = range(data.shape[1] - 7, data.shape[1])
X, y_rf, y_svc, y_tree, y_nb, y_boosting, y_lr, y_knn = np.split(data, target_columns, axis = 1)

P = np.empty(shape=(7,X.shape[1]))
T = np.empty(shape=(7,X.shape[1]))
for i, target in enumerate([y_rf, y_svc, y_tree, y_nb, y_boosting, y_lr, y_knn]):
	for j, mf in enumerate(np.split(X, range(1,X.shape[1]), axis = 1)):
		pearson, pvalue = scipy.stats.pearsonr(mf, target)
		P[i, j] = pearson
		T[i, j] = pvalue

pvalue_mask = T <= 0.05
# The 0.2 is more or less arbitrary
significant_columns_mask = [sum(col)== 7 and np.mean(P[:,i])>0.2 for i, col in enumerate(pvalue_mask.T)]
column_names_array = np.array(column_names[0])

# Prepare results for output to file.
significant_column_names = ['classifier', *column_names_array[np.where(significant_columns_mask)[0]]]
row_names = ['RandomForest', 'SVC', 'DecisionTree', 'Naive Bayes', 'Boosting', 'LR', 'kNN']
significant_column_values = np.around(P[:,np.where(significant_columns_mask)[0]], decimals=2).astype(str)

with open("pearson.tex", "w") as f:
	f.write(utils.numpy_to_latex(significant_column_values, significant_column_names, row_names))