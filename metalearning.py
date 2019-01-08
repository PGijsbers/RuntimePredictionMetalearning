import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge

from utils import root_mean_squared_error as rmse, mean_absolute_deviation as mad
import utils


def baseline_prediction(Y, strategy = "median"):
    assert strategy == "median" or strategy == "mean", "Invalid strategy used. Only median or mean are allowed."

    predictions = np.ones(Y.shape)
    for i, y in enumerate(Y):
        mask = np.ones(Y.shape,dtype=bool)
        mask[i] = False
        if strategy == "median":
            predictions[i] = np.median(Y[mask])
        if strategy == "mean":
            predictions[i] = np.mean(Y[mask])

    return predictions


def regressor_prediction(X, y, regressor_constructor):
    predictions = np.ones(y.shape)
    for i, y_i in enumerate(y):
        print("LOOCV {} of {}".format(i, len(y)))
        # find dataset id
        dataset_id = X[i, 0]

        # We create a LOOCV-split
        X_train = np.delete(X, np.where(X[:, 0] == dataset_id), 0)
        y_train = np.delete(y, np.where(X[:, 0] == dataset_id), 0).ravel()
        X_test = X[i, :].reshape(1, -1)

        # remove dataset id column
        X_train, X_test = X_train[:, 1:], X_test[:, 1:]

        regressor = regressor_constructor()
        regressor.fit(X_train, y_train)
        predictions[i] = regressor.predict(X_test)

    return predictions


def safe_log10(values):
    return np.array([np.log10(value) if value != 0 else 0 for value in values]).ravel()


# == Script takes roughly 20 minutes to run on my pc ==
# Rows 531 through 536 belong to dataset 123, which has (for unknown reason) infinite Kurtosis 
# try to find out why you have to delete 530~534 instead
data = utils.load_file('metadata.txt',
                       rows_to_ignore=[0, *range(530, 535)],
                       columns_to_ignore=[1, *range(46, 72, 3), *range(50, 66, 3)],
                       seperator=",")  #*range(46,78)],seperator=",")
column_names = utils.load_file('metadata.txt',
                               rows_to_ignore=[*range(1, 736)],
                               columns_to_ignore=[0, 1, *range(46, 72, 3), *range(50, 66, 3)],
                               seperator=",")  #*range(46,78)],seperator=",")#
data = data.astype(dtype=float)

target_columns = range(data.shape[1] - 7, data.shape[1])
X, *classifiers = np.split(data, target_columns, axis=1)
classifiers = [safe_log10(ys) for ys in classifiers]

mad_values = np.empty(shape=(5, len(classifiers)))
rmse_values = np.empty(shape=(5, len(classifiers)))

# Evaluate final regression
for j, target in enumerate(classifiers):
    median_estimates = baseline_prediction(target)
    mad_values[0, j] = mad(median_estimates, target)
    rmse_values[0, j] = rmse(median_estimates, target)

    mean_estimates = baseline_prediction(target, strategy = 'mean')
    mad_values[1, j] = mad(mean_estimates, target)
    rmse_values[1, j] = rmse(mean_estimates, target)

    rr_estimates = regressor_prediction(X, target, Ridge)
    mad_values[2, j] = mad(rr_estimates, target)
    rmse_values[2, j] = rmse(rr_estimates, target)

    rf_estimates = regressor_prediction(X, target, RandomForestRegressor)
    mad_values[3, j] = mad(rf_estimates, target)
    rmse_values[3, j] = rmse(rf_estimates, target)

    svr_estimates = regressor_prediction(X, target, SVR)
    mad_values[4, j] = mad(svr_estimates, target)
    rmse_values[4, j] = rmse(svr_estimates, target)

column_names = ['Regressor', 'RF', 'SVC', 'Tree', 'NB', 'Boosting', 'LR', 'kNN']
row_names = ['Median', 'Mean', 'RR', 'RF', 'SVR']
with open("meta-mad-results-log10-2.tex", "a") as f:
    f.write("\\documentclass[a4paper]{{article}}\n")
    f.write("\\begin{{document}}\n")

    f.write(utils.numpy_to_latex(np.around(mad_values, decimals=2).astype(str), column_names, row_names))
    f.write("\\end{{document}}\n")

with open("meta-rmse-results-log10-2.tex", "a") as f:
    f.write("\\documentclass[a4paper]{{article}}\n")
    f.write("\\begin{{document}}\n")
    f.write(utils.numpy_to_latex(np.around(rmse_values, decimals=2).astype(str), column_names, row_names))
    f.write("\\end{{document}}\n")
