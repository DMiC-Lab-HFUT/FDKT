from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import metrics
from utils.makdir import mkdir
import numpy as np


def obtain_rmse(actual, pred):
    return sqrt(mean_squared_error(actual, pred))


def obtain_mae(actual, pred):
    return mean_absolute_error(actual, pred)


def obtain_auc(actual, pred):
    fpr, tpr, thresholds = metrics.roc_curve(actual, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc


def obtain_r2(actual, pred):
    return r2_score(actual, pred)


def obtain_metrics(c_actual, c_pred):
    c_actual = np.array(c_actual).astype(float).tolist()
    try:
        auc = obtain_auc(c_actual, c_pred)
    except:
        auc = 9999

    cc_rmse = obtain_rmse(c_actual, c_pred)
    cc_mae = obtain_mae(c_actual, c_pred)
    return cc_rmse, cc_mae, auc


def write_metrics(te_cc_rmse, te_cc_mae, base_dir, args):
    mkdir(base_dir)
    mkdir(base_dir + args[0] + '/')
    mkdir(base_dir + args[0] + '/' + str(args[1]) + '/')
    mkdir(base_dir + args[0] + '/' + str(args[1]) + '/')

    with open(base_dir + args[0] + '/' + str(args[1]) + '/te_cc_rmse.txt', 'a') as f:
        for each in te_cc_rmse:
            f.writelines(str(each) + '\n')
    with open(base_dir + args[0] + '/' + str(args[1]) + '/te_cc_mae.txt', 'a') as f:
        for each in te_cc_mae:
            f.writelines(str(each) + '\n')
    print("write success")
