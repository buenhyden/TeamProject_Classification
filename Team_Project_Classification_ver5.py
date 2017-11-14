import pandas as pd
import numpy as np
import datetime
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import random
from operator import itemgetter
import time
import copy
import scipy as sp
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance


def intersect(a, b):
    return list(set(a) & set(b))


def get_features(train, test):
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    output = intersect(trainval, testval)
    output.remove('people_id')
    return sorted(output)

def Load_DataSet():
    print ("loading .....  act_train")
    act_train = pd.read_csv('./data/act_train.csv',
                            dtype={'people_id': np.str,
                                'activity_id': np.str,
                                'outcome': np.int8},
                         parse_dates=['date'])
    print ("loading ......  act_test")
    act_test = pd.read_csv('./data/act_test.csv',
                           dtype={'people_id': np.str,
                                'activity_id': np.str},
                         parse_dates=['date'])
    print ("loading ..... people")
    people = pd.read_csv('./data/people.csv',
                         dtype={'people_id': np.str,
                               'activity_id': np.str,
                               'char_38': np.int32},
                        parse_dates=['date'])
    print ("Merge ......")
    trainMerge= act_train.merge(people, on="people_id")
    testMerge = act_test.merge(people, on="people_id")
    print('Number of active people in train : {}'.format(trainMerge['people_id'].nunique()))
    print('Number of active people in test : {}'.format(testMerge['people_id'].nunique()))
    print ("Processing Train")
    for idx in trainMerge.columns:
        print (idx)
        if idx not in ['people_id', 'activity_id', 'date_x','date_y', 'char_38', 'outcome']:
            if trainMerge[idx].dtype == 'object':
                trainMerge.fillna('type 0', inplace = True)
                trainMerge[idx] = trainMerge[idx].apply(lambda x:x.split(' ')[1]).astype(np.int32)
            elif trainMerge[idx].dtype == 'bool':
                trainMerge[idx] = trainMerge[idx].astype(np.int8)
    trainMerge['date_x'] = pd.to_datetime(trainMerge['date_x'])
    trainMerge['date_y'] = pd.to_datetime(trainMerge['date_y'])
    trainMerge['year_x'] = trainMerge['date_x'].dt.year
    trainMerge['month_x'] = trainMerge['date_x'].dt.month
    trainMerge['day_x'] = trainMerge['date_x'].dt.day
    trainMerge['weekday_x'] = trainMerge['date_x'].dt.weekday
    trainMerge['weekend_x'] = ((trainMerge.weekday_x == 0) | (trainMerge.weekday_x == 6)).astype(int)
    trainMerge = trainMerge.drop('date_x', axis = 1)

    trainMerge['year_y'] = trainMerge['date_y'].dt.year
    trainMerge['month_y'] = trainMerge['date_y'].dt.month
    trainMerge['day_y'] = trainMerge['date_y'].dt.day
    trainMerge['weekday_y'] = trainMerge['date_y'].dt.weekday
    trainMerge['weekend_y'] = ((trainMerge.weekday_y == 0) | (trainMerge.weekday_y == 6)).astype(int)
    trainMerge = trainMerge.drop('date_y', axis = 1)

    print ("Processing Test")
    for idx in testMerge.columns:
        print (idx)
        if idx not in ['people_id', 'activity_id', 'date_x','date_y', 'char_38', 'outcome']:
            if testMerge[idx].dtype == 'object':
                testMerge.fillna('type 0', inplace = True)
                testMerge[idx] = testMerge[idx].apply(lambda x:x.split(' ')[1]).astype(np.int32)
            elif testMerge[idx].dtype == 'bool':
                testMerge[idx] = testMerge[idx].astype(np.int8)
    testMerge['date_x'] = pd.to_datetime(testMerge['date_x'])
    testMerge['date_y'] = pd.to_datetime(testMerge['date_y'])
    testMerge['year_x'] = testMerge['date_x'].dt.year
    testMerge['month_x'] = testMerge['date_x'].dt.month
    testMerge['day_x'] = testMerge['date_x'].dt.day
    testMerge['weekday_x'] = testMerge['date_x'].dt.weekday
    testMerge['weekend_x'] = ((testMerge.weekday_x == 0) | (testMerge.weekday_x == 6)).astype(int)
    testMerge = testMerge.drop('date_x', axis = 1)

    testMerge['year_y'] = testMerge['date_y'].dt.year
    testMerge['month_y'] = testMerge['date_y'].dt.month
    testMerge['day_y'] = testMerge['date_y'].dt.day
    testMerge['weekday_y'] = testMerge['date_y'].dt.weekday
    testMerge['weekend_y'] = ((testMerge.weekday_y == 0) | (testMerge.weekday_y == 6)).astype(int)
    testMerge = testMerge.drop('date_y', axis = 1)
    return trainMerge, testMerge


def run(train, test, random_state=0):
    eta = 1.3
    max_depth = 5
    subsample = 0.8
    colsample_bytree = 0.8
    params ={
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "max_depth" : max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent":1,
        "seed": random_state
    }
    num_boost_round = 120
    early_stopping_rounds = 10
    test_size = 0.15
    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    y_train = X_train['outcome']
    y_valid = X_valid['outcome']
    X_train = X_train.drop(['people_id','activity_id','outcome'], axis = 1)
    X_valid = X_valid.drop(['people_id','activity_id','outcome'], axis = 1)
    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_valid, y_valid)
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    check = gbm.predict(xgb.DMatrix(X_valid), ntree_limit=gbm.best_iteration+1)
    score = roc_auc_score(y_valid, check)
    testActivityId = test['activity_id']
    test = test.drop(['people_id','activity_id'],axis = 1)
    test_prediction = gbm.predict(xgb.DMatrix(test), ntree_limit=gbm.best_iteration+1)
    imp = get_importance(gbm,X_train.columns)
    print ('importance array: ', imp)
    out = pd.concat([testActivityId,pd.DataFrame(test_prediction.round())],axis = 1)
    out.rename({0:'outcome'},axis = 1, inplace = True)
    return out

def Main():
    train, test = Load_DataSet()
    out = run(train,test)
    out.to_csv('./submission7.csv',index = False)









if __name__ == "__main__":
    Main()



