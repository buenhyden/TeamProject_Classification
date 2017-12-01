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
