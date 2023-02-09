import pickle

import pandas as pd
from fit import train_and_evaluate, predict
from data_prep import read_data


class ModelWrapper:
    def __init__(self, model_name, n_cv_fold, num_boost_round, boosting_type='dart'):
        self.params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting': boosting_type,
                'num_leaves': 100,
                'learning_rate': 0.01,
                'feature_fraction': 0.20,
                'bagging_freq': 10,
                'bagging_fraction': 0.50,
                'n_jobs': -1,
                'lambda_l2': 2,
                'min_data_in_leaf': 40,
        }

        self.model_name = model_name
        self.model = None
        self.fitted = False
        self.train_score = None
        self.test_score = None
        self.type = 'lgb'
        self.label_encoders = None
        self.num_boost_round = num_boost_round
        self.n_cv_fold = n_cv_fold

    def fit(self, input_dir):
        train = read_data(input_dir, dataset='train')
        self.model, self.label_encoders, self.train_score = train_and_evaluate(self, train)
        self.fitted = True

    def predict(self, input_dir):
        test = read_data(input_dir, dataset='test')
        return predict(self, test)

