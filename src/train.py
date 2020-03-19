import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics

from . import dispatcher

# FOLD = 0


MODEL = input('enter rf/et: ')

FOLD_MAPPING = {
    0:[1,2,3,4],
    1:[0,2,3,4],
    2:[0,1,3,4],
    3:[0,1,2,4],
    4:[0,1,2,3]
}

if __name__ == "__main__":

    Avg_roc_auc = []

    for FOLD in range(4):
        df = pd.read_csv('input/train_folds.csv')
        
        train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))].reset_index(drop=True)
        valid_df = df[df.kfold==FOLD]
        # print('train_df.shape',train_df.shape)
        # print('valid_df.shape',valid_df.shape)

        ytrain = train_df.target.values
        yvalid = valid_df.target.values
        # print('ytrain len',len(ytrain))
        # print('yvalid len', len(yvalid))
        
        # training and validation data
        train_df = train_df.drop(['id','target','kfold'], axis=1)
        valid_df = valid_df.drop(['id','target','kfold'], axis=1)
        # print('train shape',train_df.shape)
        # print('valid shape',valid_df.shape)

        valid_df = valid_df[train_df.columns]
        # print(valid_df.head())

        label_encoders = {}
        for c in train_df.columns:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist())
            train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
            valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
            label_encoders[c] = lbl

            
        # # data is ready to train

        clf = dispatcher.MODELS[MODEL]
        clf.fit(train_df, ytrain)
        pred = clf.predict_proba(valid_df)[:,1]
        print(pred)
        roc_auc = metrics.roc_auc_score(yvalid, pred)
        print(roc_auc)
        Avg_roc_auc.append(roc_auc)
        
        
    print('ROC_AUC_SCORE: ', sum(Avg_roc_auc)/4)










