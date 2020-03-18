import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics

FOLD = 0

FOLD_MAPPING = {
    0:[1,2,3,4],
    1:[0,2,3,4],
    2:[0,1,3,4],
    3:[0,1,2,4],
    4:[0,1,2,3]
}

if __name__ == "__main__":
    df = pd.read_csv('input/train_folds.csv')
    

    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))].reset_index(drop=True)
    valid_df = df[df.kfold==FOLD]
    # print(train_df.head())
    # print(valid_df.head())

    ytrain = train_df.target.values
    yvalid = train_df.target.values
    # print(len(ytrain))
    # print(len(yvalid))
    
    # training and validation data
    train_df = train_df.drop(['id','target','kfold'], axis=1)
    valid_df = valid_df.drop(['id','target','kfold'], axis=1)
    # print(train_df.shape)
    # print(valid_df.shape)

    valid_df = valid_df[train_df.columns]
    # print(valid_df.head())

    label_encoders = {}
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist())
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders[c] = lbl

        
    # data is ready to train

    clf = ensemble.RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=2)
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:,1]
    print(preds)
 


