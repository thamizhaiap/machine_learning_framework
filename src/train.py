import os
import pandas as pd
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
import joblib

from . import dispatcher

total_split = 5

FOLD_MAPPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

MODEL = input('enter random_forest/extra_forest: ')
print(f'TRAINING {MODEL}_model')
print('*******************************')

# Loading k-fold data set for training
df = pd.read_csv('input/train_folds.csv')

# Loading test dataset
test_df = pd.read_csv('input/test.csv')


if __name__ == "__main__":
     
     total_roc_auc = []
     for FOLD in range(5):

        # selecting the training dataframe and valid dataframe based on the fold values. 
        train_df = df[df.kfold.isin(FOLD_MAPPPING.get(FOLD))].reset_index(drop=True)
        valid_df = df[df.kfold==FOLD]

        # training target and validation target
        train_y = train_df.target.values
        valid_y = valid_df.target.values

        # training dataset and validation dataset
        train_df = train_df.drop(['id','target','kfold'], axis=1)
        valid_df = valid_df.drop(['id','target','kfold'], axis=1)        
        
        # valid_df = valid_df[train_df.columns]
     

        label_encoders = {}

        for c in train_df.columns:
            lbl = preprocessing.LabelEncoder()
            train_df.loc[:,c] = train_df.loc[:,c].astype(str).fillna('NONE')
            valid_df.loc[:,c] = valid_df.loc[:,c].astype(str).fillna('NONE')
            test_df.loc[:,c] = test_df.loc[:,c].astype(str).fillna('NONE')

            lbl.fit(train_df[c].values.tolist()+valid_df[c].values.tolist()+test_df[c].values.tolist())
            train_df.loc[:,c] = lbl.transform(train_df[c].values.tolist())
            valid_df.loc[:,c] = lbl.transform(valid_df[c].values.tolist())
        
            label_encoders[c] = lbl


        
        clf = dispatcher.MODELS[MODEL]
        clf.fit(train_df, train_y)
        preds = clf.predict_proba(valid_df)[:,1]
        # print(preds)

        roc_auc = metrics.roc_auc_score(valid_y, preds)
        print('roc_auc :', roc_auc)
        total_roc_auc.append(roc_auc)

       

        


        joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_label_encoder.pkl")
        joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
        joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")

     print(total_roc_auc)     
     print('ROC_AUC_SCORE: ', sum(total_roc_auc)/total_split)