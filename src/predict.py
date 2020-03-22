import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
import joblib

from . import dispatcher



def predict():
    df_test = pd.read_csv('input/test.csv')
    df_test.head()
    # test_idx = df['id'].values
    # print(test_idx)
    # predictions = None

    # for FOLD in range(5):
    #     encoders = joblib.load(os.join('models'), f'{MODEL}_{FOLD}_label_encoder.pkl')
    #     cols = joblib.load(os.path.join('models', f"{MODEL}_{FOLD}_columns.pks"))
         
    #     for c in cols:
    #         lbl = label_encoders[c]
    #         df.loc[:,c] = lbl.transform(df[c].values.tolist())

    #     clf = joblib.load(os.path.join('models', f"{MODEL}_{FOLD}.pkl"))
        
    #     df = df[cols]
    #     preds = clf.predict_proba(df[:,1])

    #     if FOLD == 0:
    #         predictions = preds

    #     else:
    #         predictions += preds

    #     predictions /= 5

    #     sub = pd.DataFrame(np.columnsOstack((test_idx, predictions)),columns=['id','predict'])
    #     joblib.dump(label_encoders, f"models/{MODEL}_label_encoder.pkl")
    #     joblib.dump(clf, f"models/{MODEL}.pkl")
    #     joblib.dump(train_df.columns, f"models/{MODEL}_columns.pkl")


predict()