import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
import joblib

from . import dispatcher

model_path = "/home/tam/Desktop/DataScience/machine_learning_framework/models/"

def predict(test_data_path, model_type, model_path):
    
    test_df = pd.read_csv(test_data_path)
    # print(test_df.shape)
    test_idx = test_df["id"].values
    predictions = None


    for FOLD in range(5):
        test_df = pd.read_csv(test_data_path)
        encoders = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}_label_encoder.pkl"))
        cols = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}_columns.pkl"))
       
        for c in encoders:
            lbl = encoders[c]
            test_df.loc[:,c] = test_df.loc[:, c].astype(str).fillna("NONE")
            test_df.loc[:,c] = lbl.transform(test_df[c].values.tolist())

        clf = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}.pkl"))
        
        test_df = test_df[cols]
        preds = clf.predict_proba(test_df)[:,1]
        # print(len(preds))


        if FOLD == 0:
            predictions = preds
        else:
            predictions +=preds

    predictions /=5
    print(len(predictions))
    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=['id','target'])
    return sub
   
  
if __name__=="__main__":
    submission = predict(test_data_path="input/test.csv", model_type="rf", model_path=model_path)
    submission.loc[:,'id'] =  submission.loc[:,'id'].astype(int)
    submission.to_csv(f"models/rf_submission.csv", index=False)

        
        




        
