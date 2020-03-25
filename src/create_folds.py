import pandas as pd 
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv('input/train.csv')
    df['kfold'] = -1
    
    # shuffling the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    total_split = 5

    kf = model_selection.StratifiedKFold(n_splits=total_split, shuffle=True, random_state=42)
       
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.target.values)):
        df.loc[val_idx, 'kfold'] = fold
       
    # creating csv file       
    df.to_csv('input/train_folds.csv', index=False)

