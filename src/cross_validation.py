
import pandas as pd
from sklearn import model_selection

"""
- -- binary classification
- -- multi class classification
- -- multi label classification
- -- single column regression
- -- multi column regression
- -- holdout
"""


class CrossValidation:
    def __init__(self, 
                    df, 
                    target_cols,
                    shuffle, 
                    problem_type='binary_classification',
                    multilabel_delimeter = ',',
                    num_folds =5,
                    random_state = 42
):

        self.df = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.shuffle = shuffle
        self.problem_type = problem_type
        self.multilabel_delimeter = multilabel_delimeter
        self.num_folds = num_folds


        if self.shuffle is True:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        
        self.df['kfold'] = -1

    def split(self):
        if self.problem_type in ("binary_classification","multiclass_classification"):
            if self.num_targets != 1:
                 raise Exception("invalid number of targets for this type of problem")
            target = self.target_cols[0]

            unique_values = self.df['target'].nunique()

            if unique_values == 1:
                raise Exception(" one one unique value found!")

            elif unique_values > 1 :
                kf = model_selection.StratifiedKFold(n_splits = self.num_folds,
                                                        shuffle = False)
                
                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.df, y=self.df.target.values)):
                    self.df.loc[val_idx, 'kfold'] = fold
        
        elif self.problem_type in ("single_col_regression","multi_col_regression"):
            if self.num_targets != 1 and self.problem_type == "single_col_regression":
                raise Exception("Invalid number of targets for this problem type")

            if self.num_targets < 2 and self.problem_type == "multi_col_regression":
                raise Exception("Invalid number of targets for this problem type")

            kf = model_selection.KFold(n_splits = self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(X = self.df):
                self.dataframe.loc[val_idx, 'kfold'] =  fold

        elif self.problem_type.startswith("holdout_"):
            holdout_percentage = int(self.problem_type.split("_")[1])                
            num_holdout_samples = int(len(self.df)* holdout_percentage/100)
            self.df.loc[:len(self.df) -  num_holdout_samples, "kfold"] = 0

            #############################################check the semicolon
            self.df.loc[len(self.df) -  num_holdout_samples:, "kfold"] = 1

        elif self.problem_type == "multilabel_classification":
            if self.num_targets != 1:
                raise Exception ("Invalid number of targets for this problem")
            
            targets = self.df[self.target_cols[0]].apply(lambda x: len(x).split(self.multilabel_delimeter))

            kf = model_selection.StratifiedKFold(n_splits = self.num_folds)

            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=targets)):
                self.df.loc[val_idx, 'kfold'] = fold

        else:
            raise Exception("Problem type not understood!")

        return self.df


if __name__ == "__main__":
    df = pd.read_csv("############")
    cv = CrossValidation(df, shuffle='multilabel_classification',
                                target_cols=['attribute_ids'],
                                problem_type="multilabel_classification",
                                multilabel_delimeter = " ")

    df_split = cv.split()
    print(df_split.head())
    print(df_split.kfold.value_counts())

    