import pandas as pd 
from sklearn import model_selection

if __name__ == '__main__':
    df = pd.read_csv('input/train.csv')
    # create a kfold column to assign rows to an specific fold
    df['kfold'] = -1
    
    #shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)

    # Initialize cross validation
    skf = model_selection.StratifiedKFold(n_splits=5, shuffle=False,
            random_state=42)
    
    # Iterate through the folds to get validation and train index
    for fold, (trainIdx, valIdx) in enumerate(skf.split(X=df, y=df.target.values)):
        # get train and index shape 
        print(len(trainIdx), len(valIdx))
        # Fill up kfold column
        df.loc[valIdx, 'kfold'] = fold

    # save df to file
    df.to_csv('input/train_folds.csv', index=False)

    

