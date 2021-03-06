import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib

from . import dispatcher

TRAINING_DATA = os.environ.get('TRAINING_DATA')
TEST_DATA = os.environ.get('TEST_DATA')
FOLD = int(os.environ.get('FOLD')) # Fold must be an int
MODEL = os.environ.get('MODEL')

# create fold mapping to get train and validation sets
FOLD_MAPPING = {
    0:[1,2,3,4],
    1:[0,2,3,4],
    2:[0,1,3,4],
    3:[0,1,2,4],
    4:[0,1,2,3]
}

if __name__ == '__main__':
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)
    # create train and validation sets from dictionary
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))].reset_index(drop = True)
    valid_df = df[df.kfold==FOLD].reset_index(drop = True)

    # get target variables for training and validation
    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    # Drop unnecesary columns
    train_df = train_df.drop(['id', 'target', 'kfold'], axis = 1)
    valid_df = valid_df.drop(['id', 'target', 'kfold'], axis = 1)

    # Make sure valid and train have the same order of columns
    valid_df = valid_df[train_df.columns]

    #simple encoding of the variables
    label_encoders = {}
    for col in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[col].values.tolist() + valid_df[col].values.tolist() + df_test[col].values.tolist())
        train_df.loc[:,col] = lbl.transform(train_df[col].values.tolist())
        valid_df.loc[:,col] = lbl.transform(valid_df[col].values.tolist())
        label_encoders[col] = lbl
    
    # Initialize model
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:,1]
    print(metrics.roc_auc_score(yvalid, preds))

    # save work to file
    joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_label_encoder.pkl")
    joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")
    


