import os
import pandas as pd
import numpy as np 
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib

from . import dispatcher

TEST_DATA = os.environ.get('TEST_DATA')
MODEL = os.environ.get('MODEL')

def predict():
    df = pd.read_csv(TEST_DATA)
    test_idx = df['id'].values
    predictions = None

    for FOLD in range(5):
        #get what FOLD we are currently in 
        print(FOLD)
        df = pd.read_csv(TEST_DATA)
        # load the encoder and the column names
        encoders = joblib.load(os.path.join('models',
            f"{MODEL}_{FOLD}_label_encoder.pkl"))
        cols = joblib.load(os.path.join('models',
            f"{MODEL}_{FOLD}_columns.pkl"))
        for col in encoders:
            # check which column is currently encoding
            print(col)
            # preprocess data for model
            lbl = encoders[col]
            df.loc[:,col] = lbl.transform(df[col].values.tolist())
        
        # load trained model
        clf = joblib.load(os.path.join('models',f'{MODEL}_{FOLD}.pkl'))

        #predict on test set 
        df = df[cols]
        preds = clf.predict_proba(df)[:,1]

        # gather FOLD predictions
        if FOLD == 0:
            predictions = preds 
        else:
            predictions += preds 
        
    # get the mean FOLD prediction
    predictions /= 5

    #prepare the submission file
    sub = pd.DataFrame(np.column_stack((test_idx,predictions)),
        columns=['id', 'target'])
    
    return sub

if __name__ == '__main__':
    submission = predict()
    submission['id'] = submission['id'].astype(int)
    submission.to_csv(f'models/{MODEL}.csv', index = False)


    


