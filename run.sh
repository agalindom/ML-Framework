export TRAINING_DATA=input/train_folds.csv
export TEST_DATA=input/test.csv

export MODEL=$1 # command so one can input the variable to run

# FOLD=0 python -m src.train
# FOLD=1 python -m src.train
# FOLD=2 python -m src.train
# FOLD=3 python -m src.train
# FOLD=4 python -m src.train
# run train.py along with the variables
python -m src.predict