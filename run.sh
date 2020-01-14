export TRAINING_DATA=input/train_folds.csv
export FOLD=0
export MODEL=$1 # command so one can input the variable to run

# run train.py along with the variables
python -m src.train