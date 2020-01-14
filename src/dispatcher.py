from sklearn import ensemble

MODELS = {
    'randomForest' : ensemble.RandomForestClassifier(
        n_jobs=-1, verbose=2,n_estimators = 200),
    'extraTrees' : ensemble.ExtraTreesClassifier(
        n_jobs=-1, verbose=2,n_estimators = 200),
    'gradientBoosting': ensemble.GradientBoostingClassifier(
        verbose=2,n_estimators = 200)
}