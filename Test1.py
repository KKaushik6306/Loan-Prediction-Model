
from load_validation.load_validate import load_validate
from core.config import Config
from model.model import Train_Model

config = Config()
run_id = config.get_run_id()
data_path = config.training_data_path
a1 = load_validate(run_id ,data_path)
mdf ,a ,b = a1.load()
Abort__ = a1.column_length(b)
if Abort__ != "Yes":
    df_ = a1.feature_engineering(mdf)
    trainmodel = Train_Model(run_id ,data_path)
    X_train, X_test, y_train, y_test = trainmodel.traintest_split(df_)
    algos = trainmodel.alogs_dict()
    model_name,accuracy,params = trainmodel.tune_model(X_train,y_train,algos)
    if model_name=="RandomForestClassifier":
        criterion = params['criterion']
        max_depth = params['max_depth']
        max_features=params['max_features']
        n_estimators=params['n_estimators']
        trainmodel.save_finalmodel(model_name,criterion,max_depth,max_features,n_estimators,X_train,y_train)
