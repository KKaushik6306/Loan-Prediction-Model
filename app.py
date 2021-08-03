from flask import Flask, request, jsonify,render_template
import os
import pickle
import sys
import json
import numpy as np
from load_validation.load_validate import load_validate
from core.config import Config
from model.model import Train_Model
from prediction.Prediction import Prediction_Config
from core.config import Config
import warnings
warnings.filterwarnings('ignore')





#ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['Secret_Key'] = '123'

@app.route('/')
def get_indexpage():
    return render_template('index.html')

@app.route('/predict_loanstatus', methods=['POST'])
def predict_loanstatus():
    prediction = Prediction_Config()
    dep = request.form['dependent']
    appincome = request.form['applicantincome']
    coappincome = request.form['coapplicantincome']
    loanamount = request.form['loanamount']
    loanterm = request.form['loanamountterm']
    credithistory = request.form['credithistory']
    gender = request.form['Gender']
    married = request.form['Married']
    graduate = request.form['Graduate']
    employed = request.form['Employed']
    property = request.form['Property']
    a = prediction.Predict_LoanStatus(dep, appincome, coappincome, loanamount, loanterm, credithistory, gender, married, graduate, employed, property)
    #comp.Predict_LoanStatus(0, 7085, 0, 84, 360, 1, 'male', 'no', 'graduate', 'yes', 'semiurban')
    return a

@app.route('/training', methods=['POST'])
def train_model():
    try:
        config = Config()
        run_id = config.get_run_id()
        data_path = config.training_data_path
        a1 = load_validate(run_id, data_path)
        mdf, a, b = a1.load()
        Abort__ = a1.column_length(b)
        if Abort__ != "Yes":
            df_ = a1.feature_engineering(mdf)
            trainmodel = Train_Model(run_id, data_path)
            X_train, X_test, y_train, y_test = trainmodel.traintest_split(df_)
            algos = trainmodel.alogs_dict()
            model_name, accuracy, params = trainmodel.tune_model(X_train, y_train, algos)
            if model_name == "RandomForestClassifier":
                criterion = params['criterion']
                max_depth = params['max_depth']
                max_features = params['max_features']
                n_estimators = params['n_estimators']
                trainmodel.save_finalmodel(model_name, criterion, max_depth, max_features, n_estimators, X_train,y_train)
                return "Training successfull! and its RunID is : "+str(run_id)
    except ValueError:
        return "Error Occurred! %s" % ValueError
    except KeyError:
        return "Error Occurred! %s" % KeyError
    except Exception as e:
        return "Error Occurred! %s" % e






if __name__ == "__main__":
    app.run(debug=True)


