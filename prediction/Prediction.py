import pickle
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class Prediction_Config:
    def __init__(self):
        self.value = "Karan"

    def Predict_LoanStatus(self,Dep, Appincome, Coappincome, LoanAmount, LoanTerm, CrediHistory, gend_name, married_status,
                       educ_status, \
                       Emp_Status, Prop_Status):
        self.value = 0
        with open("./prediction/artifacts/columns.json", 'r') as f:
            X = json.load(f)['data_columns']
        with open("./prediction/artifacts/Loan_Pridiction", 'rb') as f:
            final_model = pickle.load(f)
        gend_index =X.index('gender_' + gend_name)
        married_index =X.index('married_' + married_status)
        educ_index = X.index('education_' + educ_status)
        Emp_index = X.index('self_employed_' + Emp_Status)
        Prop_index = X.index('property_area_' + Prop_Status)
        x = np.zeros(len(X))
        x[0] = Dep
        x[1] = Appincome
        x[2] = Coappincome
        x[3] = LoanAmount
        x[4] = LoanTerm
        x[5] = CrediHistory
        x[6] = Appincome + Coappincome
        if gend_index >= 0:
            x[gend_index] = 1
        if married_index >= 0:
            x[married_index] = 1
        if educ_index >= 0:
            x[educ_index] = 1
        if Emp_index >= 0:
            x[Emp_index] = 1
        if Prop_index >= 0:
            x[Prop_index] = 1
        val_ = final_model.predict([x])[0]
        if val_ == 1:
            return 'Yes, you are eligible for Loan!'
        else:
            return 'No, you are not eligible for loan!'











