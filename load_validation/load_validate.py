
#from data.training_data import
import pandas as pd
import json
from core.logger import Logger

class load_validate:
    def __init__(self,run_id,data_path):
        self.run_id = run_id
        self.training_data_path = data_path
        self.logger = Logger(self.run_id, 'TrainModel', 'training')
        #self.loadValidate = LoadValidate(self.run_id, self.data_path, 'training')

    def load(self):
        try:
            self.logger.info('Start of reading column names from new file!')
            df = pd.read_csv('./data/training_data/Data.csv')
            columns_name,column_length = self.schema_columns(df)
            return df,columns_name,column_length
        except ValueError:
            self.logger.exception('ValueError raised while Reading values From Schema')
            raise ValueError
        except KeyError:
            self.logger.exception('KeyError raised while Reading values  from new file!')
            raise KeyError
        except Exception as e:
            self.logger.exception('Exception raised while Reading values  from new file!: %s' % e)
            raise e



    def schema_columns(self,df):
        try:
            cn = list(df.columns)
            cl = len(df.columns)
            self.logger.info('End of reading column names from new file!')
            return cn,cl
        except ValueError:
            self.logger.exception('ValueError raised while Reading values  from new file!')
            raise ValueError
        except KeyError:
            self.logger.exception('KeyError raised while Reading values  from new file!')
            raise KeyError
        except Exception as e:
            self.logger.exception('Exception raised while Reading values  from new file!: %s' % e)
            raise e

    def column_length(self,col_nums):
        try:
            self.logger.info('Comparing new file columns with Json file columns!')
            with open("./load_validation/artifacts/r_columns.json", 'r') as f:
                col_json = json.load(f)['data_columns']
                if len(col_json) != col_nums:
                    self.logger.info('Columns not match, aborting training!')
                    return "Yes"
                else:
                    self.logger.info('Columns has been matched, we can proceed!')
                    return "No"
                    pass
        except ValueError:
            self.logger.exception('ValueError raised while matching columns from new file!')
            raise ValueError
        except KeyError:
            self.logger.exception('KeyError raised while matching columns from new file!')
            raise KeyError
        except Exception as e:
            self.logger.exception('Exception raised while matching columns from new file!: %s' % e)
            raise e

    def feature_engineering(self,df):
        try:
            self.logger.info('Imputation missing values in different features!')
            # Filling Gender null Values
            gend_mode = df['Gender'].mode()[0]
            df['Gender'] = df['Gender'].fillna(gend_mode)
            # Filling Married null values
            marrd_mode = df['Married'].mode()[0]
            df['Married'] = df['Married'].fillna(marrd_mode)
            # Filling Dependents null values
            df.loc[df['Dependents'] == '3+', 'Dependents'] = 3
            dep_mode = df['Dependents'].mode()[0]
            df['Dependents'] = df['Dependents'].fillna(dep_mode)
            # Filling Self Employed null values
            se_mode = df['Self_Employed'].mode()[0]
            df['Self_Employed'] = df['Self_Employed'].fillna(se_mode)
            # Filling Loan Amount null values
            mean_loanamount = df['LoanAmount'].median()
            df['LoanAmount'] = df['LoanAmount'].fillna(mean_loanamount)
            # Filling Loan Amount null values
            mean_loanterm = df['Loan_Amount_Term'].mode()[0]
            df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(mean_loanterm)
            # Filling credit history null values
            mode_crhist = df['Credit_History'].mode()[0]
            df['Credit_History'] = df['Credit_History'].fillna(mode_crhist)
            df['ApplicantIncome'] = df['ApplicantIncome'].astype(float)
            df['Dependents'] = df['Dependents'].astype(int)
            df['Total Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
            df = df.drop('Loan_ID', axis=1)
            self.logger.info('Imputation has been successfully completed!')
            return df
        except ValueError:
            self.logger.exception('ValueError raised while imputation missing values!')
            raise ValueError
        except KeyError:
            self.logger.exception('KeyError raised while imputation missing values!')
            raise KeyError
        except Exception as e:
            self.logger.exception('Exception raised while imputation missing values!: %s' % e)
            raise e















