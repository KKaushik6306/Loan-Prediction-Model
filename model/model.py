
from core.logger import Logger
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score


class Train_Model:
    def __init__(self, run_id, data_path):
        self.run_id = run_id
        self.training_data_path = data_path
        self.logger = Logger(self.run_id, 'TrainModel', 'training')
        # self.loadValidate = LoadValidate(self.run_id, self.data_path, 'training')

    def traintest_split(self, dataframe):
        self.logger.info('Performing train & test split on main dataframe!')
        ddf = dataframe.drop('Loan_Status', axis=1)
        dataframe.loc[dataframe['Loan_Status'] == 'Y', 'Loan_Status'] = 1
        dataframe.loc[dataframe['Loan_Status'] == 'N', 'Loan_Status'] = 0
        y = dataframe['Loan_Status']
        X = pd.get_dummies(ddf)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        self.logger.info('Train & test split on main dataframe has been completed!')
        return X_train, X_test, y_train, y_test

    def alogs_dict(self1):
        return {
                    'RandomForestClassifier':{
                        'model':RandomForestClassifier(),
                        'param':{
                            'n_estimators':[10,50,100,130],
                            'criterion':['gini','entropy'],
                            'max_depth': range(2,4,1),
                            'max_features':['auto','log2']
                        }
                    },
                        'XGBClassifier':{
                        'model':XGBClassifier(objective='binary:logistic'),
                        'param':{
                            'learning_rate':[0.5,0.1,0.01,0.001],
                            'max_depth': [3,5,10,20],
                            'n_estimators':[10,50,100,200]
                        }
                    },'desicion_tree':{
                            'model': DecisionTreeRegressor(),
                            'param':{
                                'criterion':['mse','friedman_mse'],
                                'splitter':['best','random']
                            }
                        }
                }

    def tune_model(self,X_train,y_train,algos):
        try:
            self.logger.info('Model Hypertuning is performing!')
            scores = []
            cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
            for algo_name, config in algos.items():
                gs = GridSearchCV(config['model'], config['param'], cv=cv, return_train_score=False)
                gs.fit(X_train, y_train)
                scores.append({
                    'model': algo_name,
                    'best_score': gs.best_score_,
                    'best_params': gs.best_params_
                })
            hdf = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
            max_score = hdf['best_score'].max()
            selected_model = hdf.loc[hdf['best_score'] == max_score]
            model_name = selected_model.iloc[0, 0]
            accuracy = selected_model.iloc[0, 1]
            params = selected_model.iloc[0, 2]
            self.logger.info('Model Hypertuning has been completed!')
            return  model_name,accuracy,params
        except ValueError:
            self.logger.exception('ValueError raised while tuning the module!')
            raise ValueError
        except KeyError:
            self.logger.exception('KeyError raised while tuning the module!')
            raise KeyError
        except Exception as e:
            self.logger.exception('Exception raised while tuning the module!: %s' % e)
            raise e

    def save_finalmodel(self,model_name,criterion,max_depth,max_features,n_estimators,X_train,y_train):
        try:
            self.logger.info('Saving final model!')
            if model_name == "RandomForestClassifier":
                final_model = RandomForestClassifier(criterion='entropy', max_depth=3, max_features='log2', n_estimators=50)
                final_model.fit(X_train, y_train)
                with open('./prediction/artifacts/Loan_Pridiction', 'wb') as f:
                    pickle.dump(final_model, f)
                    self.logger.info('Final model has been saved in prediction folder!')
        except ValueError:
            self.logger.exception('ValueError raised while saving the final model!')
            raise ValueError
        except KeyError:
            self.logger.exception('KeyError raised while saving the final model!')
            raise KeyError
        except Exception as e:
            self.logger.exception('Exception raised while saving the final model!: %s' % e)
            raise e