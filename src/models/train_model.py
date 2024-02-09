import pathlib
import sys
import yaml
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from dvclive import Live


def data_split(train,test,target,output_path):
    train=pd.read_csv(train)
    test=pd.read_csv(test)
    x_train=train.drop(columns=target)
    y_train=train[target]
    x_test=test.drop(columns=target)
    y_test=test[target]
    x_train.to_csv(output_path+'/x_train.csv',index=False)
    x_test.to_csv(output_path+'/x_test.csv',index=False)
    y_train.to_csv(output_path+'/y_train.csv',index=False)
    y_test.to_csv(output_path+'/y_test.csv',index=False)

def train_model(x_train,y_train,x_test,y_test,model,params_grid,cv,n_jobs,verbose,scoring):
    x_train=pd.read_csv(x_train)
    y_train=pd.read_csv(y_train)
    x_test=pd.read_csv(x_test)
    y_test=pd.read_csv(y_test)
    Model = model
    m= GridSearchCV(estimator=eval(model), param_grid=params_grid, cv=cv, n_jobs=n_jobs, verbose=verbose, scoring=scoring)
    m.fit(x_train,y_train)
    best_model = m.best_estimator_
    best_parameters = m.best_params_
    y_pred=m.predict(x_test)
    accuracy=r2_score(y_test,y_pred)
    return accuracy,best_model,best_parameters

def main():
    curr_dir=pathlib.Path(__file__)
    parent_dir=curr_dir.parent.parent.parent
    yaml_file_path='./params.yaml'
    yaml_file=yaml.safe_load(open(yaml_file_path))['train_model']
    train_path=parent_dir.as_posix()+'/data/processed/train.csv'
    test_path=parent_dir.as_posix()+'/data/processed/test.csv'
    output_path='./data/interim/'
    target=yaml_file['target']
    data_split(train_path,test_path,target,output_path)

    x_train=yaml_file['x_train_path']
    y_train=yaml_file['y_train_path']
    model=yaml_file['model']
    x_test=yaml_file['x_test_path']
    y_test=yaml_file['y_test_path']
    params_grid=yaml_file['params_grid']
    cv=yaml_file['cv']
    n_jobs=yaml_file['n_jobs']
    verbose=yaml_file['verbose']
    scoring=yaml_file['scoring']
    with Live(save_dvc_exp=True) as live:
            accuracy,best_model,best_parameters=train_model(x_train,y_train,x_test,y_test,model,params_grid,cv,n_jobs,verbose,scoring)
            live.log_param("best model",best_model)
            live.log_param("best parameters ",best_parameters)
            live.log_metric("r2 score",accuracy)

    joblib.dump(model,parent_dir.as_posix()+ '/model.joblib')

if __name__=="__main__":
    main()
    
