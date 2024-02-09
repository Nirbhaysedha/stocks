import yaml
import random
import sys
from dvclive import Live
import pathlib
from sklearn.model_selection import train_test_split
import pandas as pd 


def load_data(path):
    df=pd.read_csv(path)
    return df

def clean_data(df):
    df=df.dropna()
    df=df.iloc[:,1:]
    return df

def data_split(df,size,seed):
    with Live(save_dvc_exp=True) as live:
        train,test=train_test_split(df,test_size=size,random_state=seed)
        live.log_param("train/test",1)
    return train,test

def save_data(train,test,output_path):
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path+'/train.csv',index=False)
    test.to_csv(output_path+'/test.csv',index=False)

def main():
    curr_dir=pathlib.Path(__file__)
    parent=curr_dir.parent.parent.parent
    yaml_path=parent.as_posix()+'/params.yaml'
    yaml_file=yaml.safe_load(open(yaml_path))['make_dataset']
    path=yaml_file['raw_data_path']
    size=yaml_file['size']
    seed=yaml_file['seed']
    output_path=yaml_file['output_path']


    df=load_data(path)
    df=clean_data(df)
    train,test=data_split(df,size,seed)
    save_data(train,test,output_path)

if __name__=="__main__":
    try:
        main()
    except:
        print("erorr occured")
    

