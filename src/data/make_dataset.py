import yaml
import sys
import pathlib
from sklearn.model_selection import train_test_split
import pandas as pd 

def load_data(path):
    df=pd.read_csv(path)
    return df

def data_split(df,size,seed):
    train,test=train_test_split(df,test_size=size,random_state=seed)
    return train,test

def save_data(train,test,output_path):
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
    train,test=data_split(df,size,seed)
    save_data(train,test,output_path)

if __name__=="__main__":
    try:
        main()
    except:
        print("erorr occured")
    

