import pandas as pd
import os

def read_csv(fname, path=os.getcwd()):
    '''
    Reads a csv file and returns a pandas dataframe
    '''
    df = pd.read_csv(os.path.join(path,fname), header=None)
    df.columns = [fname.split('.')[0]+str(i) for i in df.columns]
    return df

def lstm_extract(fname):
    datapath = os.path.join(os.getcwd(), 'datasets', 'w_2_metric')
    files = os.listdir(datapath)
    csv_files = [f for f in files if f.split('.')[-1] == 'csv' ]
    df_total = None
    for f in csv_files:
        df = read_csv(f, path=datapath)
        if df_total is None:
            df_total = df
        else:
            df_total = df_total.join(df)
    data = df_total[fname]
    return data/data.max()
    #return data