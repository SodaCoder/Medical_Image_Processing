import pandas as pd
from sklearn.utils import shuffle
from math import floor

def createImbalanceDataset(srcPath, dstPath):
    df = pd.read_csv(srcPath)
    df_noncovid = df[df['labels'] != 2]
    df_covid = df[df['labels'] == 2]
    df_covid = shuffle(df_covid)
    num_rows, _ = df_covid.shape
    df_covid = df_covid.head(floor(num_rows/3))
    num_rows, _ = df_covid.shape
    print(num_rows)
    df_new = pd.concat([df_noncovid, df_covid], axis=0)
    df_new.to_csv(dstPath, columns = ['images', 'labels'], index = False)
    print('Writing to csv file is done!...')

createImbalanceDataset('./train1.csv', './train3.csv')
createImbalanceDataset('./val1.csv', './val3.csv')
#createImbalanceDataset('./test1.csv', './test3.csv')
