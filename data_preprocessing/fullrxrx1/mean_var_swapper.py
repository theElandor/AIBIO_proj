import pandas as pd
source_path = '/work/h2020deciderficarra_shared/rxrx1/rxrx1_v2.1/metadata/fullmetadata_v1.csv'
destination_path = '/work/h2020deciderficarra_shared/rxrx1/rxrx1_v2.1/metadata/fullmetadata_v2.csv'
df = pd.read_csv(source_path)
df.insert(df.shape[1] -1,'tmp',df.pop('mean')) #this is the variance
df.insert(df.shape[1]-1,'mean',df.pop('variance')) #ths is the mean
df.insert(df.shape[1]-1,'variance',df.pop('tmp')) #renaming the variance
df.to_csv(destination_path)
