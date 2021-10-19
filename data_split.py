import pandas as pd
import numpy as np
import networkx as nx

for i in range(10):
    data = pd.read_csv('data/neural/neural.csv')
    # test
    samdata = data.sample(frac=0.1)
    # others for train
    data2 = data.append(samdata)
    data2 = data2.drop_duplicates(keep=False)
    samdata.to_csv('data/neural/neuralsam'+str(i)+'.csv', index=False)
    data2.to_csv('data/neural/neural2'+str(i), sep='\t', index=False, header=False)
    data2.to_csv('data/neural/neural_numpy'+str(i)+'.csv', index=False)

