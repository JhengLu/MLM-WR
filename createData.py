import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import preprocessing
'''产生指定规格的真实数据'''
if __name__ == '__main__':
    df = pd.read_csv('diabetes.csv')
    # left_df = df.iloc[:50, :6].drop('Insulin', 1) # big RoundTruth
    left_df = df.iloc[:200, :6].drop('Insulin', 1) # small RoundTruth 是安排的任务
    left_df = left_df.replace(to_replace=0, value=1)
    left_df.insert(0, 'PatientId', range(1, 1 + len(left_df)))
    print(left_df)
    # left_df.to_csv("bigTruth.csv", index=False) #without source column, while init_data has the source column
    left_df.to_csv("smallTruth.csv", index=False)