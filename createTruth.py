import pandas as pd
import numpy as np
import random
import math
import os

'''生成每一轮真实值，每一轮的真实值会发生变化'''

if __name__ == '__main__':
    fileNumber = 20
    for fileId in range(1,1 + fileNumber):
        truth = pd.read_csv("smallTruth.csv")
        for roundtime in range(0, 21):
            rate = random.randint(1,20)
            judge = random.uniform(0,1)
            if judge <0.5:
                rate = rate * -1
            result_truth = truth * (1 + rate/100)
            result = result_truth.iloc[:,1:]
            if judge > 0.5:
                result = result.apply(np.ceil)
            else:
                result = result.apply(np.floor)
            result = result.replace(to_replace=0, value=1)
            result.insert(0,"PatientId",range(1,1+len(result)))
            result_name = "file_" + str(fileId) + "/" + "RoundTruth/"+"truth_"+str(roundtime)+".csv"
            result.to_csv(result_name,index=False)