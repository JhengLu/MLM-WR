import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import math



def changeToZero(x):
    return 0*x

def getRandom():
    return random.uniform(0,1)

import parameter

if __name__ == '__main__':
    truth = pd.read_csv("smallTruth.csv")
    worker_number = parameter.workerNumber
    task_number = parameter.taskNumber
    task = truth.copy(deep=True)
    part_task = task.iloc[:,1:]
    judge = 0.1
    task.iloc[:,1:] = part_task.apply(changeToZero) #make all of them to become zero
    task_choosed = task.sample(n=task_number)
    for index, row in task_choosed.iterrows():
        if getRandom() > judge:
            task_choosed.loc[index, 'Pregnancies'] = 1
        if getRandom() > judge:
            task_choosed.loc[index, 'Glucose'] = 1
        if getRandom() > judge:
            task_choosed.loc[index, 'BloodPressure'] = 1
        if getRandom() > judge:
            task_choosed.loc[index, 'SkinThickness'] = 1
        if getRandom() > judge:
            task_choosed.loc[index, 'BMI'] = 1
    filename = "task_" + str(task_number) + ".csv"
    task_choosed.to_csv(filename,index=False)
