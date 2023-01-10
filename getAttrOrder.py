import copy

import pandas as pd
import numpy as np
import random

def getAttrOrderF(roundtime,fileId):
    attrScoresFile = "file_" + str(fileId) + "/" + "workerAttrScores.csv"
    attrScores = pd.read_csv(attrScoresFile)
    attrRoundScoresFile = "file_" + str(fileId) + "/" + "ScoreRecord/workerAttrScores_"+ str(roundtime - 1) +".csv"
    attrScores.to_csv(attrRoundScoresFile,index=False)  #这是上一轮次更新的值

    attrScores = attrScores.iloc[1:, :]  # 去掉序号为0的
    attrScores.insert(0, "workerId", range(1, 1 + len(attrScores)))
    attrScores['rand_col'] = np.random.randint(1, 1 + len(attrScores), attrScores.shape[0])
    attrOrder = pd.DataFrame()
    columnName = attrScores.columns[1:-1]  # 去掉workerId,rand_col

    for column in columnName:
        tempScores = attrScores.sort_values(by=[column, "rand_col"], ascending=[False, False])  # descend
        tempScores = tempScores.reset_index(drop=True)  #这个不能少，不然会每一列都一样了
        attrOrder[column] = copy.deepcopy(tempScores['workerId'])
    attrOrderFile = "file_" + str(fileId) + "/" + "workerOrder/workerAttrOrder.csv"
    attrOrder.to_csv(attrOrderFile, index=False)
    attrOrderCheckFile = "file_" + str(fileId) + "/" + "workerOrder/workerAttrOrder_"+ str(roundtime - 1) +".csv"
    attrOrder.to_csv(attrOrderCheckFile, index=False)


    return attrOrder

if __name__ == '__main__':
    getAttrOrderF(1,1)
#     attrScoresFile = "workerAttrScores.csv"
#     attrScores = pd.read_csv(attrScoresFile)
#     attrScores = attrScores.iloc[1:,:]  #去掉序号为0的
#     attrScores.insert(0,"workerId",range(1,1+len(attrScores)))
#     attrScores['rand_col'] = np.random.randint(1, 1 + len(attrScores), attrScores.shape[0])
#     attrOrder = pd.DataFrame()
#     columnName = attrScores.columns[1:-1] # 去掉workerId,rand_col
#     for column in columnName:
#         tempScores = attrScores.sort_values(by=[column,"rand_col"], ascending=[False, False])  # descend
#         tempScores = tempScores.reset_index(drop=True)
#         attrOrder[column] = copy.deepcopy(tempScores['workerId'])
#     attrOrderFile = "workerAttrOrder.csv"
#     attrOrder.to_csv(attrOrderFile,index=False)



