import copy

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import math
import time
from multiprocessing import Process
from MergeLoc import MergeLocF
from getAttrOrder import getAttrOrderF
from random import sample
from multiprocessing import Process
import parameter

class MyProcessAP(Process):
    def __init__(self, workernumber, folder_Id):
        super(MyProcessAP, self).__init__()
        self.worker_number = workernumber
        self.file_id = folder_Id

    def run(self):
        print("开始线程：" + str(self.file_id))
        worker_number = self.worker_number  # need change
        fileId = self.file_id
        errorAllList = []
        locationNumber = parameter.locationNumber
        for roundtime in tqdm(range(1, 21)):
            normalArrangementFile = "file_" + str(fileId) + "/" + "workerArrange/normalArrange_" + str(roundtime) + ".csv"
            normalArrangement = pd.read_csv(normalArrangementFile)
            trustArrangementFile = "file_" + str(fileId) + "/" + "workerArrange/trustArrange_" + str(roundtime) + ".csv"
            UAV_placeFile = "file_" + str(fileId) + "/" + "workerArrange/UAVArrange_" + str(roundtime) + ".csv"
            UAV_place_df = pd.read_csv(UAV_placeFile)
            UAV_place = list(UAV_place_df.iloc[:, 0])
            try:
                trustArrangement = pd.read_csv(trustArrangementFile)
            except pd.errors.EmptyDataError:
                trustArrangement = pd.DataFrame()

            trustfulPlace = []
            if len(trustArrangement) != 0:
                trustfulPlace = list(trustArrangement.columns)
                trustfulPlace = list(map(int, trustfulPlace))
            else:
                trustfulPlace = []

            normalPlace = list(set(list(range(1, 1 + locationNumber))) - set(trustfulPlace))
            normalPlace = list(set(normalPlace) - set(UAV_place))  # 普通worker采集的数据才作为平台结果

            fenmu = len(normalPlace) + len(trustfulPlace) # 分母除法的时候不应该算上UAV的
            task_number = locationNumber
            filename = "task_" + str(task_number) + ".csv"
            task = pd.read_csv(filename)
            roundTruthFile = "file_" + str(fileId) + "/" + "RoundTruth/truth_" + str(roundtime) + ".csv"
            roundTruth = pd.read_csv(roundTruthFile)
            errorAll = 0

            for patientid in normalPlace:
                workerList = list(normalArrangement[str(patientid)])
                thisTask = task.loc[(task['PatientId'] == patientid)]
                thisTask = thisTask.reset_index(drop=True)
                thisTask[thisTask > 1] = 1  # 使得task中的第一项变成1
                adjTask = copy.deepcopy(thisTask)
                if 0 in workerList:
                    zeroIndex = workerList.index(0)
                    workerList = workerList[:zeroIndex]

                locError = 0
                for workerId in workerList:
                    workerFile = "file_" + str(fileId) + "/" + "workerData/round_" + str(roundtime) + "/worker_" + str(workerId) + ".csv"
                    thisData = pd.read_csv(workerFile)
                    tempData = thisData.loc[(thisData['PatientId'] == patientid)]
                    tempData = tempData.reset_index(drop=True)
                    workerData = copy.deepcopy(tempData)
                    thisTruth = roundTruth.loc[(roundTruth['PatientId'] == patientid)]
                    thisTruth = thisTruth.reset_index(drop=True)
                    errorCollectDf = (abs(workerData - thisTruth) / thisTruth)
                    errorCollectDf = errorCollectDf.mul(adjTask)
                    errorCollectDf = errorCollectDf.iloc[:, 1:]
                    thisError = errorCollectDf.to_numpy().sum()
                    locError = locError + thisError

                locError = locError / len(workerList)
                errorAll = errorAll + locError

            # errorAll = errorAll / locationNumber
            # errorAllList.append(errorAll)

            for patientid in trustfulPlace:
                locError = 0
                patientid = int(patientid)
                thisTask = task.loc[(task['PatientId'] == patientid)]
                thisTask = thisTask.reset_index(drop=True)
                thisTask[thisTask > 1] = 1  # 使得task中的第一项变成1
                adjTask = copy.deepcopy(thisTask)

                workerId = trustArrangement[str(patientid)].values[0]
                workerFile = "file_" + str(fileId) + "/" + "workerData/round_" + str(roundtime) + "/worker_" + str(workerId) + ".csv"
                thisData = pd.read_csv(workerFile)
                tempData = thisData.loc[(thisData['PatientId'] == patientid)]
                tempData = tempData.reset_index(drop=True)
                workerData = copy.deepcopy(tempData)

                thisTruth = roundTruth.loc[(roundTruth['PatientId'] == patientid)]
                thisTruth = thisTruth.reset_index(drop=True)
                errorCollectDf = (abs(workerData - thisTruth) / thisTruth)
                errorCollectDf = errorCollectDf.mul(adjTask)
                errorCollectDf = errorCollectDf.iloc[:, 1:]

                thisError = errorCollectDf.to_numpy().sum()
                errorAll = errorAll + thisError  # 就一个所以不用除以一个数字

            errorAll = errorAll / fenmu
            errorAllList.append(errorAll)

        errorAllListFile = "error/errorAll_"+ str(fileId) +".csv"
        errorAllDf = pd.DataFrame(errorAllList)
        errorAllDf.columns = ['Average Data Quality Differences']
        errorAllDf.insert(0, 'Round', range(1, 1 + len(errorAllDf)))
        errorAllDf.to_csv(errorAllListFile, index=False)

        print("退出线程：" + str(self.file_id))


# 明天修改参数引用
'''这边是不设置均值，原来论文的方法'''
if __name__ == '__main__':
    start_time = time.time()
    process_list = []
    workernumber = parameter.workerNumber

    for fileid in range(1, 21):
        p = MyProcessAP(workernumber, fileid)
        p.start()
        process_list.append(p)

    # Wait all threads to finish.
    for t in process_list:
        t.join()
    print("--- %s seconds ---" % (time.time() - start_time))
