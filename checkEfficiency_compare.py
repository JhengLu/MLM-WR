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
import parameter



from multiprocessing import Process

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
            task_number = locationNumber
            filename = "task_" + str(task_number) + ".csv"
            task = pd.read_csv(filename)
            roundTruthFile = "file_" + str(fileId) + "/" + "RoundTruth/truth_" + str(roundtime) + ".csv"
            roundTruth = pd.read_csv(roundTruthFile)
            errorAll = 0

            if roundtime == 1:
                normalPlace = list(range(1, 1 + locationNumber))
                normalArrangementFile = "file_" + str(fileId) + "/" + "TFARworkerArrange/normalArrange_" + str(roundtime) + ".csv"
                normalArrangement = pd.read_csv(normalArrangementFile)
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
                        # thisTruth = thisTruth.mul(adjTask)
                        errorCollectDf = (abs(workerData - thisTruth) / thisTruth)
                        errorCollectDf = errorCollectDf.mul(adjTask)
                        errorCollectDf = errorCollectDf.iloc[:, 1:]
                        thisError = errorCollectDf.to_numpy().sum()
                        locError = locError + thisError

                    locError = locError / len(workerList)
                    errorAll = errorAll + locError

            else:
                trustArrangementFile = "file_" + str(fileId) + "/" + "TFARworkerArrange/trustArrange_" + str(roundtime) + ".csv"
                trustArrangement = pd.read_csv(trustArrangementFile)
                trustfulPlace = list(trustArrangement.columns)
                trustfulPlace = list(map(int, trustfulPlace))
                normalPlace = list(range(1, 1 + locationNumber))
                normalArrangementFile = "file_" + str(fileId) + "/" + "TFARworkerArrange/normalArrange_" + str(roundtime) + ".csv"
                normalArrangement = pd.read_csv(normalArrangementFile)

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
                    # thisTruth = thisTruth.mul(adjTask)
                    errorCollectDf = (abs(workerData - thisTruth) / thisTruth)
                    errorCollectDf = errorCollectDf.mul(adjTask)
                    errorCollectDf = errorCollectDf.iloc[:,1:]
                    thisError = errorCollectDf.to_numpy().sum()
                    errorAll = errorAll + thisError  # 不算普通worker的时候

                    # 算普通worker的时候
                    # locError = locError + thisError  # 放在后面一起算

                # 安排的普通worker也算的话
                #    workerList = list(normalArrangement[str(patientid)])
                #    thisTask = task.loc[(task['PatientId'] == patientid)]
                #    thisTask = thisTask.reset_index(drop=True)
                #    adjTask = thisTask[thisTask > 1] = 1  # 使得task中的第一项变成1
                #    if 0 in workerList:
                #        zeroIndex = workerList.index(0)
                #        workerList = workerList[:zeroIndex]
                #
                #
                #    for workerId in workerList:
                #        workerFile = "workerData/round_" + str(roundtime) + "/worker_" + str(workerId) + ".csv"
                #        thisData = pd.read_csv(workerFile)
                #        tempData = thisData.loc[(thisData['PatientId'] == patientid)]
                #        workerData = tempData.mul(adjTask)
                #        thisTruth = roundTruth.loc[(roundTruth['PatientId'] == patientid)]
                #        thisTruth = thisTruth.mul(adjTask)
                #        errorCollectDf = (abs(workerData - thisTruth) / thisTruth).iloc[:, 1:]
                #        thisError = errorCollectDf.to_numpy().sum()
                #        locError = locError + thisError
                #    fenmu = (len(workerList) + 1)
                #    errorAll = errorAll + (locError / fenmu )
                #    算普通worker的时候

            errorAll = errorAll / locationNumber
            errorAllList.append(errorAll)

        errorAllListFile = "error/"+"errorAll_compare"+ str(fileId) +".csv"
        errorAllDf = pd.DataFrame(errorAllList)
        errorAllDf.columns = ['Average Data Quality Differences']
        errorAllDf.insert(0, 'Round', range(1, 1 + len(errorAllDf)))
        errorAllDf.to_csv(errorAllListFile, index=False)

        print("退出线程：" + str(self.file_id))




'''这边是对比的算法'''
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


  # 一种思路是根据地点来安排，一种思路是属性只根据上一轮的来安排
  # 检验的时候一种是只检验安排的指定worker，一种是检验安排的所有worker