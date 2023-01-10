import time

import pandas as pd
import numpy as np
import random
import math
import os

from multiprocessing import Process
import parameter
from tqdm import tqdm

class MyProcessAP(Process):
    def __init__(self, workernumber, folder_Id):
        super(MyProcessAP, self).__init__()
        self.worker_number = workernumber
        self.file_id = folder_Id

    def run(self):
        print("开始线程：" + str(self.file_id))
        worker_number = self.worker_number  # need change
        fileId = self.file_id
        for roundtime in tqdm(range(1, 21)):
            for workerId in range(1, worker_number + 1):
                workerInfoFile = "file_" + str(fileId) + "/" + "workerInfo/" + "workerInfo_" + str(workerId) + ".csv"
                truthFile = "file_" + str(fileId) + "/" + "RoundTruth/" + "truth_" + str(roundtime) + ".csv"
                truth = pd.read_csv(truthFile)
                workerInfo = pd.read_csv(workerInfoFile)
                Info = workerInfo.iloc[:, 1:]
                data = truth.iloc[:, 1:]
                workerData = Info * data
                workerData.insert(0, 'PatientId', range(1, 1 + len(workerData)))
                # workerData['source'] = workerId
                dataFile = "file_" + str(fileId) + "/" + "workerData/round_" + str(roundtime) + "/worker_" + str(workerId) + ".csv"
                workerData.to_csv(dataFile, index=False)

        print("退出线程：" + str(self.file_id))








'''建立worker在每一轮的数据'''
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



# class MyProcessAP(Process):
#     def __init__(self, workernumber, folder_Id):
#         super(MyProcessAP, self).__init__()
#         self.worker_number = workernumber
#         self.file_id = folder_Id
#
#     def run(self):
#         print("开始线程：" + str(self.file_id))
#         worker_number = self.worker_number  # need change
#         fileId = self.file_id
#         for roundtime in range(1, 21):
#             process_list = []
#             p = MyProcessR(worker_number, fileId,roundtime)
#             p.start()
#             process_list.append(p)
#
#         # Wait all threads to finish.
#         for t in process_list:
#             t.join()
#
#         print("退出线程：" + str(self.file_id))
#
# class MyProcessR(Process):
#     def __init__(self, workernumber, folder_Id,round_time):
#         super(MyProcessR, self).__init__()
#         self.worker_number = workernumber
#         self.file_id = folder_Id
#         self.round = round_time
#
#     def run(self):
#         print("开始线程：" + str(self.file_id))
#         worker_number = self.worker_number  # need change
#         fileId = self.file_id
#         roundtime = self.round
#         for workerId in range(1, worker_number + 1):
#             workerInfoFile = "file_" + str(fileId) + "/" + "workerInfo/" + "workerInfo_" + str(workerId) + ".csv"
#             truthFile = "file_" + str(fileId) + "/" + "RoundTruth/" + "truth_" + str(roundtime) + ".csv"
#             truth = pd.read_csv(truthFile)
#             workerInfo = pd.read_csv(workerInfoFile)
#             Info = workerInfo.iloc[:, 1:]
#             data = truth.iloc[:, 1:]
#             workerData = Info * data
#             workerData.insert(0, 'PatientId', range(1, 1 + len(workerData)))
#             # workerData['source'] = workerId
#             dataFile = "file_" + str(fileId) + "/" + "workerData/round_" + str(roundtime) + "/worker_" + str(workerId) + ".csv"
#             workerData.to_csv(dataFile, index=False)
#
#         print("退出线程：" + str(self.round))
#
#
#
#
#
#
#
# '''建立worker在每一轮的数据'''
# if __name__ == '__main__':
#     start_time = time.time()
#     process_list = []
#     workernumber = parameter.workerNumber
#     for fileid in range(1, 2):
#         p = MyProcessAP(workernumber, fileid)
#         p.start()
#         process_list.append(p)
#
#     # Wait all threads to finish.
#     for t in process_list:
#         t.join()
#     print("--- %s seconds ---" % (time.time() - start_time))
#
#
#
#
#
#
