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
            if len(roundTruth) == 0:
                print("empty")
            errorAll = 0
            workerList = list(range(1,1+worker_number))
            for workerId in workerList:
                workerFile = "file_" + str(fileId) + "/" + "workerData/round_" + str(roundtime) + "/worker_" + str(workerId) + ".csv"
                thisData = pd.read_csv(workerFile)
                if len(thisData) == 0:
                    print("empty")



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