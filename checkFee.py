import time

import pandas as pd
import numpy as np
import random
import math
import tqdm

def getCount(listOfElems, cond = None):
    'Returns the count of elements in list that satisfies the given condition'
    if cond:
        count = sum(cond(elem) for elem in listOfElems)
    else:
        count = len(listOfElems)
    return count

from collections.abc import Iterable

def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x

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
        worker_ratio = 0.2
        workerFee = 10
        UAVFee = 100
        first_part_number = int(worker_number * worker_ratio)
        second_part_number = int(worker_number * worker_ratio)
        last_part_number = worker_number - first_part_number - second_part_number
        feeRecord = pd.DataFrame()
        for roundtime in tqdm(range(1, 21)):
            totalFee = 0
            normalArrangeFile = "file_" + str(fileId) + "/" + "workerArrange/normalArrange_" + str(roundtime) + ".csv"
            normalArrange = pd.read_csv(normalArrangeFile)
            trustArrangementFile = "file_" + str(fileId) + "/" + "workerArrange/trustArrange_" + str(roundtime) + ".csv"
            try:
                trustArrangement = pd.read_csv(trustArrangementFile)
            except pd.errors.EmptyDataError:
                trustArrangement = pd.DataFrame()
            normalWorkers = normalArrange.values.tolist()
            normalWorkers = list(flatten(normalWorkers))
            normalWorkers = [i for i in normalWorkers if i != 0]
            trustWorkers = trustArrangement.values.tolist()
            trustWorkers = list(flatten(trustWorkers))
            arrangeWorkers = normalWorkers + trustWorkers
            arrangeWorkersNumber = len(arrangeWorkers)
            totalFee = UAVFee * parameter.UAVcheckNumber + workerFee * arrangeWorkersNumber
            feeRecord[roundtime] = [totalFee]

        recordFile = "Accuracy/fee_"+ str(fileId) +".csv"
        feeRecord.to_csv(recordFile, index=False)

        print("退出线程：" + str(self.file_id))

# 这边检测的是雇佣的workers中可信workers的数量
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


