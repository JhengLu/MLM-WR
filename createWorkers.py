import time

import pandas as pd
import numpy as np
import random
import math

import os
'''创建worker的地点和属性信任度和accu表格'''
from multiprocessing import  Process
import parameter
class MyProcessAP (Process):
    def __init__(self, workernumber, folder_Id):
        super(MyProcessAP, self).__init__()
        self.worker_number = workernumber
        self.file_id = folder_Id

    def run(self):
        print ("开始线程：" + str(self.file_id))
        worker_number = self.worker_number  # need change
        fileId = self.file_id
        workerInfo_attr = pd.DataFrame(columns=('workerId', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI'))
        loc_number = parameter.locationNumber
        attribute_number = parameter.attributeNumber
        new_cols = ['workerId'] + [str(i) for i in range(1, loc_number + 1)]
        workerInfo_loc = pd.DataFrame(columns=new_cols)
        worker_number = parameter.workerNumber
        worker_id = 1
        worker_ratio = parameter.workerRatio
        first_part_number = int(worker_number * worker_ratio)
        second_part_number = int(worker_number * worker_ratio)
        last_part_number = worker_number - first_part_number - second_part_number
        # RoundTruth = pd.read_csv("smallTruth.csv")
        negativeRidge = parameter.negativeRidge # 单个百分之10

        for i in range(first_part_number):
            row_value = [worker_id]
            for attri_num in range(attribute_number):
                error_rate = random.uniform(0.15, 0.2)
                judge = random.uniform(0, 1)
                if judge < negativeRidge:
                    error_rate = error_rate * (-1)
                value_rate = 1 + error_rate
                row_value.append(value_rate)
            workerInfo_attr.loc[len(workerInfo_attr)] = row_value

            row_value = [worker_id]
            for loc_num in range(loc_number):
                error_rate = random.uniform(0.15, 0.2)
                judge = random.uniform(0, 1)
                if judge < negativeRidge:
                    error_rate = error_rate * (-1)
                value_rate = 1 + error_rate
                row_value.append(value_rate)
            workerInfo_loc.loc[len(workerInfo_loc)] = row_value
            worker_id += 1

        for i in range(second_part_number):
            row_value = [worker_id]
            for attri_num in range(attribute_number):
                error_rate = random.uniform(0.1, 0.15)
                judge = random.uniform(0, 1)
                if judge < negativeRidge:
                    error_rate = error_rate * (-1)
                value_rate = 1 + error_rate
                row_value.append(value_rate)
            workerInfo_attr.loc[len(workerInfo_attr)] = row_value

            row_value = [worker_id]
            for loc_num in range(loc_number):
                error_rate = random.uniform(0.1, 0.15)
                judge = random.uniform(0, 1)
                if judge < negativeRidge:
                    error_rate = error_rate * (-1)
                value_rate = 1 + error_rate
                row_value.append(value_rate)
            workerInfo_loc.loc[len(workerInfo_loc)] = row_value
            worker_id += 1

        for i in range(last_part_number):
            row_value = [worker_id]
            for attri_num in range(attribute_number):
                error_rate = random.uniform(0.05, 0.1)
                judge = random.uniform(0, 1)
                if judge < negativeRidge:
                    error_rate = error_rate * (-1)
                value_rate = 1 + error_rate
                row_value.append(value_rate)
            workerInfo_attr.loc[len(workerInfo_attr)] = row_value

            row_value = [worker_id]
            for loc_num in range(loc_number):
                error_rate = random.uniform(0.05, 0.1)
                judge = random.uniform(0, 1)
                if judge < negativeRidge:
                    error_rate = error_rate * (-1)
                value_rate = 1 + error_rate
                row_value.append(value_rate)
            workerInfo_loc.loc[len(workerInfo_loc)] = row_value
            worker_id += 1

        filename_info = "file_"+ str(fileId) + "/" + "workerInfo_attr_" + str(worker_number) + ".csv"
        workerInfo_attr.to_csv(filename_info, index=False)

        filename_info = "file_"+ str(fileId) + "/" + "workerInfo_loc_" + str(worker_number) + ".csv"
        workerInfo_loc.to_csv(filename_info, index=False)

        for workerId in range(1, worker_number + 1):
            attri_row = workerInfo_attr.loc[(workerInfo_attr['workerId'] == workerId)].iloc[:, 1:]
            loc_row = workerInfo_loc.loc[(workerInfo_loc['workerId'] == workerId)].iloc[:, 1:]
            loc_row = loc_row.T
            result = loc_row.dot(attri_row)
            result.insert(0, "PatientId", range(1, 1 + len(result)))
            filename = "file_"+ str(fileId) + "/" + "workerInfo/" + "workerInfo_" + str(workerId) + ".csv"
            result.to_csv(filename, index=False)

        sum_number = loc_number * attribute_number
        workerInfo_accu = pd.DataFrame(columns=("accuracy",))
        for workerId in range(1, worker_number + 1):
            filename = "file_"+ str(fileId) + "/" + "workerInfo/" + "workerInfo_" + str(workerId) + ".csv"
            workerInfo = pd.read_csv(filename)
            workerValue = workerInfo.iloc[:, 1:]
            Value = (workerValue - 1).abs()
            wrong_number = Value[Value > 0.25].count().sum()
            row_value = [(sum_number - wrong_number) / sum_number]
            workerInfo_accu.loc[len(workerInfo_accu)] = row_value
        InfoFilename = "file_" + str(fileId) + "/" + "workerInfo_accu_" + str(worker_number) + ".csv"
        workerInfo_accu.insert(0, "workerId", range(1, 1 + len(workerInfo_accu)))
        workerInfo_accu.to_csv(InfoFilename, index=False)

        print ("退出线程：" + str(self.file_id))


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


