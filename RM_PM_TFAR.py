import json
import time

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from multiprocessing import  Process
import copy
from random import sample
import sys, os
from IntegrateLocation import LocTrust_PSO_Thread
from IntegrateAttribute import AttrTrust_PSO_Thread
from TFARpso_2000_task_200 import TFARtaskAssignment
import parameter

def getCount(listOfElems, cond = None):
    'Returns the count of elements in list that satisfies the given condition'
    if cond:
        count = sum(cond(elem) for elem in listOfElems)
    else:
        count = len(listOfElems)
    return count

def condition(x):
  return x > 0.25

def loc_condition(x):
    return x > 0.25

class worker: #存储每个worker的workerId和申请的愿意去采集数据的地点
    def __init__(self,workerid,visLoc):
        self.workerId = workerid
        self.location = visLoc


class locWillVisit: #定义这个地点愿意前往的worker有哪些
    def __init__(self,patientid,visLoc =None):
        self.patientId = patientid
        if visLoc:
            self.visitorList = visLoc
        else:
            self.visitorList = []


class locArrVisit: #定义这个地点安排前往的worker有哪些
    def __init__(self,patientid):
        self.patientId = patientid
        self.visitorList = []


class platform:
    def __init__(self):
        self.loc_wk = None


def attriTimeout(score,fai):
    return max(score - fai, 0)

# 这个是和论文设计的进行对比，设计思路是利用TFAR的方法，但是没有使用不断检验可信workers的过程




class MyProcessRM(Process):
    def __init__(self, workernumber, folder_Id):
        super(MyProcessRM, self).__init__()
        self.worker_number = workernumber
        self.file_id = folder_Id

    def run(self):
        print("开始线程：" + str(self.file_id))
        worker_number = self.worker_number  # need change
        fileId = self.file_id
        workernumber = parameter.workerNumber
        first_part_number = int(workernumber * 0.2)
        second_part_number = int(workernumber * 0.2)
        last_part_number = workernumber - first_part_number - second_part_number
        workerInfo = pd.read_csv("file_" + str(fileId) + "/" + "workerInfo_accu_" + str(workernumber) + ".csv")
        workerId_data = workerInfo['workerId']
        location_number = parameter.locationNumber
        rounds = 20
        rou = 4
        loc_rou = 4
        attrRou = 4
        time_co = 0.003
        attrTime_co = 0.003
        locTime_co = 0.003
        attribute_number = parameter.attributeNumber

        vice_point = 0.7  # 总体信任度check 点
        vice_loc_point = 0.6  # 地点信任度check 点
        vice_attr_point = 0.7  # 属性信任度check 点
        task_number = parameter.taskNumber
        filename = "task_" + str(task_number) + ".csv"
        task = pd.read_csv(filename)
        attriName = list(task.columns)[1::]

        for round in range(1, 21):
            truthFile = "file_" + str(fileId) + "/" + "RoundTruth/truth_" + str(round) + ".csv"  # 每一轮的truth都是不一样的
            truth = pd.read_csv(truthFile)
            if round == 1:  # 第1轮的时候是随机分配的
                # 用来记录的是workers之间的相对偏序关系,横坐标是地点
                RelLocOrder = np.zeros((workerId_data.size, location_number))
                RelAttrOrder = np.zeros((workerId_data.size, attribute_number))
                wkStructList = []  # 存储的是所有worker的结构体，结构体里面含有愿意去哪些地方的信息
                locWillStructList = []  # 存储每个地点有哪些worker愿意前往的结构体，注意是从0开始
                locArrStructList = []  # 存储每个地点安排哪些worker前往的结构体，注意是从0开始
                # 对于两个list进行结构体初始化，每个list里面存储的都是结构体，结构体里面再存储哪些是愿意去的
                for patientid in range(1, location_number + 1):
                    locArrStructList.append(locArrVisit(patientid))

                locWillFilename = "file_" + str(fileId) + "/" + "workerApply/round_" + str(round) + "_locWill.json"
                wkFilename = "file_" + str(fileId) + "/" + "workerApply/round_" + str(round) + "_locWk.json"
                with open(locWillFilename, 'r') as openfile:
                    locWillDataList = json.load(openfile)

                with open(wkFilename, 'r') as openfile:
                    WkDataList = json.load(openfile)

                for locWdata in locWillDataList:
                    locWillStructList.append(locWillVisit(locWdata["patientId"], locWdata["visitorList"]))

                for wkData in WkDataList:
                    wkStructList.append(worker(wkData["workerId"], wkData["location"]))

                sentList = []
                chooseNumber = 5  # 每个地点派遣5个worker前往
                normalArrangeDf = pd.DataFrame()
                for patientid in range(1, location_number + 1):
                    locwill = locWillStructList[patientid - 1].visitorList
                    leftwill = list(set(locwill) - set(sentList))  # 获得的是没有被分配的workers
                    if len(leftwill) < chooseNumber:  # 说明这个地点所有的候选worker都已经被选走了
                        chooseNumber = len(leftwill)

                    this_choose = sample(leftwill, chooseNumber)
                    locArrStructList[patientid - 1].visitorList = this_choose
                    sentList = sentList + this_choose
                    normalArrangeDf[patientid] = this_choose + [0] * (workernumber - len(this_choose))

                normalArrangementFile = "file_" + str(fileId) + "/" + "TFARworkerArrange/normalArrange_" + str(round) + ".csv"
                normalArrangeDf.to_csv(normalArrangementFile, index=False)

                for patientid in range(1, location_number + 1):
                    arrList = locArrStructList[patientid - 1].visitorList
                    Info = pd.DataFrame(arrList, columns=['workerId'])
                    thisTask = task.loc[(task['PatientId'] == patientid)]
                    result_weight = LocTrust_PSO_Thread(workernumber, Info, round, patientid, thisTask,fileId)
                    RelLocOrder[:, patientid - 1] = result_weight.iloc[:, 0].to_numpy()

                    result_attr_weight = AttrTrust_PSO_Thread(workernumber, Info, round, patientid, thisTask,fileId)
                    for iAttr in range(attribute_number):
                        RelAttrOrder[:, iAttr] = result_attr_weight.iloc[:, iAttr].to_numpy()

                    RelAttrOrderFile = "file_" + str(fileId) + "/" + "TFARworkerOrder/workerRelativeAttr_" + str(round) + "_" + str(patientid) + ".csv"
                    pd.DataFrame(RelAttrOrder).to_csv(RelAttrOrderFile, index=False)

                RelLocOrderFile = "file_" + str(fileId) + "/" + "TFARworkerOrder/workerRelativeLoc_" + str(round) + ".csv"
                pd.DataFrame(RelLocOrder).to_csv(RelLocOrderFile, index=False)

            else:
                # 属性的话当前轮次开始向前进行遍历，前面一个轮次的10个地点和这个轮次的10个地点看能不能组合起来构成一个worker更加多的序列
                RelLocOrder = np.zeros((workerId_data.size, location_number))
                RelAttrOrder = np.zeros((workerId_data.size, attribute_number))
                wkStructList = []  # 存储的是所有worker的结构体，结构体里面含有愿意去哪些地方的信息
                locWillStructList = []  # 存储每个地点有哪些worker愿意前往的结构体，注意是从0开始
                locArrStructList = []  # 存储每个地点安排哪些worker前往的结构体，注意是从0开始
                # 对于两个list进行结构体初始化，每个list里面存储的都是结构体，结构体里面再存储哪些是愿意去的
                for patientid in range(1, location_number + 1):
                    locArrStructList.append(locArrVisit(patientid))

                locWillFilename = "file_" + str(fileId) + "/" + "workerApply/round_" + str(round) + "_locWill.json"
                wkFilename = "file_" + str(fileId) + "/" + "workerApply/round_" + str(round) + "_locWk.json"
                with open(locWillFilename, 'r') as openfile:
                    locWillDataList = json.load(openfile)

                with open(wkFilename, 'r') as openfile:
                    WkDataList = json.load(openfile)

                for locWdata in locWillDataList:
                    locWillStructList.append(locWillVisit(locWdata["patientId"], locWdata["visitorList"]))

                for wkData in WkDataList:
                    wkStructList.append(worker(wkData["workerId"], wkData["location"]))

                locArrNorStructList, trustArrangement = TFARtaskAssignment(round, locWillStructList,fileId)

                trustfulPlace = list(trustArrangement.columns)  # 可信workers前去的地方
                trustfulPlace = list(map(int, trustfulPlace))

                for patientid in tqdm(range(1, 1 + location_number)):
                    workerList_1 = locArrNorStructList[patientid - 1].visitorList
                    if patientid in trustfulPlace:
                        workerList_1.append(trustArrangement[patientid].iloc[0])

                    Info = pd.DataFrame(workerList_1, columns=['workerId'])
                    thisTask = task.loc[(task['PatientId'] == patientid)]
                    result_weight = LocTrust_PSO_Thread(workernumber, Info, round, patientid, thisTask,fileId)
                    RelLocOrder[:, patientid - 1] = result_weight.iloc[:, 0].to_numpy()

                    # 属性操作
                    result_attr_weight = AttrTrust_PSO_Thread(workernumber, Info, round, patientid, thisTask,fileId)
                    for iAttr in range(attribute_number):
                        RelAttrOrder[:, iAttr] = result_attr_weight.iloc[:, iAttr].to_numpy()

                    RelAttrOrderFile = "file_" + str(fileId) + "/" + "TFARworkerOrder/workerRelativeAttr_" + str(round) + "_" + str(patientid) + ".csv"
                    pd.DataFrame(RelAttrOrder).to_csv(RelAttrOrderFile, index=False)
                    # 属性操作

                RelLocOrderFile = "file_" + str(fileId) + "/" + "TFARworkerOrder/workerRelativeLoc_" + str(round) + ".csv"
                pd.DataFrame(RelLocOrder).to_csv(RelLocOrderFile, index=False)

        print("退出线程：" + str(self.file_id))





if __name__ == '__main__':
    # try:
    start_time = time.time()
    workernumber = parameter.workerNumber
    process_list = []
    for fileid in range(1, 6):
        p = MyProcessRM(workernumber, fileid)
        p.start()
        process_list.append(p)

    # Wait all threads to finish.
    for t in process_list:
        t.join()

    process_list = []
    for fileid in range(6, 11):
        p = MyProcessRM(workernumber, fileid)
        p.start()
        process_list.append(p)

    # Wait all threads to finish.
    for t in process_list:
        t.join()

    process_list = []
    for fileid in range(11, 16):
        p = MyProcessRM(workernumber, fileid)
        p.start()
        process_list.append(p)

    # Wait all threads to finish.
    for t in process_list:
        t.join()

    process_list = []
    for fileid in range(16, 21):
        p = MyProcessRM(workernumber, fileid)
        p.start()
        process_list.append(p)

    # Wait all threads to finish.
    for t in process_list:
        t.join()
    #
    print("--- %s seconds ---" % (time.time() - start_time))





