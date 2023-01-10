from random import sample
import random

import pandas as pd
import copy
import numpy as np
import math
import json
from types import SimpleNamespace
import parameter

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


class worker: #存储每个worker的workerId和申请的愿意去采集数据的地点
    def __init__(self,workerid,visLoc):
        self.workerId = workerid
        self.location = visLoc

'''产生的是每个轮次前往检验的worker，生成json文件'''
if __name__ == '__main__':
    fileNumber = 20
    for fileId in range(1,1+fileNumber):
        rounds = 20
        for roundtime in range(1,rounds + 1):
            workernumber = parameter.workerNumber
            applyRate = 0.9  # 申请任务的工人比例，平均每个地方 90/10 = 9
            applyNumber = int(workernumber * applyRate)
            location_number = parameter.locationNumber
            workerList = sample(range(1, workernumber + 1), applyNumber)  # 申请的是哪些workers
            wkStructList = []  # 存储的是所有worker的结构体，结构体里面含有愿意去哪些地方的信息
            locWillStructList = []  # 存储每个地点有哪些worker愿意前往的结构体，注意是从0开始
            locArrStructList = []  # 存储每个地点安排哪些worker前往的结构体，注意是从0开始
            # 对于两个list进行结构体初始化，每个list里面存储的都是结构体，结构体里面再存储哪些是愿意去的
            for patientid in range(1, location_number + 1):
                locWillStructList.append(locWillVisit(patientid))
                locArrStructList.append(locArrVisit(patientid))

            for workerid in range(1, workernumber + 1):   #这边的locWill没有去除bad workers，所以在RM_PM任务分配机制里面需要把bad workers也去除了
                visLocNum = random.randint(1, 3)
                visLoc = sample(range(1, location_number + 1), visLocNum)
                wkStructList.append(worker(workerid, visLoc))  # 每个worker有自己的结构体，存储起来
                if workerid in workerList:
                    for patientid in visLoc:
                        locWillStructList[patientid - 1].visitorList.append(workerid)  # 这边减一是因为从0开始的

            json_locWillstring = json.dumps([ob.__dict__ for ob in locWillStructList])
            json_wKstring = json.dumps([ob.__dict__ for ob in wkStructList])

            locWillFilename = "file_" + str(fileId) + "/" + "workerApply/round_"+ str(roundtime) +"_locWill.json"
            with open(locWillFilename, "w") as outfile:
                outfile.write(json_locWillstring)

            wkFilename = "file_" + str(fileId) + "/" + "workerApply/round_" + str(roundtime) + "_locWk.json"
            with open(wkFilename, "w") as outfile:
                outfile.write(json_wKstring)