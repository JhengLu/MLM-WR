import json

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from multiprocessing import  Process
import copy
from random import sample
import parameter

class locWillVisit: #定义这个地点愿意前往的worker有哪些
    def __init__(self,patientid,visLoc =None):
        self.patientId = patientid
        if visLoc:
            self.visitorList = visLoc
        else:
            self.visitorList = []

class worker: #存储每个worker的workerId和申请的愿意去采集数据的地点
    def __init__(self,workerid,visLoc):
        self.workerId = workerid
        self.location = visLoc


'''这边是随机分配的方法'''
if __name__ == '__main__':
    fileNumber = 20
    for fileId in tqdm(range(1,1+fileNumber)):
        workernumber = parameter.workerNumber
        location_number = parameter.locationNumber
        locWillStructList = []
        wkStructList = []
        for round in range(1, 21):
            locWillFilename = "file_" + str(fileId) + "/" + "workerApply/round_" + str(round) + "_locWill.json"
            wkFilename = "file_" + str(fileId) + "/" + "workerApply/round_" + str(round) + "_locWk.json"
            with open("file_" + str(fileId) + "/" + "workerApply/round_1_locWill.json", 'r') as openfile:
                locWillDataList = json.load(openfile)

            with open("file_" + str(fileId) + "/" + "workerApply/round_1_locWk.json", 'r') as openfile:
                WkDataList = json.load(openfile)

            for locWdata in locWillDataList:
                locWillStructList.append(locWillVisit(locWdata["patientId"], locWdata["visitorList"]))

            for wkData in WkDataList:
                wkStructList.append(worker(wkData["workerId"], wkData["location"]))

            chooseNumber = 5  # 每个地点派遣5个worker前往
            sentList = []
            randomArrangeDf = pd.DataFrame()
            for patientid in range(1, location_number + 1):
                locwill = locWillStructList[patientid - 1].visitorList
                leftwill = list(set(locwill) - set(sentList))  # 获得的是没有被分配的workers
                if len(leftwill) < chooseNumber:  # 说明这个地点所有的候选worker都已经被选走了
                    chooseNumber = len(leftwill)
                    if chooseNumber == 0:
                        print(" chooose error")

                this_choose = sample(leftwill, chooseNumber)
                sentList = sentList + this_choose
                randomArrangeDf[patientid] = this_choose + [0] * (workernumber - len(this_choose))

            randomArrangementFile = "file_" + str(fileId) + "/" + "TFARworkerArrange/randomArrange_" + str(round) + ".csv"
            randomArrangeDf.to_csv(randomArrangementFile, index=False)
