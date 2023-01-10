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
from PSO_2000_task_200 import taskAssignment
from multiprocessing import  Process
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

class MyProcessRM (Process):
    def __init__(self, workernumber, folder_Id):
        super(MyProcessRM, self).__init__()
        self.worker_number = workernumber
        self.file_id = folder_Id

    def run(self):
        print ("开始线程：" + str(self.file_id))
        workernumber = self.worker_number  # need change
        fileId = self.file_id

        for UAV_check_number in (range(20, 21)):
            UAV_check_number = parameter.UAV_check_Number
            first_part_number = int(workernumber * 0.2)
            second_part_number = int(workernumber * 0.2)
            last_part_number = workernumber - first_part_number - second_part_number
            InfoFilename = "file_" + str(fileId) + "/" + "workerInfo_accu_" + str(workernumber) + ".csv"
            workerInfo = pd.read_csv(InfoFilename)
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

            vice_GTD_workerId = []

            for round in range(1, 21):
                truthFile = "file_" + str(fileId) + "/" + "RoundTruth/truth_" + str(round) + ".csv"  # 每一轮的truth都是不一样的
                truth = pd.read_csv(truthFile)
                if round == 1:  # 第1轮的时候是随机分配的
                    # 用来记录的是workers之间的相对偏序关系,横坐标是地点
                    RelLocOrder = np.zeros((workerId_data.size, location_number))
                    sample_id = []  # 用来存储被检测的worker
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
                    scores = np.full((workerId_data.size + 1, rounds + 1), 0.5, dtype=float)
                    # 用来记录 workers的属性绝对偏好度，横的分别代表属性 Pregnancies,Glucose,BloodPressure,SkinThickness,BMI
                    attrScores = np.full((workerId_data.size + 1, attribute_number), 0.5, dtype=float)
                    # 用来记录workers的地点绝对偏好度，横的代表地点
                    locScores = np.full((workerId_data.size + 1, location_number), 0.5, dtype=float)

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

                    normalArrangementFile = "file_" + str(fileId) + "/" + "workerArrange/normalArrange_" + str(round) + ".csv"
                    normalArrangeDf.to_csv(normalArrangementFile, index=False)
                    trustArrangement = pd.DataFrame()
                    trustArrangementFile = "file_" + str(fileId) + "/" + "workerArrange/trustArrange_" + str(round) + ".csv"
                    trustArrangement.to_csv(trustArrangementFile, index=False)

                    # create locations where UAV visits
                    UAV_place = sample(range(1, location_number + 1), UAV_check_number)
                    No_UAV_place = list(set(range(1, location_number + 1)) - set(UAV_place))
                    UAV_place_df = pd.DataFrame(UAV_place)
                    UAV_place_dfFile = "file_" + str(fileId) + "/" + "workerArrange/UAVArrange_" + str(round) + ".csv"
                    UAV_place_df.to_csv(UAV_place_dfFile, index=False)

                    for patientid in UAV_place:
                        arrList = locArrStructList[patientid - 1].visitorList
                        answer = truth.loc[(truth['PatientId'] == patientid)]
                        answer = answer.reset_index(drop=True)
                        fenmu = copy.deepcopy(answer)
                        fenmu = fenmu.replace(to_replace=0, value=1)  # 防止原来的数据是0的话，0在分母上不好计算
                        this_task = task.loc[(task['PatientId'] == patientid)]
                        patientList = range(1, location_number + 1)
                        leftPatient = list(patientList).remove(patientid)  # 获得前往这个地点因此剩余的其他的地点
                        for workerid in arrList:
                            sample_id.append(workerid)
                            workerDataFile = "file_" + str(fileId) + "/" + "workerData/round_" + str(round) + "/worker_" + str(workerid) + ".csv"
                            workerData = pd.read_csv(workerDataFile)
                            predict = workerData.loc[(workerData['PatientId'] == patientid)]
                            predict = predict.reset_index(drop=True)
                            result = (predict - answer).abs() / fenmu
                            this_task = this_task.reset_index(drop=True)
                            adj_result = result.mul(this_task)

                            adjust_res = adj_result.values.tolist()[0][1::]  # 要除去最前面的patientId,得到的是误差的大小
                            error_count = sum(condition(x) for x in adjust_res)  # count means wrong number
                            last_score = scores[workerid, round - 1]
                            focusNumber = np.count_nonzero(this_task) - 1  # 要除去最前面的patientId
                            sigma = 1 - (error_count / focusNumber)  # means accuracy rate
                            if error_count > focusNumber / 2:  # apply punishment
                                score = last_score + (sigma - 1) * (last_score / rou)
                                scores[workerid, round] = score
                            else:  # reward
                                score = last_score + (sigma / rou) * (1 - last_score)
                                scores[workerid, round] = score

                            # 更新地点信任度
                            loc_error_count = sum(loc_condition(x) for x in adjust_res)
                            loc_sigma = 1 - (loc_error_count / focusNumber)
                            last_loc_score = locScores[workerid][patientid - 1]
                            # 将这个worker其他没有被检验的地方进行Timeout处理,Timeout放在前面是为了加速处理
                            locScore = locScores[workerid]
                            fai = locTime_co * (1 - locScore)
                            finalScore = locScore - fai
                            finalScore[finalScore < 0] = 0
                            locScores[workerid] = finalScore

                            if loc_error_count > focusNumber / 2:  # 惩罚
                                locScores[workerid][patientid - 1] = last_loc_score + (loc_sigma - 1) * (last_loc_score / loc_rou)
                            else:
                                locScores[workerid][patientid - 1] = last_loc_score + (loc_sigma / loc_rou) * (1 - last_loc_score)

                            # 更新属性信任度
                            nonZero = this_task.loc[:, (this_task != 0).any(axis=0)]
                            nonZeroName = list(nonZero.columns)[1::]  # 注意要把patientId给排除了
                            ZeroName = list(set(attriName) - set(nonZeroName))
                            for name in nonZeroName:
                                predictValue = workerData.loc[(workerData['PatientId'] == patientid)][name].values[0]
                                answerValue = truth.loc[(truth['PatientId'] == patientid)][name].values[0]
                                judgeResult = abs(predictValue - answerValue) / answerValue
                                coordinate = attriName.index(name)  # 获取下标
                                if judgeResult > 0.25:  # 将会被惩罚
                                    if judgeResult > 1:
                                        judgeResult = 1
                                    attrScores[workerid][coordinate] = attrScores[workerid][coordinate] - (judgeResult / attrRou) * attrScores[workerid][coordinate]
                                else:
                                    attrScores[workerid][coordinate] = attrScores[workerid][coordinate] + ((1 - judgeResult) / attrRou )* (1 - attrScores[workerid][coordinate])
                            for name in ZeroName:
                                coordinate = attriName.index(name)  # 获取下标
                                fai = attrTime_co * (1 - attrScores[workerid][coordinate])
                                attrScores[workerid][coordinate] = max(0, attrScores[workerid][coordinate] - fai)

                    # 对于没有被UAV检查的workers进行操作
                    noCheck = list(set(range(1, workernumber + 1)) - set(sample_id))
                    for noCheckid in noCheck:  # time punishment
                        last_score = scores[noCheckid, round - 1]
                        delta = time_co * (1 - last_score)
                        score = max(last_score - delta, 0)
                        scores[noCheckid, round] = score

                        attrScore = attrScores[noCheckid]
                        fai = attrTime_co * (1 - attrScore)
                        finalScore = attrScore - fai
                        finalScore[finalScore < 0] = 0
                        attrScores[noCheckid] = finalScore

                        # Timeout地点的绝对偏好度
                        locScore = locScores[noCheckid]
                        loc_fai = locTime_co * (1 - locScore)
                        locfinalScore = locScore - loc_fai
                        locfinalScore[locfinalScore < 0] = 0
                        locScores[noCheckid] = locfinalScore

                    # 对于没有被UAV访问的地点获取相对偏好度
                    for patientid in No_UAV_place:
                        arrList = locArrStructList[patientid - 1].visitorList
                        Info = pd.DataFrame(arrList, columns=['workerId'])
                        thisTask = task.loc[(task['PatientId'] == patientid)]
                        result_weight = LocTrust_PSO_Thread(workernumber, Info, round, patientid, thisTask,fileId)
                        RelLocOrder[:, patientid - 1] = result_weight.iloc[:, 0].to_numpy()
                        print()

                    attrScoresFile = "file_" + str(fileId) + "/" + "workerAttrScores.csv"
                    pd.DataFrame(attrScores).to_csv(attrScoresFile, index=False)

                    locScoresFile = "file_" + str(fileId) + "/" + "workerLocScores.csv"
                    pd.DataFrame(locScores).to_csv(locScoresFile, index=False)

                    scoresFile = "file_" + str(fileId) + "/" + "workerOverallScores.csv"
                    pd.DataFrame(scores).to_csv(scoresFile, index=False)

                    # 混合的时候需要考虑上一轮的相对偏好度
                    RelLocOrderFile = "file_" + str(fileId) + "/" + "workerOrder/workerRelativeLoc_" + str(round) + ".csv"
                    pd.DataFrame(RelLocOrder).to_csv(RelLocOrderFile, index=False)


                else:  # 当不是第一次随机安排的时候，就需要根据任务算法安排分配的进行调用了
                    # 接下来去做根据第一次的结果是安排任务的算法
                    scoresFile = "file_" + str(fileId) + "/" + "workerOverallScores.csv"
                    scores = pd.read_csv(scoresFile)
                    scores = scores.to_numpy()
                    # 用来记录 workers的属性绝对偏好度，横的分别代表属性 Pregnancies,Glucose,BloodPressure,SkinThickness,BMI
                    attrScoresFile = "file_" + str(fileId) + "/" + "workerAttrScores.csv"
                    attrScores = pd.read_csv(attrScoresFile)
                    attrScores = attrScores.to_numpy()
                    # 用来记录workers的地点绝对偏好度，横的代表地点
                    locScoresFile = "file_" + str(fileId) + "/" + "workerLocScores.csv"
                    locScores = pd.read_csv(locScoresFile)
                    locScores = locScores.to_numpy()
                    # 上一轮的地点相对偏序关系
                    RelLocOrderFile = "file_" + str(fileId) + "/" + "workerOrder/workerRelativeLoc_" + str(round - 1) + ".csv"
                    RelLocOrder = pd.read_csv(RelLocOrderFile)

                    sample_id = []  # 用来存储被总体信任度更新的worker, 包括UAV和可信workers检测的
                    sample_loc_id = []  # 用来存储被地点信任度更新的worker, 包括UAV和可信workers检测的
                    for i in range(location_number):
                        sample_loc_id.append([])
                    sample_attr_id = []  # 用来存储被属性信任度更新的worker, 包括UAV和可信workers检测的
                    for i in range(attribute_number):
                        sample_attr_id.append([])

                    LTOverall = scores[:, round - 1]
                    badWorkersVa = np.where(LTOverall < 0.3)[0]  # 小于0.3认为不可信
                    if len(badWorkersVa) == 0:
                        badWorkers = []
                    else:
                        badWorkers = list(badWorkersVa)

                    Trustful = np.where(LTOverall > vice_point)[0]  # 可信度大于vice_point则认为是可信workers
                    if len(Trustful) == 0:
                        TrustfulWorkers = []
                    else:
                        TrustfulWorkers = list(Trustful)

                    workerList = []
                    wkStructList = []  # 存储的是所有worker的结构体，结构体里面含有愿意去哪些地方的信息
                    locWillStructList = []  # 存储每个地点有哪些worker愿意前往,包含了bad workers的结构体，注意List是从0开始
                    locArrNorStructList = []  # 存储每个地点安排哪些 普通 worker前往的结构体，注意List是从0开始
                    trustArrangement = pd.DataFrame()  # 存储的是可信的workers前往的地点，一个地点一个可信的worker

                    locWillFilename = "file_" + str(fileId) + "/" + "workerApply/round_" + str(round) + "_locWill.json"
                    wkFilename = "file_" + str(fileId) + "/" + "workerApply/round_" + str(round) + "_locWk.json"
                    with open(locWillFilename, 'r') as openfile:
                        locWillDataList = json.load(openfile)

                    with open(wkFilename, 'r') as openfile:
                        WkDataList = json.load(openfile)

                    for locWdata in locWillDataList:
                        locWillStructList.append(locWillVisit(locWdata["patientId"], locWdata["visitorList"]))
                        workerList = workerList + locWdata["visitorList"]

                    for wkData in WkDataList:
                        wkStructList.append(worker(wkData["workerId"], wkData["location"]))

                    workerList = list(set(workerList))  # 去除重复的
                    leftWorkers = list(set(workerList) - set(badWorkers))  # 是申请的workers中把恶意的workers给排除了
                    willTrustful = list(set(workerList).intersection(TrustfulWorkers))  # 不仅是可信的worker，并且也是申请了的

                    sentList = []
                    # create locations where UAV visits
                    locArrNorStructList, trustArrangement, UAV_place = taskAssignment(round, UAV_check_number,
                                                                                      leftWorkers, willTrustful,
                                                                                      wkStructList,fileId)
                    trustfulPlace = list(trustArrangement.columns)  # 可信workers前去的地方
                    trustfulPlace = list(map(int, trustfulPlace))
                    No_UAV_place = list(set(range(1, location_number + 1)) - set(UAV_place))
                    No_check_place = list(set(No_UAV_place) - set(trustfulPlace))

                    for patientid in UAV_place:  # UAV作为标准数据的地方
                        arrList = locArrNorStructList[patientid - 1].visitorList
                        answer = truth.loc[(truth['PatientId'] == patientid)]
                        answer = answer.reset_index(drop=True)
                        fenmu = copy.deepcopy(answer)
                        fenmu = fenmu.replace(to_replace=0, value=1)  # 防止原来的数据是0的话，0在分母上不好计算
                        this_task = task.loc[(task['PatientId'] == patientid)]
                        patientList = range(1, location_number + 1)
                        leftPatient = list(patientList).remove(patientid)  # 获得前往这个地点因此剩余的其他的地点
                        for workerid in arrList:
                            sample_id.append(workerid)
                            workerDataFile = "file_" + str(fileId) + "/" + "workerData/round_" + str(round) + "/worker_" + str(workerid) + ".csv"
                            workerData = pd.read_csv(workerDataFile)
                            predict = workerData.loc[(workerData['PatientId'] == patientid)]
                            predict = predict.reset_index(drop=True)
                            result = (predict - answer).abs() / fenmu
                            this_task = this_task.reset_index(drop=True)
                            adj_result = result.mul(this_task)

                            adjust_res = adj_result.values.tolist()[0][1::]  # 要除去最前面的patientId,得到的是误差的大小
                            error_count = sum(condition(x) for x in adjust_res)  # count means wrong number
                            last_score = scores[workerid, round - 1]
                            focusNumber = np.count_nonzero(this_task) - 1  # 要除去最前面的patientId
                            sigma = 1 - (error_count / focusNumber)  # means accuracy rate
                            if error_count > focusNumber / 2:  # apply punishment
                                score = last_score + (sigma - 1) * (last_score / rou)
                                scores[workerid, round] = score
                            else:  # reward
                                score = last_score + (sigma / rou) * (1 - last_score)
                                scores[workerid, round] = score

                            # 更新地点信任度
                            sample_loc_id[patientid - 1].append(workerid)
                            loc_error_count = sum(loc_condition(x) for x in adjust_res)
                            loc_sigma = 1 - (loc_error_count / focusNumber)
                            last_loc_score = locScores[workerid][patientid - 1]
                            # 将这个worker其他没有被检验的地方进行Timeout处理,Timeout放在前面是为了加速处理
                            locScore = locScores[workerid]
                            fai = locTime_co * (1 - locScore)
                            finalScore = locScore - fai
                            finalScore[finalScore < 0] = 0
                            locScores[workerid] = finalScore

                            if loc_error_count > focusNumber / 2:  # 惩罚
                                locScores[workerid][patientid - 1] = last_loc_score + (loc_sigma - 1) * (
                                            last_loc_score / loc_rou)
                            else:
                                locScores[workerid][patientid - 1] = last_loc_score + (loc_sigma / loc_rou) * (
                                            1 - last_loc_score)

                            # 更新属性信任度
                            nonZero = this_task.loc[:, (this_task != 0).any(axis=0)]
                            nonZeroName = list(nonZero.columns)[1::]  # 注意要把patientId给排除了
                            ZeroName = list(set(attriName) - set(nonZeroName))
                            for name in nonZeroName:
                                predictValue = workerData.loc[(workerData['PatientId'] == patientid)][name].values[0]
                                answerValue = truth.loc[(truth['PatientId'] == patientid)][name].values[0]
                                judgeResult = abs(predictValue - answerValue) / answerValue
                                coordinate = attriName.index(name)  # 获取下标
                                sample_attr_id[coordinate].append(workerid)

                                if judgeResult > 0.25:  # 将会被惩罚
                                    attrScores[workerid][coordinate] = attrScores[workerid][coordinate] - (1 / attrRou) * \
                                                                       attrScores[workerid][coordinate]
                                else:
                                    attrScores[workerid][coordinate] = attrScores[workerid][
                                                                           coordinate] + (1 / attrRou) * (
                                                                                   1 - attrScores[workerid][coordinate])
                            for name in ZeroName:
                                coordinate = attriName.index(name)  # 获取下标
                                fai = attrTime_co * (1 - attrScores[workerid][coordinate])
                                attrScores[workerid][coordinate] = max(0, attrScores[workerid][coordinate] - fai)

                    for patientid in trustfulPlace:  # 可信workers作为标准数据的地方
                        trust_workerid = trustArrangement.iloc[0][patientid]
                        trust_worker_score = scores[trust_workerid, round - 1]
                        trustDataFile = "file_" + str(fileId) + "/" + "workerData/round_" + str(round) + "/worker_" + str(trust_workerid) + ".csv"
                        trustData = pd.read_csv(trustDataFile)
                        answer = trustData.loc[(trustData['PatientId'] == patientid)]
                        answer = answer.reset_index(drop=True)
                        arrList = locArrNorStructList[patientid - 1].visitorList
                        fenmu = copy.deepcopy(answer)
                        fenmu = fenmu.replace(to_replace=0, value=1)  # 防止原来的数据是0的话，0在分母上不好计算
                        this_task = task.loc[(task['PatientId'] == patientid)]
                        patientList = range(1, location_number + 1)
                        leftPatient = list(patientList).remove(patientid)  # 获得前往这个地点因此剩余的其他的地点
                        for workerid in arrList:
                            sample_id.append(workerid)
                            workerDataFile = "file_" + str(fileId) + "/" + "workerData/round_" + str(round) + "/worker_" + str(workerid) + ".csv"
                            workerData = pd.read_csv(workerDataFile)
                            predict = workerData.loc[(workerData['PatientId'] == patientid)]
                            predict = predict.reset_index(drop=True)
                            result = (predict - answer).abs() / fenmu
                            this_task = this_task.reset_index(drop=True)
                            adj_result = result.mul(this_task)

                            adjust_res = adj_result.values.tolist()[0][1::]  # 要除去最前面的patientId,得到的是误差的大小
                            error_count = sum(condition(x) for x in adjust_res)  # count means wrong number
                            last_score = scores[workerid, round - 1]
                            focusNumber = np.count_nonzero(this_task) - 1  # 要除去最前面的patientId
                            sigma = 1 - (error_count / focusNumber)  # means accuracy rate
                            if error_count > focusNumber / 2:  # apply punishment
                                score = last_score + (sigma - 1) * (last_score / rou) * trust_worker_score
                                scores[workerid, round] = score
                            else:  # reward
                                score = last_score + (sigma / rou) * (1 - last_score) * trust_worker_score
                                scores[workerid, round] = score

                    # 修改成每个地方的属性和地点信任度由这个地方属性和地点信任度比较高的worker来作为标准
                    for patientid in No_UAV_place:  # 没有UAV的地方
                        thisArrList = locArrNorStructList[patientid - 1].visitorList  # 这个地方的所有workers
                        if patientid in trustfulPlace:
                            thisArrList.append(patientid)
                        maxLocScore = 0
                        maxLocWorkerid = 0
                        maxAttrScore = [0] * attribute_number
                        maxAttrWorkerid = [0] * attribute_number
                        this_task = task.loc[(task['PatientId'] == patientid)]
                        nonZero = this_task.loc[:, (this_task != 0).any(axis=0)]
                        nonZeroName = list(nonZero.columns)[1::]  # 注意要把patientId给排除了

                        for workerid in thisArrList:
                            loc_score = locScores[workerid][patientid - 1]
                            # 获取最高的地点可信度
                            if loc_score > maxLocScore:
                                maxLocScore = loc_score
                                maxLocWorkerid = workerid
                            # 获取最高的属性可信度
                            for name in nonZeroName:
                                coordinate = attriName.index(name)
                                if attrScores[workerid][coordinate] > maxAttrScore[coordinate]:
                                    maxAttrScore[coordinate] = attrScores[workerid][coordinate]
                                    maxAttrWorkerid[coordinate] = workerid

                        # 更新地点绝对信任度
                        if maxLocScore > vice_loc_point:  # 过了标准就开始检验
                            checkList = copy.deepcopy(thisArrList)
                            checkList.remove(maxLocWorkerid)
                            sample_loc_id[patientid - 1] = sample_loc_id[patientid - 1] + checkList  # 添加入检验名单
                            trustDataFile = "file_" + str(fileId) + "/" + "workerData/round_" + str(round) + "/worker_" + str(maxLocWorkerid) + ".csv"
                            trustData = pd.read_csv(trustDataFile)
                            answer = trustData.loc[(trustData['PatientId'] == patientid)]
                            answer = answer.reset_index(drop=True)
                            arrList = locArrNorStructList[patientid - 1].visitorList
                            fenmu = copy.deepcopy(answer)
                            fenmu = fenmu.replace(to_replace=0, value=1)  # 防止原来的数据是0的话，0在分母上不好计算
                            focusNumber = np.count_nonzero(this_task) - 1
                            for workerid in checkList:
                                workerDataFile = "file_" + str(fileId) + "/" + "workerData/round_" + str(round) + "/worker_" + str(workerid) + ".csv"
                                workerData = pd.read_csv(workerDataFile)
                                predict = workerData.loc[(workerData['PatientId'] == patientid)]
                                predict = predict.reset_index(drop=True)
                                result = (predict - answer).abs() / fenmu
                                this_task = this_task.reset_index(drop=True)
                                adj_result = result.mul(this_task)
                                adjust_res = adj_result.values.tolist()[0][1::]  # 要除去最前面的patientId,得到的是误差的大小
                                loc_error_count = sum(loc_condition(x) for x in adjust_res)
                                loc_sigma = 1 - (loc_error_count / focusNumber)
                                last_loc_score = locScores[workerid][patientid - 1]

                                if loc_error_count > focusNumber / 2:  # 惩罚
                                    locScores[workerid][patientid - 1] = last_loc_score + (loc_sigma - 1) * (
                                                last_loc_score / loc_rou) * maxLocScore
                                else:
                                    locScores[workerid][patientid - 1] = last_loc_score + (loc_sigma / loc_rou) * (
                                                1 - last_loc_score) * maxLocScore

                        # 更新属性信任度
                        for name in nonZeroName:
                            coordinate = attriName.index(name)
                            if maxAttrScore[coordinate] > vice_attr_point:
                                checkList = copy.deepcopy(thisArrList)
                                checkList.remove(maxAttrWorkerid[coordinate])
                                sample_attr_id[coordinate] = sample_attr_id[coordinate] + checkList  # 添加入检验名单
                                trustDataFile = "file_" + str(fileId) + "/" + "workerData/round_" + str(round) + "/worker_" + str(
                                    maxAttrWorkerid[coordinate]) + ".csv"
                                trustData = pd.read_csv(trustDataFile)
                                for workerid in checkList:
                                    workerDataFile = "file_" + str(fileId) + "/" + "workerData/round_" + str(round) + "/worker_" + str(
                                        workerid) + ".csv"
                                    workerData = pd.read_csv(workerDataFile)
                                    predictValue = workerData.loc[(workerData['PatientId'] == patientid)][name].values[0]
                                    answerValue = trustData.loc[(trustData['PatientId'] == patientid)][name].values[0]
                                    judgeResult = abs(predictValue - answerValue) / answerValue

                                    if judgeResult > 0.25:  # 将会被惩罚
                                        if judgeResult > 1:
                                            judgeResult = 1
                                        attrScores[workerid][coordinate] = attrScores[workerid][coordinate] - (judgeResult / attrRou) * attrScores[workerid][coordinate] * maxAttrScore[coordinate]
                                    else:
                                        attrScores[workerid][coordinate] = attrScores[workerid][coordinate] + (1 - judgeResult) / attrRou * (1 - attrScores[workerid][coordinate]) * maxAttrScore[coordinate]

                    # 对于没有进行总体信任度更新的workers进行操作
                    noCheck = list(set(range(1, workernumber + 1)) - set(sample_id))
                    for noCheckid in noCheck:  # time punishment
                        last_score = scores[noCheckid, round - 1]
                        delta = time_co * (1 - last_score)
                        score = max(last_score - delta, 0)
                        scores[noCheckid, round] = score

                    # 对于workers中所有地点没有被更新的进行Timeout
                    for patientid in range(1, 1 + location_number):
                        CheckList = sample_loc_id[patientid - 1]
                        noCheckList = list(set(range(1, workernumber + 1)) - set(CheckList))
                        for workerid in noCheckList:
                            locScore = locScores[workerid][patientid - 1]
                            loc_fai = locTime_co * (1 - locScore)
                            locfinalScore = locScore - loc_fai
                            if locfinalScore < 0:
                                locfinalScore = 0
                            locScores[workerid][patientid - 1] = locfinalScore

                    # 对于workers中所有属性没有被更新的进行Timeout
                    for name in attriName:
                        coordinate = attriName.index(name)
                        CheckList = sample_attr_id[coordinate]
                        noCheckList = list(set(range(1, workernumber + 1)) - set(CheckList))
                        for workerid in noCheckList:
                            attrScore = attrScores[workerid][coordinate]
                            fai = attrTime_co * (1 - attrScore)
                            finalScore = attrScore - fai
                            if finalScore < 0:
                                finalScore = 0
                            attrScores[workerid][coordinate] = finalScore

                    thisRelLocOrder = np.zeros((workerId_data.size, location_number))
                    for patientid in No_check_place:  # 没有UAV和可信workers的地方
                        arrList = locArrNorStructList[patientid - 1].visitorList
                        Info = pd.DataFrame(arrList, columns=['workerId'])
                        thisTask = task.loc[(task['PatientId'] == patientid)]
                        result_weight = LocTrust_PSO_Thread(workernumber, Info, round, patientid, thisTask,fileId)
                        thisRelLocOrder[:, patientid - 1] = result_weight.iloc[:, 0].to_numpy()
                        print()

                    attrScoresFile = "file_" + str(fileId) + "/" + "workerAttrScores.csv"
                    pd.DataFrame(attrScores).to_csv(attrScoresFile, index=False)

                    locScoresFile = "file_" + str(fileId) + "/" + "workerLocScores.csv"
                    pd.DataFrame(locScores).to_csv(locScoresFile, index=False)

                    scoresFile = "file_" + str(fileId) + "/" + "workerOverallScores.csv"
                    pd.DataFrame(scores).to_csv(scoresFile, index=False)

                    # 混合的时候需要考虑上一轮的相对偏好度
                    RelLocOrderFile = "file_" + str(fileId) + "/" + "workerOrder/workerRelativeLoc_" + str(round) + ".csv"
                    pd.DataFrame(thisRelLocOrder).to_csv(RelLocOrderFile, index=False)



        print ("退出线程：" + str(self.file_id))





if __name__ == '__main__':
    # try:
    start_time = time.time()
    workernumber = parameter.workerNumber

    process_list = []
    for fileid in range(1, 11):
        p = MyProcessRM(workernumber, fileid)
        p.start()
        process_list.append(p)

    # Wait all threads to finish.
    for t in process_list:
        t.join()

    process_list = []
    for fileid in range(11, 21):
        p = MyProcessRM(workernumber, fileid)
        p.start()
        process_list.append(p)

    # # Wait all threads to finish.
    # for t in process_list:
    #     t.join()
    #
    # process_list = []
    # for fileid in range(11, 16):
    #     p = MyProcessRM(workernumber, fileid)
    #     p.start()
    #     process_list.append(p)
    #
    # # Wait all threads to finish.
    # for t in process_list:
    #     t.join()
    #
    # process_list = []
    # for fileid in range(16, 21):
    #     p = MyProcessRM(workernumber, fileid)
    #     p.start()
    #     process_list.append(p)
    #
    # # Wait all threads to finish.
    # for t in process_list:
    #     t.join()

    print("--- %s seconds ---" % (time.time() - start_time))




    # except Exception as e:
    #     exc_type, exc_obj, exc_tb = sys.exc_info()
    #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #     print(exc_type, fname, exc_tb.tb_lineno)




'''
            for location in UAV_place:
                if location not in worker_place:
                    continue
                workerId_temp = [i for i, e in enumerate(worker_place) if e == location]  # 获取和UAV在同一个地方的workers
                for workerid_minus in workerId_temp:
                    workerId = workerid_minus + 1
                    sample_id.append(workerId)
                    answer = truth.loc[(truth['PatientId'] == location)]
                    answer = answer.reset_index(drop=True)
                    fenmu = copy.deepcopy(answer)
                    fenmu = fenmu.replace(to_replace=0, value=1)  # 防止原来的数据是0的话，0在分母上不好计算
                    workerDataFile = "workerData/round_" + str(round) + "/worker_" + str(workerId) + ".csv"
                    workerData = pd.read_csv(workerDataFile)
                    predict = workerData.loc[(workerData['PatientId'] == location)].iloc[:, :]
                    predict = predict.reset_index(drop=True)
                    result = (predict - answer).abs() / fenmu
                    result = result.values.tolist()[0]
                    error_count = sum(condition(x) for x in result)  # count means wrong number
                    last_score = scores[workerId, round - 1]
                    sigma = 1 - error_count / attribute_number  # means accuracy rate
                    if error_count > 2:  # apply punishment
                        score = last_score + (sigma - 1) * (last_score / rou)
                        scores[workerId, round] = score
                    else:  # reward
                        score = last_score + (sigma / rou) * (1 - last_score)
                        scores[workerId, round] = score
            vice_place = []
            for vice_workerId in vice_GTD_workerId:
                location = worker_place[vice_workerId - 1]
                workerId_temp = [i for i, e in enumerate(worker_place) if e == location]
                for workerid_minus in workerId_temp:
                    workerId = workerid_minus + 1
                    if workerId == vice_workerId:
                        continue
                    sample_id.append(workerId)
                    answer = all_data.loc[
                                 (all_data['PatientId'] == location) & (all_data['source'] == vice_workerId)].iloc[:,
                             :-1]
                    answer = answer.reset_index(drop=True)
                    predict = all_data.loc[(all_data['PatientId'] == location) & (all_data['source'] == workerId)].iloc[
                              :, :-1]
                    predict = predict.reset_index(drop=True)
                    fenmu = answer.copy()
                    fenmu = fenmu.replace(to_replace=0, value=1)
                    result = (predict - answer).abs() / fenmu
                    result = result.values.tolist()[0]
                    error_count = sum(condition(x) for x in result)  # count means wrong number
                    last_score = scores[workerId, round - 1]
                    sigma = 1 - error_count / attribute_number  # means accuracy rate
                    if error_count > 2:  # apply punishment
                        score = last_score + (sigma - 1) * (last_score / rou)
                        scores[workerId, round] = score
                    else:  # reward
                        score = last_score + (sigma / rou) * (1 - last_score)
                        scores[workerId, round] = score

            noCheck = list(set(workerList) - set(sample_id))
            for noCheckid in noCheck:  # time punishment
                last_score = scores[noCheckid, round - 1]
                delta = time_co * (1 - last_score)
                score = max(last_score - delta, 0)
                scores[noCheckid, round] = score

            find = scores[:, round]
            filtered = (find > vice_point).nonzero()
            filtered = np.array(filtered).tolist()[0]
            vice_GTD_workerId = []
            for workerId in filtered:
                vice_GTD_workerId.append(workerId)

        scores = scores.T
        df = pd.DataFrame(scores)
        df = df.iloc[:, 1:]
        filename = "Data_" + str(workernumber) + "_vice_" + str(vice_point) + "/test_" + str(
            testId) + "/Mech_wks_" + str(UAV_check_number) + '_rou_' + str(rou) + ".csv"
        df.to_csv(filename)

        first_part = df.iloc[:, 0:first_part_number].mean(axis=1)
        second_part = df.iloc[:, first_part_number:first_part_number + second_part_number].mean(axis=1)
        final_part = df.iloc[:, first_part_number + second_part_number:].mean(axis=1)
        frames = [first_part, second_part, final_part]
        result = pd.concat(frames, axis=1)
        filename_result = "Data_" + str(workernumber) + "_vice_" + str(vice_point) + "/test_" + str(
            testId) + "/Mech_wks_Cct_" + str(UAV_check_number) + '_rou_' + str(rou) + ".csv"
        result.to_csv(filename_result)
'''











