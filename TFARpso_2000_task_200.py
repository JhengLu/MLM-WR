import copy
import json

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import math
import time
from multiprocessing import  Process
from TFARMergeLoc import TFARMergeLocF
from TFARMergeAttr import TFARMergeAttrF
from random import sample

from collections import defaultdict
import parameter

def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items()
                            if len(locs)>1)

def checkIfDuplicates_1(listOfElems):
    ''' Check if given list contains any duplicates '''
    if len(listOfElems) == len(set(listOfElems)):
        return False
    else:
        return True

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


# leftWorkers指的是在申请的workers中把总体信任度过低的workers给排除了，不对其进行任务分配
# willTrustful指的是剩余workers中可信的workers
def TFARtaskAssignment(roundtime, locWillStructList:list,fileId):

    worker_number = parameter.workerNumber
    punish_co = 2
    task_number = parameter.taskNumber
    particle_number =  2 * task_number
    location_number = parameter.locationNumber

    filename = "task_" + str(task_number) + ".csv"
    task = pd.read_csv(filename)
    task = task.sort_values(by=['PatientId'])  #注意这边，不然下面直接相乘是会出错的
    task = task.reset_index(drop=True)

    # 获取当前轮次最新历史地点偏序关系组合
    integLocOrder = TFARMergeLocF(roundtime,fileId)
    attrOrder = TFARMergeAttrF(roundtime,fileId)
    attrName = list(task.columns)[1:]
    attrNumber = len(attrName)

    busyPlace = list(range(1,1+location_number)) #这边的busyPlace就需要是所有的地点了

    # 对于busyPlace进行PSO算法
    busyPlaceRange = [0] * len(busyPlace)
    PSO = np.zeros((particle_number, len(busyPlace)))
    PSO_reveal = np.zeros((particle_number, len(busyPlace)))  # 显示的是安排的workers
    PSO_best = np.zeros((particle_number, len(busyPlace)))
    velocity = np.zeros((particle_number, len(busyPlace)))
    fitness = np.zeros(particle_number)
    fitness_te = np.zeros(particle_number)
    plenth = np.zeros(particle_number)
    fitness_local_best = np.zeros(particle_number)
    plenth_local_best = np.zeros(particle_number)

    global_PSO_best = np.zeros(len(busyPlace))  # means global over the history
    global_fitness_best_value = -1000

    #记录每个地方是有几个worker
    willWorkers = []
    wkMapPlace = np.zeros((len(busyPlace),worker_number))  #是每个地点存放的待定worker，查询起来会快捷很多
    for i in range(0,len(busyPlace)):
        patientid = busyPlace[i]
        busyPlaceRange[i] = len(locWillStructList[patientid - 1].visitorList)
        willWorkers = willWorkers + locWillStructList[patientid - 1].visitorList
        wkMapPlace[i] = locWillStructList[patientid - 1].visitorList + [0] * (worker_number -len(locWillStructList[patientid - 1].visitorList))
    willWorkers = list(set(willWorkers))   # 注意这边，不然会出现重复的，需要set一下子


    generation_number =  120
    best_particle_record = np.zeros((generation_number, task_number))
    bestLenth = 0 # 记录的是最多的能有可信worker前往的地点数量

    # 属性操作
    attrDf = copy.deepcopy(attrOrder)
    valueDf = pd.DataFrame(list(range(len(attrOrder), 0, -1)), columns=['value', ])
    attrCountDf = pd.DataFrame()  # 每个worker对应的属性排位表
    for workerid in range(1, 1 + worker_number):
        maskDf = attrDf[attrDf == workerid]
        maskDf = maskDf.fillna(0)
        maskDf[maskDf > 1] = 1
        resultDf = maskDf[attrName].multiply(valueDf["value"], axis="index")
        sumSeries = resultDf.sum(axis=0).to_frame().T
        attrCountDf = attrCountDf.append(sumSeries, ignore_index=True)[sumSeries.columns.tolist()]
    attrCountDf.insert(0, "workerId", range(1, 1 + worker_number))
    # 属性操作

    first_reveal = np.zeros((particle_number, len(busyPlace)))  # 显示的是安排的workers

    # initialize the population
    for i in tqdm(range(particle_number)):
        # sample_worker = my_filtered_worker.sample(n=task_number)
        sample_worker = []
        arranged = []
        noArranged = copy.deepcopy(willWorkers)  # 在设置particle的时候用来保证每个位置安排的worker是互斥的
        for number in range(len(busyPlaceRange)):
            patientid = busyPlace[number]
            list1 = arranged
            list2 = locWillStructList[patientid - 1].visitorList
            #check if list1 contains all elements in list2
            result = all(elem in list1 for elem in list2)
            if result:  #说明这个地点能够安排的全都已经被安排了
                sample_worker.append(-1) # -1代表的是没有worker可以安排
                first_reveal[i][number] = -1
            else:
                common_list = list(set(noArranged).intersection(list2))
                workerid = random.sample(common_list, 1)[0]  #注意要加后面那个，不然就是一个list
                noArranged.remove(workerid)
                arranged.append(workerid)
                first_reveal[i][number] = workerid
                n = list2.index(workerid)
                sample_worker.append(n)
        PSO[i] = sample_worker
        PSO_best[i] = PSO[i]
        #速度初始化
        for l in range(len(busyPlaceRange)):
            velocity[i][l] = random.uniform(0.1, busyPlaceRange[l] / 3)

    # test
    # for i in tqdm(range(particle_number)):
    #     this_particle = PSO[i]
    #     checkList = []
    #     for indexl in range(len(this_particle)):
    #         patientid = busyPlace[indexl]
    #         workerOrder = this_particle[indexl]
    #         if workerOrder != -1:
    #             workerid = locWillStructList[patientid - 1].visitorList[int(workerOrder)]
    #             if workerid in checkList:
    #                 print("wrong")
    #             checkList.append(workerid)
    # test

    # 计算第一次的所有particle
    first_revealDf = pd.DataFrame(first_reveal)
    first_revealDf[first_revealDf < 0] = 0
    locDf = copy.deepcopy(integLocOrder)
    first_revealDf.columns = locDf.columns
    locDf.insert(0, "value", range(len(locDf), 0, -1))
    locName = list(integLocOrder.columns)
    locResult = pd.DataFrame()  # 每个particle安排的worker的地点对应位次
    for loc in locName:
        tempResult = first_revealDf.merge(locDf, how='left', on=[loc])
        locResult[loc] = tempResult['value']
    locResult = locResult.fillna(0)
    first_revealDf_T = first_revealDf.T
    first_revealDf_T = first_revealDf_T.reset_index(drop=True)
    # 属性操作
    attrResult = pd.DataFrame()
    particNameList = list(first_revealDf_T.columns)
    pure_task = task.iloc[:, 1:]
    for part in particNameList:
        temp = first_revealDf_T.merge(attrCountDf, how='left', left_on=[part], right_on='workerId')
        temp = temp.iloc[:, -attrNumber:]  # 因为是添加在最右边的
        temp = temp.fillna(0)
        temp = temp.mul(pure_task)
        attrResult[part] = temp.sum(axis=1)
    # 属性操作

    locResult = locResult.T  # 变成竖着的是一个particle
    locResult = locResult.reset_index(drop=True)

    # 属性操作
    attrResult = attrResult.reset_index(drop=True)
    particleResult = locResult.mul(attrResult)
    fitness_df = particleResult.sum(axis=0)
    fitness[:] = fitness_df
    fitness_local_best = copy.deepcopy(fitness)

    plenthDf = copy.deepcopy(first_revealDf)
    plenthDf[plenthDf > 0] = 1
    plenthDf = plenthDf.sum(axis=1)
    plenth[:] = plenthDf
    plenth_local_best = copy.deepcopy(plenth)

    # 挑选出来最好的fitness，从最长的里面进行挑选
    n_zeros = np.count_nonzero(PSO == -1, axis=1)
    winner = np.argwhere(n_zeros == np.amin(n_zeros)).flatten().tolist()
    bestLenth = len(busyPlace) - np.amin(n_zeros)
    for i in winner:
        if fitness[i] > global_fitness_best_value:  # 注意现在是好的数值大，所以是大于
            global_fitness_best_value = fitness[i]
            global_PSO_best = PSO[i]

    # # just for test
    # recordFile = "file_" + str(fileId) + "/" + "TFARRecord/round_" + str(roundtime) + "/init_" + ".csv"
    # pd.DataFrame(PSO).to_csv(recordFile, index=False)
    # rFile = "file_" + str(fileId) + "/" + "TFARRecord/round_" + str(roundtime) + "/initReveal_" + ".csv"
    # pd.DataFrame(first_reveal).to_csv(rFile, index=False)
    # # just for test

    # start the repeat
    for generation in tqdm(range(generation_number)):

        # update
        reveal_PSO = np.zeros((particle_number, len(busyPlace)))
        r1_array = np.random.rand(particle_number, 1)
        r2_array = np.random.rand(particle_number, 1)
        c1 = 0.3
        c2 = 0.3
        w = 0.9
        velocity = w * velocity + c1 * r1_array * (PSO_best - PSO) + c2 * r2_array * (global_PSO_best - PSO)
        PSO = np.floor(velocity + PSO)
        adj_busyPlaceRange = [x - 1 for x in busyPlaceRange]
        judge = np.array(busyPlaceRange) - 1 - PSO
        PSO = np.where(judge < 0, adj_busyPlaceRange, PSO)
        PSO = np.where(PSO < -1, -1, PSO)

        # check and evaluate fitness
        # 思路换成直接每个particle一起求解，然后对于重复的比较
        for i in range(particle_number):
            locIndList = list(range(0,0+location_number))  # 为了找映射的时候找到
            vCod = list(copy.deepcopy(PSO[i]))
            vCod = list(map(int,vCod))  # 作为索引需要是整数
            ans = wkMapPlace[locIndList,vCod]
            negIndex = list(np.where(PSO[i] == -1)[0])
            ans[negIndex] = -1    # 没有worker的地方需要没有
            ans_list1 = list(ans)
            PSO_reveal[i] = ans_list1

        # # for test
        # test_reveal = np.zeros((particle_number, len(busyPlace)))
        # for i in range(particle_number):
        #     this_particle = PSO[i]
        #     for indexl in range(len(busyPlace)):
        #         patientid = busyPlace[indexl]
        #         workerOrder = this_particle[indexl]
        #         if workerOrder != -1:
        #             workerid = locWillStructList[patientid - 1].visitorList[int(workerOrder)]
        #             test_reveal[i][indexl] = workerid
        #         else:
        #             test_reveal[i][indexl] = -1
        #
        # for i in range(particle_number):
        #     first = list(PSO_reveal[i])
        #     second = list(test_reveal[i])
        #     check = PSO_reveal[i] - test_reveal[i]
        #     if np.sum(check) == 0:
        #         continue
        #     else:
        #         print("wrong")
        #
        #
        # # for test end



        check_reveal = copy.deepcopy(PSO_reveal)
        checkRevealDf = pd.DataFrame(check_reveal)
        checkRevealDf[checkRevealDf < 0] = 0
        locDf = copy.deepcopy(integLocOrder)
        checkRevealDf.columns = locDf.columns
        locName = list(integLocOrder.columns)
        locDf.insert(0, "value", range(len(locDf), 0, -1))
        locResult = pd.DataFrame()  # 每个particle安排的worker的地点对应位次
        for loc in locName:
            tempResult = checkRevealDf.merge(locDf, how='left', on=[loc])
            locResult[loc] = tempResult['value']
        locResult = locResult.fillna(0)

        checkRevealDf_T = checkRevealDf.T
        checkRevealDf_T = checkRevealDf_T.reset_index(drop=True)

        # 属性操作
        attrResult = pd.DataFrame()
        particNameList = list(checkRevealDf_T.columns)
        pure_task = task.iloc[:, 1:]
        for part in particNameList:
            temp = checkRevealDf_T.merge(attrCountDf, how='left', left_on=[part], right_on='workerId')
            temp = temp.iloc[:, -attrNumber:]  # 因为是添加在最右边的
            temp = temp.fillna(0)
            temp = temp.mul(pure_task)
            attrResult[part] = temp.sum(axis=1)
        # 属性操作

        locResult = locResult.T  # 变成竖着的是一个particle
        locResult = locResult.reset_index(drop=True)

        # 属性操作
        attrResult = attrResult.reset_index(drop=True)
        particleResult = locResult.mul(attrResult)

        for i in tqdm(range(particle_number)):
            list1 = PSO_reveal[i]
            for dup in sorted(list_duplicates(list1)):
                workerid = dup[0]
                indexList = copy.deepcopy(dup[1])  # 这个worker安排在了第几个地点的list
                fitnessCompare = particleResult.iloc[indexList,i]  # 竖着的是particle
                maxCoIn = fitnessCompare.idxmax()  # 最好的地点是第几个
                indexList.remove(maxCoIn)
                for j in indexList:
                    PSO[i][j] = -1
                    PSO_reveal[i][j] = -1
                    particleResult.iat[j,i] = 0

            # for check
            tempReveal = list(PSO_reveal[i])
            dupResult = checkIfDuplicates_1(tempReveal)
            if dupResult:
                for dup in sorted(list_duplicates(tempReveal)):
                    workerid = dup[0]
                    if math.isclose(workerid,-1):
                        continue
                    indexList = copy.deepcopy(dup[1])
                    print("duplicate")
            else:
                pass

        # for check
        for i in tqdm(range(particle_number)):
            this_particle = PSO[i]
            checkList = []
            for indexl in range(len(this_particle)):
                patientid = busyPlace[indexl]
                workerOrder = this_particle[indexl]
                if workerOrder != -1:
                    workerid = locWillStructList[patientid - 1].visitorList[int(workerOrder)]
                    if workerid in checkList:
                        print("wrong")
                    checkList.append(workerid)
        # for check end

        psoRevealDf = pd.DataFrame(PSO_reveal)
        fitness_df = particleResult.sum(axis=0)
        fitness[:] = copy.deepcopy(fitness_df)

        plenthDf = copy.deepcopy(psoRevealDf)
        plenthDf[plenthDf > 0] = 1
        plenthDf = plenthDf.sum(axis=1)
        plenth[:] = plenthDf


        n_zeros = np.count_nonzero(PSO == -1, axis=1)
        Lenth = len(busyPlace) - np.amin(n_zeros)
        # 只有长度大于等于原来的时候才考虑
        if Lenth > bestLenth:
            bestLenth = Lenth
            winner = np.argwhere(n_zeros == np.amin(n_zeros)).flatten().tolist()
            this_bFitness = -1000
            besti = None
            for i in winner:
                if fitness[i] > this_bFitness:  # 注意是好的数值大
                    this_bFitness = fitness[i]
                    besti = i
            global_PSO_best = PSO[besti]
            global_fitness_best_value = this_bFitness
        elif Lenth == bestLenth:
            n_zeros = np.count_nonzero(PSO == -1, axis=1)
            winner = np.argwhere(n_zeros == np.amin(n_zeros)).flatten().tolist()
            for i in winner:
                if fitness[i] > global_fitness_best_value:  # 注意是好的数值大
                    global_fitness_best_value = fitness[i]
                    global_PSO_best = PSO[i]

        # for each particle 更新local最佳
        for i in range(particle_number):
            if plenth[i] > plenth_local_best[i]:
                plenth_local_best[i] = copy.deepcopy(plenth[i])
                fitness_local_best[i] = fitness[i]
                PSO_best[i] = PSO[i]
            elif plenth[i] == plenth_local_best[i]:
                if fitness[i] > fitness_local_best[i]:
                    fitness_local_best[i] = fitness[i]
                    PSO_best[i] = PSO[i]

        # # just for test
        # recordFile = "file_" + str(fileId) + "/" + "TFARRecord/round_" + str(
        #     roundtime) + "/generation_" + str(generation) + ".csv"
        # pd.DataFrame(PSO).to_csv(recordFile, index=False)
        # reveal_PSOFile = "file_" + str(fileId) + "/" + "TFARRecord/round_" + str(
        #     roundtime) + "/reveal_" + str(generation) + ".csv"
        # pd.DataFrame(reveal_PSO).to_csv(reveal_PSOFile, index=False)
        # # just for test

    best_result = copy.deepcopy(global_PSO_best)
    trustArrangement = pd.DataFrame()
    sentList = []
    locArrNorStructList = []  # 每个地方安排哪些普通前往
    for patientid in range(1, location_number + 1):
        locArrNorStructList.append(locArrVisit(patientid))

    for indexl in range(len(busyPlace)):
        patientid = busyPlace[indexl]
        workerOrder = best_result[indexl]
        if workerOrder != -1 :
            workerid = locWillStructList[patientid - 1].visitorList[int(workerOrder)]
            trustArrangement[patientid] = [workerid]
            if workerid in sentList:
                print("wrong")
            sentList.append(workerid)


    trustArrangementFile = "file_" + str(fileId) + "/" + "TFARworkerArrange/trustArrange_" + str(roundtime) +".csv"
    trustArrangement.to_csv(trustArrangementFile,index=False)

    rl_chooseNumber = 4
    normalArrangeDf = pd.DataFrame()
    for patientid in range(1,1+location_number):
        locwill = locWillStructList[patientid - 1].visitorList
        leftwill = list(set(locwill) - set(sentList))  # 获得的是这个地点没有被分配的workers
        if len(leftwill) < rl_chooseNumber:  # 说明这个地点所有的候选worker数量不够
            rl_chooseNumber = len(leftwill)

        this_choose = sample(leftwill,rl_chooseNumber)
        locArrNorStructList[patientid - 1].visitorList = this_choose
        sentList = sentList + this_choose
        normalArrangeDf[patientid] = this_choose + [0] * (worker_number - len(this_choose))

    normalArrangementFile = "file_" + str(fileId) + "/" + "TFARworkerArrange/normalArrange_" + str(roundtime) + ".csv"
    normalArrangeDf.to_csv(normalArrangementFile, index=False)


    return locArrNorStructList, trustArrangement





if __name__ == '__main__':
    workernumber = parameter.workerNumber
    fileId = 1
    locWillStructList = []
    round = 2
    locWillFilename = "file_" + str(fileId) + "/" + "workerApply/round_" + str(round) + "_locWill.json"
    wkFilename = "file_" + str(fileId) + "/" + "workerApply/round_" + str(round) + "_locWk.json"
    with open(locWillFilename, 'r') as openfile:
        locWillDataList = json.load(openfile)
    for locWdata in locWillDataList:
        locWillStructList.append(locWillVisit(locWdata["patientId"], locWdata["visitorList"]))
    locArrNorStructList, trustArrangement = TFARtaskAssignment(round, locWillStructList, fileId)

    # vice_point = 0.6
    # location_number = 10
    # UAV_check_number = 1
    # round = 5
    # # 接下来去做根据第一次的结果是安排任务的算法
    # scoresFile = "workerOverallScores.csv"
    # scores = pd.read_csv(scoresFile)
    # # 用来记录 workers的属性绝对偏好度，横的分别代表属性 Pregnancies,Glucose,BloodPressure,SkinThickness,BMI
    # attrScoresFile = "workerAttrScores.csv"
    # attrScores = pd.read_csv(attrScoresFile)
    # # 用来记录workers的地点绝对偏好度，横的代表地点
    # locScoresFile = "workerLocScores.csv"
    # locScores = pd.read_csv(locScoresFile)
    # # 上一轮的地点相对偏序关系
    # RelLocOrderFile = "workerOrder/workerRelativeLoc_" + str(round - 1) + ".csv"
    # RelLocOrder = pd.read_csv(RelLocOrderFile)
    #
    # sample_id = []  # 用来存储被检测的worker
    # applyRate = 0.9  # 申请任务的工人比例，平均每个地方 90/10 = 9
    # applyNumber = int(workernumber * applyRate)
    # workerList = random.sample(range(1, workernumber + 1), applyNumber)  # 申请的是哪些workers
    # LTOverall = scores.iloc[:, round - 1]
    # badWorkersVa = LTOverall[LTOverall < 0.1]  # 小于0.1认为不可信
    # badWorkers = list(badWorkersVa.index)
    #
    # Trustful = LTOverall[LTOverall > vice_point]  # 可信度大于vice_point则认为是可信workers
    # TrustfulWorkers = list(Trustful.index)
    #
    # leftWorkers = list(set(workerList) - set(badWorkers))  # 是申请的workers中把恶意的workers给排除了
    # willTrustful = list(set(workerList).intersection(TrustfulWorkers))  # 不仅是可信的worker，并且也是申请了的
    #
    # wkStructList = []  # 存储的是所有worker的结构体，结构体里面含有愿意去哪些地方的信息
    # locWillStructList = []  # 存储每个地点有哪些worker愿意前往的结构体，注意List是从0开始
    # locArrStructList = []  # 存储每个地点安排哪些worker前往的结构体，注意List是从0开始
    # # 对于两个list进行结构体初始化，每个list里面存储的都是结构体，结构体里面再存储哪些是愿意去的
    # for patientid in range(1, location_number + 1):
    #     locWillStructList.append(locWillVisit(patientid))
    #     locArrStructList.append(locArrVisit(patientid))
    #
    # for workerid in range(1,workernumber + 1):
    #     visLocNum = random.randint(1, 3)
    #     visLoc = random.sample(range(1, location_number + 1), visLocNum)
    #     wkStructList.append(worker(workerid, visLoc))  # 每个worker有自己的结构体，存储起来
    #     if workerid in leftWorkers:
    #         for patientid in visLoc:
    #             locWillStructList[patientid - 1].visitorList.append(workerid)  # 这边减一是因为从0开始的
    #
    # sentList = []
    # # create locations where UAV visits
    # UAV_check_number = 2



