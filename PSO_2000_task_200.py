import copy

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import math
import time
from multiprocessing import  Process
from MergeLoc import MergeLocF
from getAttrOrder import getAttrOrderF
from random import sample

from collections import defaultdict
import parameter

def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items()
                            if len(locs)>1)


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
def taskAssignment(roundtime, UAV_check_number,leftWorkers:list, willTrustful:list,wkStructList,fileId):

    worker_number = parameter.workerNumber
    punish_co = 2
    task_number = parameter.taskNumber
    particle_number = 2 * task_number
    location_number = parameter.locationNumber

    filename = "task_" + str(task_number) + ".csv"
    task = pd.read_csv(filename)


    # 获取当前轮次历史地点偏序关系组合
    integLocOrder = MergeLocF(roundtime,fileId)
    attrOrder = getAttrOrderF(roundtime,fileId)
    attrName = list(task.columns)[1:]
    attrNumber = len(attrName)
    UAV_place = []

    # 属性操作
    attrDf = copy.deepcopy(attrOrder)
    attrDf.columns = attrName  # 这边文件结构和TFAR那边不一样，这边需要添加一下。
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


    locTrustWillStructList = []  #存储每个地点有哪些是可信的worker愿意去的
    for patientid in range(1, location_number + 1): #初始化
        locTrustWillStructList.append(locWillVisit(patientid))


    #先安排可信的workers
    emptyPlace = list(range(1,location_number + 1))  # 记录的是没有可信workers愿意前往的地方
    busyPlace = list(range(1,location_number + 1))   # 记录的是有可信workers愿意前往的地方
    # 更新了每个地点是有哪些可信worker
    for workerid in willTrustful:
        visLocation = wkStructList[workerid - 1].location
        for patientid in visLocation:
            locTrustWillStructList[patientid - 1].visitorList.append(workerid)
            if patientid in emptyPlace:
                emptyPlace.remove(patientid)

    busyPlace = list(map(int, busyPlace))
    emptyPlace = list(map(int, emptyPlace))

    busyPlace = list(set(busyPlace) - set(emptyPlace))
    wkMapPlace = np.zeros((len(busyPlace), worker_number))  # 建立第几个busyPlace里面的映射关系



    if len(busyPlace) <= location_number - UAV_check_number:
        locList = list(range(1, location_number + 1))
        chooseList = list(set(locList) - set(busyPlace))
        UAV_place = sample(chooseList, UAV_check_number)
    else:
        locList = list(range(1, location_number + 1))
        chooseList = list(set(locList) - set(busyPlace))
        chooseNumber = len(chooseList)
        leftNumber = UAV_check_number - chooseNumber
        UAV_place = sample(busyPlace,leftNumber) + chooseList # 此时可信worker比较多，那么就选择一部分，剩下来的依旧还会使由UAV前去

    busyPlace = list(set(busyPlace) - set(UAV_place))   # 保证了 可信worker前去的和UAV前去的不是一个地方
    # 如果是可信worker愿意去同时也是UAV去的那就没有必要
    UAV_place_df = pd.DataFrame(UAV_place)
    UAV_place_dfFile = "file_" + str(fileId) + "/" + "workerArrange/UAVArrange_" + str(roundtime) +".csv"
    UAV_place_df.to_csv(UAV_place_dfFile,index=False)



    if busyPlace!=[] :  #有可以安排的可信的workers
        # 对于busyPlace进行PSO算法
        busyPlaceRange = [0] * len(busyPlace)
        PSO = np.zeros((particle_number, len(busyPlace)))
        PSO_reveal = np.zeros((particle_number, len(busyPlace)))  # 显示的是安排的workers
        PSO_best = np.zeros((particle_number, len(busyPlace)))
        velocity = np.zeros((particle_number, len(busyPlace)))
        fitness = np.zeros(particle_number)
        plenth = np.zeros(particle_number)
        fitness_local_best = np.zeros(particle_number)
        plenth_local_best = np.zeros(particle_number)

        global_PSO_best = np.zeros(len(busyPlace))  # means global over the history
        global_fitness_best_value = -1000

        #记录每个地方是有几个可信worker
        for i in range(0,len(busyPlace)):
            patientid = busyPlace[i]
            busyPlaceRange[i] = len(locTrustWillStructList[patientid - 1].visitorList)
            wkMapPlace[i] = locTrustWillStructList[patientid - 1].visitorList + [0] * (worker_number - len(locTrustWillStructList[patientid - 1].visitorList))

        generation_number = 120
        best_particle_record = np.zeros((generation_number, task_number))
        bestLenth = 0 # 记录的是最多的能有可信worker前往的地点数量

        first_reveal = np.zeros((particle_number, len(busyPlace)))  # 显示的是安排的workers

        # initialize the population
        for i in tqdm(range(particle_number)):
            # sample_worker = my_filtered_worker.sample(n=task_number)
            sample_worker = []
            arranged = []
            noArranged = copy.deepcopy(willTrustful)  # 在设置particle的时候用来保证每个位置安排的worker是互斥的
            for number in range(len(busyPlaceRange)):
                patientid = busyPlace[number]
                list1 = arranged
                list2 = locTrustWillStructList[patientid - 1].visitorList
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

        # 计算第一次的所有particle


        first_revealDf = pd.DataFrame(first_reveal)
        first_revealDf[first_revealDf < 0] = 0
        locDf = copy.deepcopy(integLocOrder)
        locName = copy.deepcopy(busyPlace)
        locName = [x - 1 for x in locName]  # 这个操作是因为在文件中的表头，地点是从0-9
        locName = list(map(str, locName))
        first_revealDf.columns = locName
        locDf.insert(0, "value", range(len(locDf), 0, -1))
        locResult = pd.DataFrame()  # 每个particle安排的worker的地点对应位次
        for loc in locName:
            tempResult = first_revealDf.merge(locDf, how='left', on=[loc])
            locResult[loc] = tempResult['value']
        locResult = locResult.fillna(0)  # 没有安排是-1的时候会是nan
        first_revealDf_T = first_revealDf.T
        first_revealDf_T = first_revealDf_T.reset_index(drop=True)

        # 属性操作
        attrResult = pd.DataFrame()
        particNameList = list(first_revealDf_T.columns)
        bpDf = pd.DataFrame(busyPlace)
        bpDf.columns = ['PatientId']
        pure_task = bpDf.merge(task, on='PatientId', how='left')  # 保持了原有busyPlace的顺序
        pure_task = pure_task.iloc[:, 1:]

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

        n_zeros = np.count_nonzero(PSO == -1, axis=1)
        winner = np.argwhere(n_zeros == np.amin(n_zeros)).flatten().tolist()
        bestLenth = len(busyPlace) - np.amin(n_zeros)
        for i in winner:
            if fitness[i] > global_fitness_best_value:  # 注意现在是好的数值大，所以是大于
                global_fitness_best_value = fitness[i]
                global_PSO_best = PSO[i]


        # start the repeat
        for generation in tqdm(range(generation_number)):
            reveal_PSO = np.zeros((particle_number, len(busyPlace)))

            # update
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
            # 思路换成直接所有particle一起求解，然后对于重复的比较
            for i in range(particle_number):
                locIndList = list(range(len(busyPlace)))  # 为了找映射的时候找到
                vCod = list(copy.deepcopy(PSO[i]))
                vCod = list(map(int, vCod))  # 作为索引需要是整数
                ans = wkMapPlace[locIndList, vCod]
                negIndex = list(np.where(PSO[i] == -1)[0])
                ans[negIndex] = -1  # 没有worker的地方需要没有
                ans_list1 = list(ans)
                PSO_reveal[i] = ans_list1

            check_reveal = copy.deepcopy(PSO_reveal)
            checkRevealDf = pd.DataFrame(check_reveal)
            checkRevealDf[checkRevealDf < 0] = 0
            locDf = copy.deepcopy(integLocOrder)
            locName = copy.deepcopy(busyPlace)
            locName = [x - 1 for x in locName]  # 这个操作是因为在文件中的表头，地点是从0-9
            locName = list(map(str, locName))
            checkRevealDf.columns = locName
            locDf.insert(0, "value", range(len(locDf), 0, -1))
            locResult = pd.DataFrame()  # 每个particle安排的worker的地点对应位次
            for loc in locName:
                tempResult = checkRevealDf.merge(locDf, how='left', on=[loc])
                locResult[loc] = tempResult['value']
            locResult = locResult.fillna(0)  # 没有安排是-1的时候会是nan
            checkRevealDf_T = checkRevealDf.T
            checkRevealDf_T = checkRevealDf_T.reset_index(drop=True)

            # 属性操作
            attrResult = pd.DataFrame()
            particNameList = list(checkRevealDf_T.columns)
            bpDf = pd.DataFrame(busyPlace)
            bpDf.columns = ['PatientId']
            pure_task = bpDf.merge(task, on='PatientId', how='left')  # 保持了原有busyPlace的顺序
            pure_task = pure_task.iloc[:, 1:]

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

            for i in range(particle_number):
                list1 = PSO_reveal[i]
                for dup in sorted(list_duplicates(list1)):
                    workerid = dup[0]
                    indexList = copy.deepcopy(dup[1])  # 这个worker安排在了第几个地点的list
                    fitnessCompare = particleResult.iloc[indexList, i]  # 竖着的是particle
                    maxCoIn = fitnessCompare.idxmax()  # 最好的地点是第几个
                    indexList.remove(maxCoIn)
                    for j in indexList:
                        PSO[i][j] = -1
                        PSO_reveal[i][j] = -1
                        particleResult.iat[j, i] = 0

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



        best_result = copy.deepcopy(global_PSO_best)
        trustArrangement = pd.DataFrame()

        noBadWorkers = copy.deepcopy(leftWorkers) # 用来安排普通workers过去被检验
        needArrangeL = list(range(1,1 + location_number))
        for indexl in range(len(busyPlace)):
            patientid = busyPlace[indexl]
            workerOrder = best_result[indexl]
            if workerOrder != -1 :
                workerid = locTrustWillStructList[patientid - 1].visitorList[int(workerOrder)]
                trustArrangement[patientid] = [workerid]
                noBadWorkers.remove(workerid)
                needArrangeL.remove(patientid)

        # 去掉了可信workers之后，再去掉UAV访问的地方
        needArrangeL = list(set(needArrangeL) - set(UAV_place))

        # 可信workers和UAV前往的地方随机安排workers前往就可以了，因为只需要检验数据就行了
        randomArrangeL = list(set(list(range(1,1 + location_number))) - set(needArrangeL))

        trustArrangementFile = "file_" + str(fileId) + "/" + "workerArrange/trustArrange_" + str(roundtime) +".csv"
        trustArrangement.to_csv(trustArrangementFile,index=False)

        locnbStructList = []  #每个地方有哪些noBadworkers愿意前往
        sentList = [] #已经分配的普通workers
        locArrStructList = []  #每个地方安排哪些noBadworkers前往
        for patientid in range(1, location_number + 1):
            locArrStructList.append(locArrVisit(patientid))
            locnbStructList.append(locArrVisit(patientid))

        # 安排没有UAV和可信worker前往的地点
        for workerid in noBadWorkers:
            visLocation = wkStructList[workerid - 1].location
            for patientid in visLocation:
                locnbStructList[patientid - 1].visitorList.append(workerid)

        normalArrangeDf = pd.DataFrame()
        nl_chooseNumber = 3  # 每个地点派遣3个worker前往
        scoresFile = "file_" + str(fileId) + "/" + "workerOverallScores.csv"
        scores = pd.read_csv(scoresFile)
        RoundScores = scores.iloc[:, roundtime - 1]
        RoundScoresDf = pd.DataFrame(RoundScores).reset_index(drop=True)
        RoundScoresDf.insert(0, 'workerId', range(1, 1 + len(RoundScores)))


        for patientid in needArrangeL:
            locwill = locnbStructList[patientid - 1].visitorList
            leftwill = list(set(locwill) - set(sentList))  # 获得的是这个地点没有被分配的workers
            if len(leftwill) < nl_chooseNumber:  # 说明这个地点所有的候选worker数量不够
                nl_chooseNumber = len(leftwill)

            workerDf = pd.DataFrame(leftwill, columns=['workerId', ])
            worker_score = pd.merge(workerDf, RoundScoresDf, how='inner', on='workerId')
            worker_score.columns = ['workerId', 'value']
            worker_score['rand_col'] = np.random.randint(1, 1 + len(worker_score), worker_score.shape[0])
            worker_score = worker_score.sort_values(by=['value', "rand_col"], ascending=[False, False])
            sortedId = list(worker_score['workerId'])
            this_choose = sortedId[:nl_chooseNumber]  # 按照得分的高低来进行选择
            locArrStructList[patientid - 1].visitorList = this_choose
            sentList = sentList + this_choose
            normalArrangeDf[patientid] = this_choose + [0] * (worker_number - len(this_choose))


        rl_chooseNumber = 0
        if len(willTrustful) < 360:
            rl_chooseNumber = 4
        elif len(willTrustful) < 600:
            rl_chooseNumber = 3
        elif len(willTrustful) < 840:
            rl_chooseNumber = 2
        elif len(willTrustful) < 1000:
            rl_chooseNumber = 1
        else:
            rl_chooseNumber = 0


        for patientid in randomArrangeL:
            locwill = locnbStructList[patientid - 1].visitorList
            leftwill = list(set(locwill) - set(sentList))  # 获得的是这个地点没有被分配的workers
            if len(leftwill) < rl_chooseNumber:  # 说明这个地点所有的候选worker数量不够
                rl_chooseNumber = len(leftwill)


            workerDf = pd.DataFrame(leftwill, columns=['workerId', ])
            worker_score = pd.merge(workerDf, RoundScoresDf, how='inner', on='workerId')
            worker_score.columns = ['workerId', 'value']
            worker_score['rand_col'] = np.random.randint(1, 1 + len(worker_score), worker_score.shape[0])
            worker_score = worker_score.sort_values(by=['value', "rand_col"], ascending=[False, False])
            sortedId = list(worker_score['workerId'])
            this_choose = sortedId[:rl_chooseNumber]  # 按照得分的高低来进行选择
            locArrStructList[patientid - 1].visitorList = this_choose
            sentList = sentList + this_choose
            normalArrangeDf[patientid] = this_choose + [0] * (worker_number - len(this_choose))



        normalArrangementFile = "file_" + str(fileId) + "/" + "workerArrange/normalArrange_" + str(roundtime) + ".csv"
        normalArrangeDf.to_csv(normalArrangementFile, index=False)

        return locArrStructList,trustArrangement,UAV_place

    # 安排的普通的workers
    else: # 当可信的workers为0的时候
        locArrStructList = []  # 每个地方安排哪些noBadworkers前往
        locnbStructList = []  # 每个地方有哪些noBadworkers愿意前往

        scoresFile = "file_" + str(fileId) + "/" + "workerOverallScores.csv"
        scores = pd.read_csv(scoresFile)
        RoundScores = scores.iloc[:, roundtime - 1]
        RoundScoresDf = pd.DataFrame(RoundScores).reset_index(drop=True)
        RoundScoresDf.insert(0, 'workerId', range(1, 1 + len(RoundScores)))

        for patientid in range(1, location_number + 1):
            locArrStructList.append(locArrVisit(patientid))
            locnbStructList.append(locArrVisit(patientid))

        noBadWorkers = copy.deepcopy(leftWorkers)  # 用来安排普通workers过去被检验
        for workerid in noBadWorkers:
            visLocation = wkStructList[workerid - 1].location
            for patientid in visLocation:
                locnbStructList[patientid - 1].visitorList.append(workerid)

        sentList = []
        chooseNumber = 5  # 每个地点派遣5个worker前往
        normalArrangeDf = pd.DataFrame()
        for patientid in range(1, location_number + 1):
            locwill = locnbStructList[patientid - 1].visitorList
            leftwill = list(set(locwill) - set(sentList))  # 获得的是没有被分配的workers
            if len(leftwill) < chooseNumber:  # 说明这个地点所有的候选worker数量不够
                chooseNumber = len(leftwill)

            workerDf = pd.DataFrame(leftwill, columns=['workerId', ])
            worker_score = pd.merge(workerDf, RoundScoresDf, how='inner', on='workerId')
            worker_score.columns = ['workerId', 'value']
            worker_score['rand_col'] = np.random.randint(1, 1 + len(worker_score), worker_score.shape[0])
            worker_score = worker_score.sort_values(by=['value', "rand_col"], ascending=[False, False])
            sortedId = list(worker_score['workerId'])  # 注意得分相同的要乱序
            this_choose = sortedId[:chooseNumber]  # 按照得分的高低来进行选择
            locArrStructList[patientid - 1].visitorList = this_choose
            sentList = sentList + this_choose
            normalArrangeDf[patientid] = this_choose + [0] * (worker_number - len(this_choose))

        normalArrangementFile = "file_" + str(fileId) + "/" + "workerArrange/normalArrange_" + str(roundtime) + ".csv"
        normalArrangeDf.to_csv(normalArrangementFile, index=False)
        trustArrangement = pd.DataFrame()
        trustArrangementFile = "file_" + str(fileId) + "/" + "workerArrange/trustArrange_" + str(roundtime) + ".csv"
        trustArrangement.to_csv(trustArrangementFile, index=False)

        return locArrStructList,trustArrangement,UAV_place





if __name__ == '__main__':

    workernumber = parameter.workerNumber
    vice_point = 0.5
    location_number = parameter.locationNumber
    UAV_check_number = 1
    round = 3
    # 接下来去做根据第一次的结果是安排任务的算法
    scoresFile = "file_1/workerOverallScores.csv"
    scores = pd.read_csv(scoresFile)


    sample_id = []  # 用来存储被检测的worker
    applyRate = 0.9  # 申请任务的工人比例，平均每个地方 90/10 = 9
    applyNumber = int(workernumber * applyRate)
    workerList = random.sample(range(1, workernumber + 1), applyNumber)  # 申请的是哪些workers
    LTOverall = scores.iloc[:, round - 1]
    badWorkersVa = LTOverall[LTOverall < 0.1]  # 小于0.1认为不可信
    badWorkers = list(badWorkersVa.index)

    Trustful = LTOverall[LTOverall > vice_point]  # 可信度大于vice_point则认为是可信workers
    TrustfulWorkers = list(Trustful.index)

    leftWorkers = list(set(workerList) - set(badWorkers))  # 是申请的workers中把恶意的workers给排除了
    willTrustful = list(set(workerList).intersection(TrustfulWorkers))  # 不仅是可信的worker，并且也是申请了的

    wkStructList = []  # 存储的是所有worker的结构体，结构体里面含有愿意去哪些地方的信息
    locWillStructList = []  # 存储每个地点有哪些worker愿意前往的结构体，注意List是从0开始
    locArrStructList = []  # 存储每个地点安排哪些worker前往的结构体，注意List是从0开始
    # 对于两个list进行结构体初始化，每个list里面存储的都是结构体，结构体里面再存储哪些是愿意去的
    for patientid in range(1, location_number + 1):
        locWillStructList.append(locWillVisit(patientid))
        locArrStructList.append(locArrVisit(patientid))

    for workerid in range(1,workernumber + 1):
        visLocNum = random.randint(1, 3)
        visLoc = random.sample(range(1, location_number + 1), visLocNum)
        wkStructList.append(worker(workerid, visLoc))  # 每个worker有自己的结构体，存储起来
        if workerid in leftWorkers:
            for patientid in visLoc:
                locWillStructList[patientid - 1].visitorList.append(workerid)  # 这边减一是因为从0开始的

    sentList = []
    # create locations where UAV visits
    UAV_check_number = 2
    taskAssignment(round,UAV_check_number,leftWorkers,willTrustful,wkStructList,1)



