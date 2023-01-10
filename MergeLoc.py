import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from multiprocessing import  Process
import copy
from random import sample
import parameter

def MergeLocF(thisround,fileId):
    locScoresFile = "file_" + str(fileId) + "/" + "workerLocScores.csv"
    locScores = pd.read_csv(locScoresFile)
    RelLocOrder = None
    workerNumber = parameter.workerNumber  # need change
    if thisround == 2:  # 因为只有第一轮有了
        RelLocOrderFile = "file_" + str(fileId) + "/" + "workerOrder/workerRelativeLoc_" + str(thisround - 1) + ".csv"
        RelLocOrder = pd.read_csv(RelLocOrderFile)
    else:
        RelLocOrder = pd.DataFrame()
        lastround = thisround - 1
        thisRelLocOrderFile = "file_" + str(fileId) + "/" + "workerOrder/workerRelativeLoc_" + str(thisround - 1) + ".csv"  # this指的是本轮次上一轮
        thisRelLocOrder = pd.read_csv(thisRelLocOrderFile)

        lastRelLocOrderFile = "file_" + str(fileId) + "/" + "workerOrder/workerRelativeLoc_" + str(lastround - 1) + ".csv"  #指的是本轮次的上一轮的上一轮
        lastRelLocOrder = pd.read_csv(lastRelLocOrderFile)

        #相对偏好度新一轮没有的直接带入上一轮次的
        locationName = list(thisRelLocOrder.columns)
        allWorkerList = list(range(1, workerNumber + 1))
        for name in locationName:
            # name = int(name)
            if (thisRelLocOrder[name] == 0 ).all() and (lastRelLocOrder[name] == 0).all():  #当两轮都没有的时候
                RelLocOrder[name] = lastRelLocOrder[name]
            elif (thisRelLocOrder[name] == 0 ).all(): #当这一轮没有的时候
                RelLocOrder[name] = lastRelLocOrder[name]
            elif (lastRelLocOrder[name] == 0).all(): #当上一轮没有的时候
                RelLocOrder[name] = thisRelLocOrder[name]
            else: #当两轮都有的时候
                thisOrder = thisRelLocOrder[name].tolist()
                zeroIndex = thisOrder.index(0)
                thisOrder = thisOrder[:zeroIndex]

                lastOrder = lastRelLocOrder[name].tolist()
                zeroIndex = lastOrder.index(0)
                lastOrder = lastOrder[:zeroIndex]
                common_list = list(set(thisOrder).intersection(lastOrder))
                allWorkerList = list(range(1, workerNumber + 1))
                if len(common_list) == 0:  # 没有交集的话就不连接了
                    combine = thisOrder
                    RelLocOrder[name] = combine + [0] * (workerNumber - len(combine))
                elif len(common_list) == 1:
                    workerid = common_list[0]
                    if thisOrder.index(workerid) == len(thisOrder) - 1:  # thisorder在前面
                        firstPart = thisOrder + lastOrder[lastOrder.index(workerid) + 1:]  # +1是为了不出现重复项
                        combine = firstPart
                    elif thisOrder.index(workerid) == 0:
                        firstPart = lastOrder[:lastOrder.index(workerid)] + thisOrder
                        combine = firstPart
                    else:  # 不在本次的首尾就不连接了
                        combine = thisOrder
                    RelLocOrder[name] = combine + [0] * (workerNumber - len(combine))
                else:
                    # 思路是先放两边，再填充中间
                    combine = []
                    if thisOrder[-1] in common_list:
                        value = thisOrder[-1]
                        copy_common = copy.deepcopy(common_list)
                        copy_common.remove(value)
                        last_order = [x for x in lastOrder if x not in copy_common]
                        firstPart = thisOrder + last_order[last_order.index(value) + 1:]  # +1是为了不出现重复项
                        combine = firstPart

                    elif thisOrder[0] in common_list:
                        value = thisOrder[0]
                        copy_common = copy.deepcopy(common_list)
                        copy_common.remove(value)
                        last_order = [x for x in lastOrder if x not in copy_common]
                        firstPart = last_order[:last_order.index(value)] + thisOrder
                        combine = firstPart
                    else:
                        combine = thisOrder

                    for i in range(len(thisOrder) - 1):
                        if thisOrder[i] in common_list and thisOrder[i + 1] in common_list:
                            index1 = lastOrder.index(thisOrder[i])
                            index2 = lastOrder.index(thisOrder[i + 1])
                            if index2 > index1 + 1:
                                part = lastOrder[index1 + 1: index2]
                                part = [x for x in part if x not in combine]

                                index_1 = combine.index(thisOrder[i])
                                combine = combine[:index_1 + 1] + part + combine[index_1 + 1:]

                    RelLocOrder[name] = combine + [0] * (workerNumber - len(combine))

        # 例如在第2轮的时候要把第1轮的相对偏序和第0轮的相对偏序组合起来，得到一个最新的历史综合相对偏序
        # 第3轮的时候要把第2轮的历史综合相对偏序和第2轮自己探测到的相对偏序组合起来得到第3轮的历史综合相对偏序
        # 意思也就是相对偏序关系要是现在的和历史的进行组合，然后给后面使用

    RelLocOrderFile = "file_" + str(fileId) + "/" + "workerOrder/workerRelativeLoc_" + str(thisround - 1) + ".csv"
    RelLocOrder.to_csv(RelLocOrderFile,index=False)  # 将上一次的地点相对偏序关系进行了更新

    locationName = RelLocOrder.columns
    locTime_co = 0.003

    initScore = 0.5
    lastScore = initScore

    finalScore = None
    for roundtime in range(1, thisround):
        locScore = lastScore
        fai = locTime_co * (1 - locScore)
        finalScore = locScore - fai
        lastScore = finalScore

    # 获得了当前timeout计算的值
    absLocScore = copy.deepcopy(locScores)
    absLocScore.insert(0, 'workerId', range(0, 0 + len(locScores)))
    mask = np.isclose(absLocScore, finalScore)
    absLocScore.mask(mask, other=np.NAN, inplace=True)
    absLocScore = absLocScore.dropna(axis=1, how='all')
    absLocScore = absLocScore.iloc[1:,:]
    columnName = absLocScore.columns[1:]  # 去除第一个workerId


    absResult = pd.DataFrame()
    for column in columnName:
        tempScores = absLocScore.sort_values(by=[column], ascending=False)  # descend
        tempScores = tempScores.reset_index(drop=True)
        tempScores.loc[tempScores[column] > 0, column] = 1
        absResult[column] = tempScores[column] * tempScores['workerId']

    absResultFile = "file_" + str(fileId) + "/" + "workerLocAbOrder.csv"
    absResult = absResult.fillna(0)
    absResult.to_csv(absResultFile, index=False)

    absColumn = absResult.columns


    integralResult = pd.DataFrame()
    allWorkerList = list(range(1, workerNumber + 1))
    for name in locationName:
        if name in absColumn:  # 说明这个地点已经建立了绝对信任度
            if (RelLocOrder[name] == 0).all():  # 这个地点只有绝对可信度但是没有相对可信度
                col_list = absResult[name].tolist()
                zeroIndex = col_list.index(0)
                firstPart = col_list[:zeroIndex]
                secondPart = list(set(allWorkerList) - set(firstPart))
                # 注意不要加前面的
                random.shuffle(secondPart)
                combine = firstPart + secondPart
                integralResult[name] = combine
            else:  # 这个地点有相对可信度和绝对可信度
                relOrder = RelLocOrder[name].tolist()
                if 0 in relOrder:
                    zeroIndex = relOrder.index(0)
                    relOrder = relOrder[:zeroIndex]

                absOrder = absResult[name].tolist()
                if 0 in absOrder:
                    zeroIndex = absOrder.index(0)
                    absOrder = absOrder[:zeroIndex]



                common_list = list(set(relOrder).intersection(absOrder))
                allWorkerList = list(range(1, workerNumber + 1))
                if len(common_list) == 0:  # 没有交集的话就不连接了
                    combine = absOrder
                    secondPart = list(set(allWorkerList) - set(combine))
                    random.shuffle(secondPart)
                    integralResult[name] = combine + secondPart
                elif len(common_list) == 1:
                    workerid = common_list[0]
                    combine = None
                    if absOrder.index(workerid) == len(absOrder) - 1:  # absorder在前面
                        firstPart = absOrder + relOrder[relOrder.index(workerid) + 1:]  # +1是为了不出现重复项
                        combine = firstPart
                    elif absOrder.index(workerid) == 0:  # absOrder在后面
                        firstPart = relOrder[:relOrder.index(workerid)] + absOrder
                        combine = firstPart
                    else:  # 不在本次的首尾就不连接了,直接是absOrder
                        combine = absOrder

                    secondPart = list(set(allWorkerList) - set(combine))
                    random.shuffle(secondPart)
                    integralResult[name] = combine + secondPart
                else:
                    # 思路是先放两边，再填充中间
                    combine = []
                    if absOrder[-1] in common_list:
                        value = absOrder[-1]
                        copy_common = copy.deepcopy(common_list)
                        copy_common.remove(value)
                        rel_Order = copy.deepcopy(relOrder)
                        rel_Order = [x for x in rel_Order if x not in copy_common]
                        firstPart = absOrder + rel_Order[rel_Order.index(value) + 1:]  # +1是为了不出现重复项
                        combine = firstPart

                    elif absOrder[0] in common_list:
                        value = absOrder[0]
                        copy_common = copy.deepcopy(common_list)
                        copy_common.remove(value)
                        rel_Order = copy.deepcopy(relOrder)  # 下面还会使用relOrder,所以这边记住需要copy
                        rel_Order = [x for x in rel_Order if x not in copy_common]
                        firstPart = rel_Order[:rel_Order.index(value)] + absOrder
                        combine = firstPart
                    else:
                        combine = absOrder

                    for i in range(len(absOrder) - 1):
                        if absOrder[i] in common_list and absOrder[i + 1] in common_list:
                            index1 = relOrder.index(absOrder[i])
                            index2 = relOrder.index(absOrder[i + 1])
                            part = relOrder[index1 + 1: index2]
                            part = [x for x in part if x not in combine]

                            index_1 = combine.index(absOrder[i])
                            combine = combine[:index_1 + 1] + part + combine[index_1 + 1:]

                    secondPart = list(set(allWorkerList) - set(combine))
                    random.shuffle(secondPart)
                    integralResult[name] = combine + secondPart


        else:  # 这个地点没有绝对信任度那么就是相对信任度
            col_list = RelLocOrder[name].tolist()
            zeroIndex = col_list.index(0)
            firstPart = col_list[:zeroIndex]
            secondPart = list(set(allWorkerList) - set(firstPart))
            random.shuffle(secondPart)   #对于不知道的进行随机的处理
            combine = firstPart + secondPart
            integralResult[name] = combine

    #  this round 任务分配的时候根据历史两轮次得出来的综合的历史地点可信度偏序关系，也就是截止到
    # this round -1 的综合排序
    integralResultFile = "file_" + str(fileId) + "/" + "workerOrder/integLoc_" + str(thisround - 1) + ".csv"
    integralResult.to_csv(integralResultFile,index=False)
    return integralResult


if __name__ == '__main__':
    MergeLocF(1,1)
#     locScoresFile = "workerLocScores.csv"
#     locScores = pd.read_csv(locScoresFile)
#     thisround = 1  # 设置成为函数的时候需要修改
#     RelLocOrder = None
#     if thisround == 1:
#         RelLocOrderFile = "workerOrder/workerRelativeLoc_"+ str(thisround - 1) +".csv"
#         RelLocOrder = pd.read_csv(RelLocOrderFile)
#     else:
#         # 例如在第2轮的时候要把第1轮的相对偏序和第0轮的相对偏序组合起来，得到一个最新的历史综合相对偏序
#         # 第3轮的时候要把第2轮的历史综合相对偏序和第2轮自己探测到的相对偏序组合起来得到第3轮的历史综合相对偏序
#         # 意思也就是相对偏序关系要是现在的和历史的进行组合，然后给后面使用
#         pass
#
#     locationName = RelLocOrder.columns
#     locTime_co = 0.003
#
#     initScore = 0.5
#     lastScore = initScore
#     workerNumber = 100
#     finalScore = None
#     for roundtime in range(0,thisround):
#         locScore = initScore
#         fai = locTime_co * (1 - locScore)
#         finalScore = locScore - fai
#     # 获得了当前timeout计算的值
#     absLocScore = copy.deepcopy(locScores)
#     absLocScore.insert(0, 'workerId', range(0, 0 + len(locScores)))
#     absLocScore = absLocScore.replace(to_replace=finalScore, value= np.NAN).iloc[1:,:]
#     absLocScore = absLocScore.dropna(axis=1, how='all')
#     columnName = absLocScore.columns[1:] #去除第一个workerId
#     absResult = pd.DataFrame()
#     for column in columnName:
#         absLocScore = absLocScore.sort_values(by=[column], ascending=False)  # descend
#         absLocScore.loc[absLocScore[column] > 0, column] = 1
#         absResult[column] = absLocScore[column] * absLocScore['workerId']
#     print()
#     absResultFile = "workerLocAbOrder.csv"
#     absResult = absResult.fillna(0)
#     absResult.to_csv(absResultFile,index=False)
#     absColumn = absResult.columns
#     integralResult = pd.DataFrame()
#     allWorkerList = list(range(1, workerNumber + 1))
#     for name in locationName:
#         if name in absColumn:  #说明这个地点已经建立了绝对信任度
#            if (RelLocOrder[name] == 0).all():  #这个地点只有绝对可信度但是没有相对可信度
#                col_list = absResult[name].tolist()
#                zeroIndex = col_list.index(0)
#                firstPart = col_list[:zeroIndex]
#                secondPart = list(set(allWorkerList) - set(firstPart))
#                secondPart = random.shuffle(secondPart)
#                combine = firstPart + secondPart
#                integralResult[name] = combine
#            else: #这个地点有相对可信度和绝对可信度
#                 pass #第0轮暂时没有出现
#
#
#         else:  #这个地点没有绝对信任度那么就是相对信任度
#             col_list = RelLocOrder[name].tolist()
#             zeroIndex = col_list.index(0)
#             firstPart = col_list[:zeroIndex]
#             secondPart = list(set(allWorkerList) - set(firstPart))
#             secondPart = random.shuffle(secondPart)
#             combine = firstPart + secondPart
#             integralResult[name] = combine
#
#     # 根据 this round 得出来的综合的历史地点可信度偏序关系
#     integralResultFile = "workerOrder/integLoc_" + str(thisround) + ".csv"
#     integralResult.to_csv(integralResultFile)










