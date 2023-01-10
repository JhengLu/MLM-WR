import numpy as np
import random
from tqdm import tqdm
from multiprocessing import Process
import copy
from random import sample
import pandas as pd
import parameter

def TFARMergeAttrF(thisround,fileId):
    workerNumber = parameter.workerNumber
    locationNumber = parameter.locationNumber
    attrOrder = pd.DataFrame()
    task_number = parameter.taskNumber
    filename = "task_" + str(task_number) + ".csv"
    task = pd.read_csv(filename)
    attriNameList = list(task.columns)[1::]

    # 此操作是不考虑历史情况
    if  True:  # 因为只有第一轮有了thisround == 2
        attrInte = pd.DataFrame()
        crossInte = pd.DataFrame()
        for name in attriNameList:
            thisAttr = []
            for patientid in range(1,1+locationNumber):
                RelAttrOrderFile = "file_" + str(fileId) + "/" + "TFARworkerOrder/workerRelativeAttr_" + str(thisround-1) + "_" + str(patientid) + ".csv"
                RelAttrOrder = pd.read_csv(RelAttrOrderFile)
                if (RelAttrOrder[str(attriNameList.index(name))] == 0).all():
                    continue
                else:
                    order = list(RelAttrOrder[str(attriNameList.index(name))])
                    order = [aaa for aaa in order if aaa != 0]
                    thisAttr.append(order)
            thisResult = []
            thisAttr_c = copy.deepcopy(thisAttr)
            whileNumber = len(thisAttr_c)
            while whileNumber > 0 :
                for i in range(len(thisAttr_c)):
                    if len(thisAttr[i]) > 1:
                        thisResult.append(thisAttr[i][0])
                        thisAttr[i] = thisAttr[i][1:]
                    elif len(thisAttr[i]) == 1:
                        thisResult.append(thisAttr[i][0])
                        thisAttr[i] = []
                        whileNumber = whileNumber - 1
                    else:
                        continue
            crossInte[name] = thisResult + [0] * (workerNumber - len(thisResult))
            allWorkerList = list(range(1, workerNumber + 1))
            secondPart = list(set(allWorkerList) - set(thisResult))
            random.shuffle(secondPart)
            attrInte[name] = thisResult + secondPart

        attrInteFile = "file_" + str(fileId) + "/" + "TFARworkerOrder/attrInte_"+ str(thisround-1)  + ".csv"
        attrInte.to_csv(attrInteFile,index=False)
        crossInteFile = "file_" + str(fileId) + "/" + "TFARworkerOrder/crossInte_" + str(thisround - 1) + ".csv"
        crossInte.to_csv(crossInteFile, index=False)

        return attrInte
                # 交叉混合，比如都是在1号位，那么就放在1~10的位置，剩余的workers就打乱了放在后面
    else:
        # 这个时候结合历史看看能不能混合，不能混合就对最新的同上面操作。
        # 如果有可以混合的，那么就混合起来。
        # 现在的这个版本也交叉混合，过去的版本也交叉混合，然后用merge算法形成一个最新的交叉混合版本。
        RelLocOrder = pd.DataFrame()
        attrInte = pd.DataFrame()
        crossInte = pd.DataFrame()
        lastround = thisround - 1  # 上一轮的交叉混合需要这边后面才能知道，现在只能直接带入的是前一轮的交叉混合
        lastcrossInteFile = "TFARworkerOrder/crossInte_" + str(lastround - 1) + ".csv"
        lastcrossInte = pd.read_csv(lastcrossInteFile)
        for name in attriNameList:
            thisAttr = []
            for patientid in range(1,1+locationNumber):
                RelAttrOrderFile = "TFARworkerOrder/workerRelativeAttr_" + str(thisround-1) + "_" + str(patientid) + ".csv"
                RelAttrOrder = pd.read_csv(RelAttrOrderFile)
                if (RelAttrOrder[str(attriNameList.index(name))] == 0).all():
                    continue
                else:
                    order = list(RelAttrOrder[str(attriNameList.index(name))])
                    order = [aaa for aaa in order if aaa != 0]
                    thisAttr.append(order)
            thisResult = []
            thisAttr_c = copy.deepcopy(thisAttr)
            whileNumber = len(thisAttr_c)
            while whileNumber > 0 :
                for i in range(len(thisAttr_c)):
                    if len(thisAttr[i]) > 1:
                        thisResult.append(thisAttr[i][0])
                        thisAttr[i] = thisAttr[i][1:]
                    elif len(thisAttr[i]) == 1:
                        thisResult.append(thisAttr[i][0])
                        thisAttr[i] = []
                        whileNumber = whileNumber - 1
                    else:
                        continue
            crossInte[name] = thisResult + [0] * (workerNumber - len(thisResult))

        for name in attriNameList:
            relOrder = lastcrossInte[name].tolist()
            zeroIndex = relOrder.index(0)
            relOrder = relOrder[:zeroIndex]

            absOrder = crossInte[name].tolist()
            zeroIndex = absOrder.index(0)
            absOrder = absOrder[:zeroIndex]

            common_list = list(set(relOrder).intersection(absOrder))
            allWorkerList = list(range(1, workerNumber + 1))
            if len(common_list) == 0:  # 没有交集的话就不连接了
                combine = absOrder
                secondPart = list(set(allWorkerList) - set(combine))
                random.shuffle(secondPart)
                attrInte[name] = combine + secondPart
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
                attrInte[name] = combine + secondPart
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
                attrInte[name] = combine + secondPart

        attrInteFile = "TFARworkerOrder/attrInte_" + str(thisround - 1) + ".csv"
        attrInte.to_csv(attrInteFile, index=False)
        crossInteFile = "TFARworkerOrder/crossInte_" + str(thisround - 1) + ".csv"
        crossInte.to_csv(crossInteFile, index=False)
        return attrInte











