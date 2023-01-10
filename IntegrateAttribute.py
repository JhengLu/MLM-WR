import copy

import pandas as pd
import numpy as np
import random
import math

from tqdm import tqdm

def AttrTrust_PSO_Thread(workernumber,workerInfo,roundtime,HpatientId,thisTask,fileId):

    # create Init
    thisTask = thisTask.reset_index(drop=True)
    nonZeroTask = thisTask.loc[:, (thisTask != 0).any(axis=0)]
    nonZeroName = list(nonZeroTask.columns)[1::]  # 注意要把patientId给排除了
    attriName = list(thisTask.columns)[1::]
    ZeroName = list(set(attriName) - set(nonZeroName))
    adjTask = copy.deepcopy(thisTask)
    adjTask[adjTask > 1] = 1
    adjTask = adjTask.reset_index(drop=True)
    filtered_data = pd.DataFrame(columns=('PatientId', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'source'))
    for index, row in workerInfo.iterrows():
        workerId = row['workerId']
        workerFile = "file_" + str(fileId) + "/" + "workerData/round_" + str(roundtime) + "/worker_" + str(workerId) + ".csv"
        thisData = pd.read_csv(workerFile)
        tempData = thisData.loc[(thisData['PatientId'] == HpatientId)]
        tempData = tempData.reset_index(drop=True)
        tempData = tempData.mul(adjTask)
        tempData['source'] = workerId
        filtered_data = filtered_data.append(tempData)
    init_data = filtered_data.iloc[:, :-1].mean()  # except the source column
    init_data = init_data.to_frame().T
    # create init end

    # create my strat
    all_data = filtered_data
    hard_dict = {}
    easy_dict = {}
    old_dis = -1000
    filtered_number = workerInfo.shape[0]
    source = workerInfo['workerId']
    weight = np.zeros(((workernumber + 1), len(attriName)), dtype=float)
    for name in attriName:
        if name in nonZeroName:
            count_hard = 0
            count_easy = 0
            temp = all_data.loc[all_data['PatientId'] == HpatientId]
            result_number = temp[name].value_counts().max()
            if result_number > (len(all_data) / 2):  # 如果对于一个数出现的频率大于采集worker数量的一半，则认为是easy
                count_easy += 1
            else:
                count_hard += 1
            hard_dict[name] = count_hard
            easy_dict[name] = count_easy

    vice_truth = init_data  # 是用来对比前后两次的变化
    acciCount = 0
    while (True):
        if acciCount>20:
            break
        acciCount = acciCount + 1
        fenzi = 0
        for indexi, ki in source.items():
            predict_df = all_data.loc[(all_data['source'] == ki)].iloc[:, :-1]
            predict_df = predict_df.reset_index(drop=True)
            subtract_df = predict_df.subtract(init_data).abs()
            fenzi += subtract_df.to_numpy().sum()

        for index, k in source.items():  # 来源id名字   更新权重
            for n in range(len(attriName)):  # 属性个数
                if attriName[n] in nonZeroName:  # 只检验平台关注的属性
                    fenmu = 0
                    predict = all_data.loc[all_data['source'] == k][attriName[n]].values[0]  # v- n'm' - k'
                    answer = init_data.iat[0,n]  # 因为只是一个series
                    fenmu += abs(predict - answer)
                    weight[k][n] = math.log(fenzi / fenmu)
                    weight[k][n] *= (1 + hard_dict[attriName[n]]) / (1 + easy_dict[attriName[n]])


        my_weight = pd.DataFrame(weight, columns=('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI'))


        predict = np.zeros(len(attriName), dtype=float)

        for n in range(len(attriName)):  # 属性个数
            if attriName[n] in nonZeroName:
                fenmu = 0
                value_df = copy.deepcopy(all_data)
                value_df = value_df.sort_values(by=['source']).iloc[:,1:]  # ascend get rid of patientId
                value_attri_df = value_df.iloc[:,n]
                value_attri_df = value_attri_df.reset_index(drop=True)

                weight_df = pd.DataFrame(weight)
                weight_df = weight_df[(weight_df.T != 0).any()]
                weight_attri_df = weight_df.iloc[:,n]
                weight_attri_df = weight_attri_df.reset_index(drop=True)
                mul_attri_df = value_attri_df.mul(weight_attri_df)

                fenmu = weight_attri_df.to_numpy().sum()
                predict_n = mul_attri_df.to_numpy().sum()
                predict[n] = predict_n/fenmu
            else:
                predict[n] = 0

        df = pd.DataFrame(predict)
        df = df.T
        df.insert(0, 'PatientId', HpatientId, allow_duplicates=False)
        df.columns = ['PatientId', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
        init_data = df
        data = df

        temp = data.subtract(vice_truth).abs()  #当前态和过去态进行对比
        judgeDf = temp.div(vice_truth)
        judgeDf = judgeDf.fillna(0)  # 这个是因为对于任务不关注的属性，值会是0
        numberDf = judgeDf[judgeDf < 0.03]
        number = np.sum(numberDf.count()) - 1# 除去patientid
        if number > len(attriName) - 2:
            break
        vice_truth = data

    my_weight = pd.DataFrame(weight)
    my_weight.insert(0, 'workerId', range(0, 0 + len(my_weight)))
    my_weight.columns = ['workerId', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
    my_weight = my_weight.iloc[1:, :]  # 把worker 0给去除掉
    my_weight['rand_col'] = np.random.randint(1, 1 + len(my_weight), my_weight.shape[0])
    result_weight = pd.DataFrame()
    for name in attriName:
        temp_weight = my_weight.sort_values(by=[name, "rand_col"],ascending=[False, False])  # to make the index ascend
        temp_weight = temp_weight.reset_index(drop=True)  # 这个不能少，不然会每一列都一样了
        temp_weight = temp_weight[['workerId', name]]
        temp_weight.columns = ['workerId','value']
        temp_weight.loc[temp_weight.value == 0, 'workerId'] = 0
        result_weight[name] = copy.deepcopy(temp_weight['workerId'])

    return result_weight












