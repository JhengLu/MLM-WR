import copy

import pandas as pd
import numpy as np
import random
import math

from tqdm import tqdm


def change(x, y):
    return x + y

def LocTrust_PSO_Thread(workernumber,workerInfo,roundtime,HpatientId,thisTask,fileId): # 对于一个地方的进行评价
    pd.set_option('mode.chained_assignment', None)

    # create Init
    thisTask = thisTask.reset_index(drop=True)
    nonZeroTask = thisTask.loc[:, (thisTask != 0).any(axis=0)]
    nonZeroName = list(nonZeroTask.columns)[1::]  # 注意要把patientId给排除了
    attriName = list(thisTask.columns)[1::]
    ZeroName = list(set(attriName) - set(nonZeroName))
    adjTask = copy.deepcopy(thisTask)
    adjTask[adjTask>1] = 1
    # adjTask = thisTask[thisTask > 1] = 1 # 使得task中的第一项变成1
    filtered_data = pd.DataFrame(columns=('PatientId','Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI','source'))
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
    # create init end

    # create my start

    weight = np.zeros((workernumber + 1), dtype=float) # 这边一定要再重新生成，不然weight_truth和weight就是指向的是一个对象
    all_data = filtered_data
    filtered_worker = workerInfo
    old_dis = -1000
    new_distance = 0
    weight_g = None
    column_name = list(all_data)  # delete source and patientid
    column_name = column_name[1:-1]
    source = filtered_worker['workerId']

    vice_truth = init_data  # 是用来对比前后两次的变化
    while (True):
        fenzi = 0
        for indexi, ki in source.items():
            predict_df = all_data.loc[(all_data['source'] == ki)].iloc[:, :-1]
            predict_df = predict_df.reset_index(drop=True)
            subtract_df = predict_df.subtract(init_data).abs()
            fenzi += subtract_df.to_numpy().sum()

        for index, k in tqdm(source.items()):  # 来源id名字   更新权重
            predict_loc_df = all_data.loc[(all_data['source'] == k)].iloc[:, :-1]
            predict_loc_df = predict_loc_df.reset_index(drop=True)
            subtract_loc_df = predict_loc_df.subtract(init_data).abs()
            fenmu_df = subtract_loc_df.sum(axis = 1)
            fenmu_value = fenmu_df.values[0]
            # weight_k = ( fenzi / fenmu_df ).apply(math.log).to_numpy().flatten()
            weight_k = ( fenzi / fenmu_df ).apply(math.log)
            weight[k] = weight_k




        predict = np.zeros(len(column_name), dtype=float)
        weight_df = pd.DataFrame(weight)
        weight_df = weight_df[(weight_df.T != 0).any()]
        weight_loc_df = copy.deepcopy(weight_df)
        weight_loc_df = weight_loc_df.reset_index(drop=True).iloc[:,0]
        fenmu = weight_loc_df.to_numpy().sum()
        for n in range(len(column_name)):
            value_df = copy.deepcopy(all_data)
            value_df = value_df.sort_values(by=['source']).iloc[:, 1:]  # ascend get rid of patientId
            value_loc_df = value_df.iloc[:, n]
            value_loc_df = value_loc_df.reset_index(drop=True)

            mul_loc_df = value_loc_df.mul(weight_loc_df)
            predict_m_n = mul_loc_df.to_numpy().sum()
            predict[n] = predict_m_n / fenmu


        df = pd.DataFrame(predict)
        df = df.T
        df.insert(0, 'PatientId', HpatientId, allow_duplicates=False)
        df.columns = ['PatientId', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
        init_data = df
        data = df


        temp = data.subtract(vice_truth).abs()  # 当前态和过去态进行对比
        judgeDf = temp.div(vice_truth)
        judgeDf = judgeDf.fillna(0)  # 这个是因为对于任务不关注的属性，值会是0
        numberDf = judgeDf[judgeDf < 0.01]
        number = np.sum(numberDf.count()) - 1
        if number > len(attriName) - 2:
            break
        vice_truth = data


    my_weight = pd.DataFrame(weight)
    my_weight.insert(0, 'workerId', range(0, 0 + len(my_weight)))
    my_weight.columns = ['workerId','value']
    my_weight = my_weight.iloc[1:,:] #把worker 0给去除掉
    my_weight['rand_col'] = np.random.randint(1, 1 + len(my_weight), my_weight.shape[0])
    result_weight = my_weight.sort_values(by=['value', "rand_col"],ascending=[False, False])  # to make the index ascend
    result_weight.loc[result_weight.value == 0, 'workerId'] = 0

    return result_weight






