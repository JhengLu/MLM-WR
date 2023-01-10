import os
import parameter
if __name__ == '__main__':
    worker_number = parameter.workerNumber
    task_number = parameter.taskNumber
    round_number = 20
    punish_co = 2
    # root_path = 'PSO_workers_' + str(worker_number)
    # root_path = 'PSO_workers_' + str(worker_number) + '_' + judge
    # root_path = "PSO_particle_" + str(worker_number) + "_task_" + str(task_number) + "_pun_" + str(punish_co) + "_arrange"
    # root_path = "Record"
    # root_path = "TFARRecord"

    # #生成最底层的file文件
    # root_path = ""
    # list = []
    # folderNumber = 20
    # for i in range(1, round_number + 1):
    #     foldername = "file_"+ str(i)
    #     list.append(foldername)
    #
    # for items in list:
    #     path = os.path.join(root_path, items)
    #     os.mkdir(path)
    # #生成最底层的file文件

    folderNumber = 20
    for folder in range(1,1 + folderNumber):
        root_path = "file_" + str(folder) + "/TFARRecord"
        list = []
        for i in range(1,round_number + 1):
            foldername = "round_"+ str(i)
            list.append(foldername)

        # for i in range(0, round_number + 1):
        #     foldername = "file_"+ str(i)
        #     list.append(foldername)

        # for i in range(0, 1):
        #     foldername =  "RoundTruth"
        #     list.append(foldername)

        # for roundtime in range(1,round_number + 1):
        #     foldername = "round_" + str(roundtime)
        #     list.append(foldername)

        for items in list:
            path = os.path.join(root_path, items)
            os.mkdir(path)