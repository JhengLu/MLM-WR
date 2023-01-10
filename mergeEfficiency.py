import pandas as pd

if __name__ == '__main__':
    fileNumber = 20
    compareAverage = []
    randomAverage = []
    MLMAverage = []

    for fileId in range(1,fileNumber + 1):
        compareFile = "error/" + "errorAll_compare" + str(fileId) + ".csv"
        compare = pd.read_csv(compareFile)
        compare.columns = ["Round","TFGR+TFAR"]
        compareAverage.append(compare)

        randomFile = "error/" + "errorAll_random" + str(fileId) + ".csv"
        random = pd.read_csv(randomFile)
        random.columns = ["Round","random"]
        randomAverage.append(random)

        MLMFile = "error/" + "errorAll_" + str(fileId) + ".csv"
        MLM = pd.read_csv(MLMFile)
        MLM.columns = ["Round", "MLM-WR"]
        MLMAverage.append(MLM)

    MLMResult = pd.concat(MLMAverage).groupby(level=0).mean()
    compareResult = pd.concat(compareAverage).groupby(level=0).mean()
    randomResult = pd.concat(randomAverage).groupby(level=0).mean()
    result = compareResult.merge(randomResult,on="Round")
    result = result.merge(MLMResult,on="Round")
    # result = MLMResult
    filename = "Average/errorAll_merge.csv"
    result.to_csv(filename, index=False)

