import pandas as pd

if __name__ == '__main__':
    fileNumber = 20
    MLMAverage = []
    compareAverage = []
    randomAverage = []
    for fileId in range(1,fileNumber + 1):
        MLMFile = "Accuracy/fee_"+ str(fileId) +".csv"
        MLM = pd.read_csv(MLMFile)
        MLM = MLM.T
        MLM.insert(0, 'Round', range(1, 1 + len(MLM)))
        MLM.columns = ['Round','MLM-WR']
        MLMAverage.append(MLM)

        compareFile = "Accuracy/TFARfee_"+ str(fileId) +".csv"
        compare = pd.read_csv(compareFile)
        compare = compare.T
        compare.insert(0, 'Round', range(1, 1 + len(compare)))
        compare.columns = ["Round", "TFGR+TFAR"]
        compareAverage.append(compare)

        randomFile = "Accuracy/Randomfee_"+ str(fileId) +".csv"
        random = pd.read_csv(randomFile)
        random = random.T
        random.insert(0, 'Round', range(1, 1 + len(random)))
        random.columns = ["Round", "Random"]
        randomAverage.append(random)

    MLMResult = pd.concat(MLMAverage).groupby(level=0).mean()
    MLMResult = MLMResult.sort_values(by=['Round'])
    compareResult = pd.concat(compareAverage).groupby(level=0).mean()
    compareResult = compareResult.sort_values(by=['Round'])
    randomResult = pd.concat(randomAverage).groupby(level=0).mean()
    randomResult = randomResult.sort_values(by=['Round'])
    result = compareResult.merge(randomResult, on="Round",how='left')
    result = result.merge(MLMResult, on="Round",how='left')
    filename = "Average/fee_merge.csv"
    result.to_csv(filename, index=False)

