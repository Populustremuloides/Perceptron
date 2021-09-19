import pandas as pd
import numpy as np
import copy

def getHeaderInfo(file):
    columns = []

    # grab the column information
    with open(file) as ifile:
        i = 0
        for line in ifile:
            line = line.lower()
            if line.startswith("@attribute"):
                line = line.replace("@attribute ", "")
                line = line.replace("\n","")
                columns.append(line)
            elif line.startswith("@data"): # save the location of the start index
                dataStart = i
                break
            i = i + 1

    return columns, dataStart

def getData(file, columns, dataStart):
    # prep the data structure
    columnsToData = {}
    for col in columns:
        columnsToData[col] = []

    # parse the data
    with open(file) as ifile:
        i = 0
        for line in ifile:
            if i > dataStart:
                line = line.replace("\n","")
                line = line.split(",")
                for j in range(len(line)):
                    datum = line[j]
                    column = columns[j]
                    columnsToData[column].append(datum)
            i = i + 1

    return columnsToData

def convertToNumbers(df):
    for col in df.columns:
        # check if it is numeric or alphabetical
        sample = str(df[col][0])
        sample = sample.replace("'","")

        if sample.isnumeric():
            numberedCol = []
            for item in df[col]:
                numberedCol.append(float(item))
            df[col] = numberedCol
        elif sample.isalpha():
            # get all the possible types
            types = list(set(df[col]))
            types.sort()
            numbers = range(len(types))
            typeToNumber = dict(zip(types, numbers))
            numberedCol = [typeToNumber[x] for x in list(df[col])]
            df[col] = numberedCol
        else:
            print("ERROR in Arff.py")

    return df

def readArff(file):
    columns, dataStart = getHeaderInfo(file)
    columnsToData = getData(file, columns, dataStart)
    df = pd.DataFrame.from_dict(columnsToData)
    df = convertToNumbers(df)
    return df

def splitTrainTest(df, seed):
    df = copy.copy(df)
    np.random.seed(seed)

    indices = np.asarray(list(range(len(df[df.columns[0]]))))
    df["indices"] = indices
    np.random.shuffle(indices)

    numIndices = len(indices)
    cutoffIndex = round(0.8 * numIndices)

    train = indices[:cutoffIndex]
    test = indices[cutoffIndex:]

    trainDf = df[df["indices"].isin(train)]
    testDf = df[df["indices"].isin(test)]

    trainDf = trainDf.drop("indices", axis=1)
    testDf = testDf.drop("indices", axis=1)

    return trainDf, testDf
